"""A7: Metadata-Only MLP Diagnostic Baseline.

Quantifies how predictive tabular metadata features are in isolation (without images).
Uses the exact 23 encoded metadata features from src/engine.py:
  - State (one-hot, 4 dims)
  - Species (one-hot, 15 dims)
  - Pre_GSHH_NDVI (continuous, ~[0,1])
  - Height_Ave_cm (continuous, normalized /100)
  - Sampling month (cyclical sin/cos, 2 dims)

Protocol:
  - 5 fixed seeds: [17, 29, 43, 59, 71].
  - 5 fixed outer folds from folds_seed17.csv.
  - Inner 80/20 train/val split for epoch selection (no outer fold epoch selection).
  - Retrain on full outer train split for selected epoch count, then evaluate outer fold once.
  - Output: output/metadata_repair_2026_09_13/A7_Metadata_Only_MLP.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

import engine
load_train_data = engine.load_train_data
weighted_r2_score = engine.weighted_r2_score


class A7Config:
    DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT / "csiro-biomass"))
    TRAIN_CSV = str(DATA_DIR / "train.csv")
    FOLD_FILE = str(PROJ_ROOT / "output" / "reruns_2026_09_13" / "folds_seed17.csv")
    OUTPUT_DIR = str(PROJ_ROOT / "output" / "metadata_repair_2026_09_13" / "A7_Metadata_Only_MLP")

    SEED = 17
    SEEDS = [17, 29, 43, 59, 71]
    N_FOLDS = 5
    USE_METADATA = True

    HIDDEN_DIMS = (128, 64)
    DROPOUT = 0.1
    LR = 1e-3
    WEIGHT_DECAY = 1e-3
    BATCH_SIZE = 32
    MAX_EPOCHS = 150
    PATIENCE = 20
    HUBER_BETA = 5.0
    TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


class MetadataOnlyMLP(nn.Module):
    def __init__(self, in_dim=23, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_green = nn.Sequential(nn.Linear(hidden_dims[1], 1), nn.Softplus())
        self.head_dead = nn.Sequential(nn.Linear(hidden_dims[1], 1), nn.Softplus())
        self.head_clover = nn.Sequential(nn.Linear(hidden_dims[1], 1), nn.Softplus())

    def forward(self, x):
        h = self.net(x)
        green = self.head_green(h)
        dead = self.head_dead(h)
        clover = self.head_clover(h)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=-1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_epoch(model, x, y, optimizer, batch_size, beta=5.0):
    model.train()
    n = len(x)
    indices = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        b_idx = indices[i : i + batch_size]
        bx, by = x[b_idx], y[b_idx]

        optimizer.zero_grad()
        pred = model(bx)
        loss = nn.functional.smooth_l1_loss(pred, by, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    pred = model(x).cpu().numpy()
    target = y.cpu().numpy()
    weighted, per_target = weighted_r2_score(target, pred)
    return weighted, per_target, pred


def run_a7():
    os.makedirs(A7Config.OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    print(f"Loading data from {A7Config.TRAIN_CSV}...")
    df = load_train_data(A7Config)
    meta_cols = A7Config._META_COLS
    assert len(meta_cols) == 23, f"Expected 23 metadata features, got {len(meta_cols)}"

    folds_df = pd.read_csv(A7Config.FOLD_FILE)
    df = df.drop(columns=["fold"]).merge(folds_df[["image_id", "fold"]], on="image_id")
    assert len(df) == 357, f"Expected 357 rows, got {len(df)}"

    print(f"Metadata features ({len(meta_cols)}): {meta_cols}")
    with open(os.path.join(A7Config.OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(meta_cols, f, indent=2)

    X_all = df[meta_cols].values.astype(np.float32)
    Y_all = df[A7Config.TARGET_COLS].values.astype(np.float32)
    image_ids = df["image_id"].values
    fold_assignments = df["fold"].values

    sample_model = MetadataOnlyMLP(in_dim=len(meta_cols), hidden_dims=A7Config.HIDDEN_DIMS)
    total_params = sum(p.numel() for p in sample_model.parameters())
    print(f"A7 MetadataOnlyMLP parameter count: {total_params}")

    seed_results = []
    all_seed_oof_preds = np.zeros((len(A7Config.SEEDS), len(df), 5), dtype=np.float32)

    for s_idx, seed in enumerate(A7Config.SEEDS):
        print(f"\n{'='*50}\nRUNNING SEED {seed} ({s_idx+1}/{len(A7Config.SEEDS)})\n{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        oof_preds = np.zeros_like(Y_all)
        fold_scores = []
        fold_details = []

        for fold in range(A7Config.N_FOLDS):
            outer_train_mask = (fold_assignments != fold)
            outer_val_mask = (fold_assignments == fold)

            X_outer_tr = torch.tensor(X_all[outer_train_mask])
            Y_outer_tr = torch.tensor(Y_all[outer_train_mask])
            X_outer_val = torch.tensor(X_all[outer_val_mask])
            Y_outer_val = torch.tensor(Y_all[outer_val_mask])

            # Inner 80/20 train/validation split for epoch selection.
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            in_tr_idx, in_val_idx = next(inner_kf.split(X_outer_tr))

            X_in_tr, Y_in_tr = X_outer_tr[in_tr_idx], Y_outer_tr[in_tr_idx]
            X_in_val, Y_in_val = X_outer_tr[in_val_idx], Y_outer_tr[in_val_idx]

            # 1. Inner validation training to find best epoch.
            inner_model = MetadataOnlyMLP(in_dim=23, hidden_dims=A7Config.HIDDEN_DIMS, dropout=A7Config.DROPOUT)
            inner_opt = optim.AdamW(inner_model.parameters(), lr=A7Config.LR, weight_decay=A7Config.WEIGHT_DECAY)

            best_inner_r2 = -float("inf")
            best_epoch = 1
            patience_ctr = 0

            for ep in range(1, A7Config.MAX_EPOCHS + 1):
                train_epoch(inner_model, X_in_tr, Y_in_tr, inner_opt, A7Config.BATCH_SIZE, beta=A7Config.HUBER_BETA)
                val_r2, _, _ = evaluate(inner_model, X_in_val, Y_in_val)

                if val_r2 > best_inner_r2:
                    best_inner_r2 = val_r2
                    best_epoch = ep
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= A7Config.PATIENCE:
                        break

            # 2. Retrain on full outer train split up to best_epoch.
            final_model = MetadataOnlyMLP(in_dim=23, hidden_dims=A7Config.HIDDEN_DIMS, dropout=A7Config.DROPOUT)
            final_opt = optim.AdamW(final_model.parameters(), lr=A7Config.LR, weight_decay=A7Config.WEIGHT_DECAY)

            for ep in range(best_epoch):
                train_epoch(final_model, X_outer_tr, Y_outer_tr, final_opt, A7Config.BATCH_SIZE, beta=A7Config.HUBER_BETA)

            # 3. Evaluate once on held-out outer validation fold.
            outer_r2, outer_per_target, fold_val_preds = evaluate(final_model, X_outer_val, Y_outer_val)
            oof_preds[outer_val_mask] = fold_val_preds
            fold_scores.append(float(outer_r2))

            fold_details.append({
                "fold": fold,
                "selected_epoch": best_epoch,
                "inner_best_weighted_r2": float(best_inner_r2),
                "outer_weighted_r2": float(outer_r2),
                "outer_per_target_r2": {
                    col: float(val) for col, val in zip(A7Config.TARGET_COLS, outer_per_target)
                },
                "train_samples": int(outer_train_mask.sum()),
                "val_samples": int(outer_val_mask.sum()),
            })
            print(f"Seed {seed} Fold {fold}: Selected Epoch {best_epoch}, Outer Weighted R² = {outer_r2:.4f}")

        # Seed level metrics.
        seed_mean_r2 = float(np.mean(fold_scores))
        seed_fold_disp = float(np.std(fold_scores))
        seed_pooled_r2, seed_pooled_per_target = weighted_r2_score(Y_all, oof_preds)

        all_seed_oof_preds[s_idx] = oof_preds

        seed_results.append({
            "seed": seed,
            "mean_fold_weighted_r2": seed_mean_r2,
            "fold_dispersion_std": seed_fold_disp,
            "pooled_oof_weighted_r2": float(seed_pooled_r2),
            "pooled_oof_per_target_r2": {
                col: float(val) for col, val in zip(A7Config.TARGET_COLS, seed_pooled_per_target)
            },
            "fold_results": fold_details,
        })
        print(f"Seed {seed} Pooled OOF R²: {seed_pooled_r2:.4f}, Mean Fold R²: {seed_mean_r2:.4f}")

    # Across seed aggregation.
    pooled_r2_list = [s["pooled_oof_weighted_r2"] for s in seed_results]
    mean_fold_r2_list = [s["mean_fold_weighted_r2"] for s in seed_results]

    summary = {
        "model_identifier": "A7_Metadata_Only_MLP",
        "protocol": "Tabular metadata only, 2-layer MLP, 5 seeds, inner-val epoch selection",
        "parameter_count": total_params,
        "input_features": meta_cols,
        "feature_dim": len(meta_cols),
        "seeds": A7Config.SEEDS,
        "seeds_summary": {
            "pooled_oof_weighted_r2_mean": float(np.mean(pooled_r2_list)),
            "pooled_oof_weighted_r2_std": float(np.std(pooled_r2_list)),
            "mean_fold_weighted_r2_mean": float(np.mean(mean_fold_r2_list)),
            "mean_fold_weighted_r2_std": float(np.std(mean_fold_r2_list)),
        },
        "per_seed_results": seed_results,
        "dataset_train_csv_sha256": sha256_file(A7Config.TRAIN_CSV),
        "fold_file_sha256": sha256_file(A7Config.FOLD_FILE),
        "runtime_seconds": float(time.time() - start_time),
    }

    summary_path = os.path.join(A7Config.OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(A7Config.OUTPUT_DIR, "oof_predictions.npz"),
        seeds=np.array(A7Config.SEEDS),
        predictions=all_seed_oof_preds,
        targets=Y_all,
        image_ids=image_ids,
        fold_assignments=fold_assignments,
    )

    print("\n" + "=" * 50)
    print("A7 EVALUATION COMPLETE!")
    print(f"Pooled OOF R² across 5 seeds: {summary['seeds_summary']['pooled_oof_weighted_r2_mean']:.4f} ± {summary['seeds_summary']['pooled_oof_weighted_r2_std']:.4f}")
    print(f"Mean Fold R² across 5 seeds: {summary['seeds_summary']['mean_fold_weighted_r2_mean']:.4f} ± {summary['seeds_summary']['mean_fold_weighted_r2_std']:.4f}")
    print(f"Summary saved to {summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_a7()
