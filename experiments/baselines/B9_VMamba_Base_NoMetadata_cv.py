"""
B9: VMamba Base plus two Mamba fusion blocks without metadata.

This is the dedicated reproduction entry point for paper baseline B9. The old
A6 script could instantiate this architecture, but mixed two backbone choices
under one ablation output directory and did not identify B9 explicitly.

Run inside WSL with the project Conda environment:
    BIOMASS_DATA_DIR=/path/to/csiro-biomass \
    conda run --no-capture-output -n mambahar \
    python experiments/baselines/B9_VMamba_Base_NoMetadata_cv.py
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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from engine import load_train_data, run_cv, weighted_r2_score
from models import BiomassModelVMamba


DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT / "csiro-biomass"))
RUN_ROOT = PROJ_ROOT / "output" / "reruns_2026_09_13"
OUTPUT_ROOT = PROJ_ROOT / "output" / "metadata_repair_2026_09_13"


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / "train.csv")
    TRAIN_IMAGE_DIR = str(DATA_DIR / "train")
    FOLD_FILE = str(RUN_ROOT / "folds_seed17.csv")

    MODEL_NAME = "vmamba_base"
    VMAMBA_VARIANT = "vmamba_base"
    MODEL_DIR = str(OUTPUT_ROOT / "B9_VMamba_Base_Mamba_NoMeta")
    OUTPUT_DIR = MODEL_DIR

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]
    CV_STRATEGY = "stratified_group"
    USE_METADATA = False

    IMG_SIZE = 512
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 2
    NUM_WORKERS = 2
    EVAL_BATCH_SIZE = 16
    EVAL_NUM_WORKERS = 0

    EPOCHS = 50
    WARMUP_EPOCHS = 5
    LR_BACKBONE = 1e-5
    LR_HEAD = 5e-4
    WD = 1e-2
    CLIP_GRAD_NORM = 1.0
    DROPOUT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    LOSS_FN = "huber"
    HUBER_BETA = 5.0
    VAL_EVERY_N_EPOCHS = 1
    TARGET_COLS = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]


def model_factory(cfg):
    return BiomassModelVMamba(
        pretrained_path="",
        dropout=cfg.DROPOUT,
        use_mamba_ssm=True,
        use_metadata=False,
        meta_input_dim=0,
        variant=cfg.VMAMBA_VARIANT,
    )


class CommonEvaluatorDataset(Dataset):
    """Use the image decoding and preprocessing of the unified evaluator."""

    def __init__(self, df, image_dir, target_cols, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img = np.array(Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB"))
        width = img.shape[1]
        left = img[:, :width // 2]
        right = img[:, width // 2:]

        left = self.transform(image=left)["image"]
        right = self.transform(image=right)["image"]
        targets = row[self.target_cols].values.astype(np.float32)
        return left, right, torch.tensor(targets, dtype=torch.float32)


def get_common_eval_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_fixed_folds():
    fold_path = Path(CFG.FOLD_FILE)
    if not fold_path.exists():
        raise FileNotFoundError(f"Fixed fold file is missing: {fold_path}")

    df = load_train_data(CFG)
    fixed = pd.read_csv(fold_path)[["image_id", "fold"]]
    comparison = df[["image_id", "fold"]].merge(
        fixed,
        on="image_id",
        how="outer",
        suffixes=("_generated", "_fixed"),
        indicator=True,
    )
    mismatch = comparison[
        (comparison["_merge"] != "both")
        | (comparison["fold_generated"] != comparison["fold_fixed"])
    ]
    if not mismatch.empty:
        raise ValueError(
            f"Fixed fold verification failed for {len(mismatch)} image IDs."
        )
    print(f"Fixed fold verification passed for {len(comparison)} image IDs.")
    return df


def main():
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    start_time = time.time()

    print(f"Verifying fixed fold split from {CFG.FOLD_FILE}...")
    df = verify_fixed_folds()

    CFG._CACHE_DIR = f"/tmp/biomass_cache_{CFG.IMG_SIZE}"

    print("\nStarting B9 VMamba-Base 5-Fold Cross-Validation...")
    # run_cv handles training or resume only. Metrics reported below come from the common evaluator protocol.
    run_cv(CFG, model_factory, use_compile=True)

    print("\nCollecting out-of-fold predictions across all 5 folds...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_preds = np.zeros((len(df), 5), dtype=np.float32)
    oof_targets = np.zeros((len(df), 5), dtype=np.float32)
    fold_details = []

    for fold in range(CFG.N_FOLDS):
        val_idx = df[df["fold"] == fold].index.values
        val_data = df.iloc[val_idx].reset_index(drop=True)
        val_ds = CommonEvaluatorDataset(
            val_data,
            CFG.TRAIN_IMAGE_DIR,
            CFG.TARGET_COLS,
            get_common_eval_transform(CFG.IMG_SIZE),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=CFG.EVAL_NUM_WORKERS,
        )

        best_path = os.path.join(CFG.MODEL_DIR, f"fold{fold}_best.pth")
        done_path = os.path.join(CFG.MODEL_DIR, f"fold{fold}_done.json")
        best_epoch = -1
        if os.path.exists(done_path):
            with open(done_path) as f:
                best_epoch = json.load(f).get("best_epoch", -1)

        model = model_factory(CFG).to(device)
        raw_state = torch.load(best_path, map_location="cpu")
        if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
            state = raw_state["model_state_dict"]
        elif isinstance(raw_state, dict) and "state_dict" in raw_state:
            state = raw_state["state_dict"]
        else:
            state = raw_state
        model.load_state_dict(state, strict=True)
        model.eval()

        fold_p = []
        fold_y = []
        with torch.inference_mode():
            for left, right, labels in val_loader:
                left = left.to(device)
                right = right.to(device)
                out = model(left, right)
                fold_p.append(out.cpu().float().numpy())
                fold_y.append(labels.numpy())

        fold_p = np.concatenate(fold_p, axis=0)
        fold_y = np.concatenate(fold_y, axis=0)
        oof_preds[val_idx] = fold_p
        oof_targets[val_idx] = fold_y

        r2_w, r2_per = weighted_r2_score(fold_y, fold_p)
        fold_details.append({
            "fold": fold,
            "selected_epoch": best_epoch,
            "outer_weighted_r2": float(r2_w),
            "outer_per_target_r2": {col: float(v) for col, v in zip(CFG.TARGET_COLS, r2_per)},
            "train_samples": int((df["fold"] != fold).sum()),
            "val_samples": int((df["fold"] == fold).sum()),
            "checkpoint_sha256": sha256_file(best_path),
        })

    pooled_w, pooled_per = weighted_r2_score(oof_targets, oof_preds)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    summary = {
        "status": "completed",
        "model_identifier": "B9_VMamba_Base_Mamba_NoMeta",
        "backbone": CFG.MODEL_NAME,
        "protocol": "matched dual-view, 2x Mamba fusion, compositional output, raw target Huber",
        "evaluation_protocol": "unified common evaluator: PIL RGB, INTER_LINEAR, batch 16, workers 0, full precision",
        "seed": CFG.SEED,
        "config": {
            "img_size": CFG.IMG_SIZE,
            "batch_size": CFG.BATCH_SIZE,
            "evaluation_batch_size": CFG.EVAL_BATCH_SIZE,
            "grad_accum_steps": CFG.GRAD_ACCUM_STEPS,
            "epochs": CFG.EPOCHS,
            "warmup_epochs": CFG.WARMUP_EPOCHS,
            "lr_backbone": CFG.LR_BACKBONE,
            "lr_head": CFG.LR_HEAD,
            "weight_decay": CFG.WD,
            "huber_beta": CFG.HUBER_BETA,
        },
        "fold_results": fold_details,
        "mean_fold_weighted_r2": float(np.mean([f["outer_weighted_r2"] for f in fold_details])),
        "fold_dispersion_std": float(np.std([f["outer_weighted_r2"] for f in fold_details])),
        "pooled_oof_weighted_r2": float(pooled_w),
        "pooled_oof_per_target_r2": {col: float(v) for col, v in zip(CFG.TARGET_COLS, pooled_per)},
        "dataset_train_csv_sha256": sha256_file(CFG.TRAIN_CSV),
        "fold_file_sha256": sha256_file(CFG.FOLD_FILE),
        "runtime_seconds": float(time.time() - start_time),
        "peak_vram_gb": float(peak_vram_gb),
    }

    summary_path = os.path.join(CFG.MODEL_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(CFG.MODEL_DIR, "oof_predictions.npz"),
        predictions=oof_preds,
        targets=oof_targets,
        image_ids=df["image_id"].values,
        fold_assignments=df["fold"].values,
    )

    print("\n" + "=" * 60)
    print("B9 VMAMBA-BASE RERUN COMPLETE!")
    print(f"Mean Fold Weighted R²: {summary['mean_fold_weighted_r2']:.4f} ± {summary['fold_dispersion_std']:.4f}")
    print(f"Pooled OOF Weighted R²: {summary['pooled_oof_weighted_r2']:.4f}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
