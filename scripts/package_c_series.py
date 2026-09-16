#!/usr/bin/env python3
"""
Package classical C-series models into c-series-biomass-models/
following the exact style of checkpoints/B5_DINOv3_L_GatedDWConv.
"""

import os
import sys
import shutil
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
C_SRC_DIR = REPO_ROOT / "output" / "classical_baselines_2026_09_15"
TARGET_DIR = REPO_ROOT / "c-series-biomass-models"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

C_MODELS = {
    "C1": {
        "folder_name": "C1_Metadata_Ridge",
        "title": "C1 (Metadata Multioutput Ridge)",
        "model_type": "Linear Ridge Regression",
        "description": "Multioutput Ridge regression over 23 standardized tabular metadata features with inner 3-fold CV tuning.",
    },
    "C2": {
        "folder_name": "C2_Metadata_XGBoost",
        "title": "C2 (Metadata Independent XGBoost)",
        "model_type": "Gradient Boosted Decision Trees",
        "description": "Three independent XGBoost regressors over 23 one-hot and continuous metadata features with inner 3-fold CV tuning.",
    },
    "C3": {
        "folder_name": "C3_Handcrafted_Image_Ridge",
        "title": "C3 (Handcrafted Image Ridge)",
        "model_type": "Linear Ridge Regression",
        "description": "Multioutput Ridge regression over 652 cross-view handcrafted color, vegetation index, texture, and spatial heterogeneity descriptors.",
    },
    "C4": {
        "folder_name": "C4_Handcrafted_Image_XGBoost",
        "title": "C4 (Handcrafted Image XGBoost)",
        "model_type": "Gradient Boosted Decision Trees",
        "description": "Three independent XGBoost regressors over 652 cross-view handcrafted image descriptors with inner 3-fold CV tuning.",
    },
    "C5": {
        "folder_name": "C5_Handcrafted_Image_Meta_XGBoost",
        "title": "C5 (Handcrafted Image + Metadata XGBoost)",
        "model_type": "Multimodal Decision Trees",
        "description": "Three independent XGBoost regressors over 675 multimodal features (652 handcrafted image + 23 metadata features).",
    },
    "C6": {
        "folder_name": "C6_Frozen_DINOv2_Base_Ridge",
        "title": "C6 (Frozen DINOv2-Base + Ridge)",
        "model_type": "Linear Feature Probe",
        "description": "Multioutput Ridge regression over 3072-dimensional cross-view frozen DINOv2-Base features with inner 3-fold CV tuning.",
    },
    "C7": {
        "folder_name": "C7_Frozen_DINOv3_Large_Ridge",
        "title": "C7 (Frozen DINOv3-Large + Ridge)",
        "model_type": "Linear Feature Probe",
        "description": "Multioutput Ridge regression over 4096-dimensional cross-view frozen DINOv3-Large features (512x512 resolution).",
    },
    "C8": {
        "folder_name": "C8_Frozen_DINOv3_Large_PCA_XGBoost",
        "title": "C8 (Frozen DINOv3-Large + PCA + XGBoost)",
        "model_type": "PCA + Gradient Boosted Decision Trees",
        "description": "PCA dimension reduction (32 components) followed by three independent XGBoost regressors over 4096-dimensional frozen DINOv3-Large features.",
    },
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def copy_or_link(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    print("=================================================================")
    print("Packaging C-Series Models into c-series-biomass-models/")
    print("=================================================================")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Load source CSVs
    fold_metrics_df = pd.read_csv(C_SRC_DIR / "fold_metrics.csv")
    agg_metrics_df = pd.read_csv(C_SRC_DIR / "aggregate_metrics.csv")
    selected_hp_df = pd.read_csv(C_SRC_DIR / "selected_hyperparameters.csv")
    per_target_df = pd.read_csv(C_SRC_DIR / "per_target_metrics.csv")
    oof_df = pd.read_csv(C_SRC_DIR / "oof_predictions.csv")

    manifest = {}
    verified_at = time.strftime("%Y-%m-%d %H:%M:%S")

    target_cols = ["true_Dry_Green_g", "true_Dry_Dead_g", "true_Dry_Clover_g", "true_GDM_g", "true_Dry_Total_g"]
    pred_cols = ["pred_Dry_Green_g", "pred_Dry_Dead_g", "pred_Dry_Clover_g", "pred_GDM_g", "pred_Dry_Total_g"]

    for m_id, info in C_MODELS.items():
        subfolder_name = info["folder_name"]
        dest_dir = TARGET_DIR / subfolder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {m_id}: {subfolder_name}...")

        # 1. Copy fold model checkpoints
        for fold in range(5):
            src_model = C_SRC_DIR / "fold_models" / m_id / f"fold_{fold}_model.joblib"
            dst_best = dest_dir / f"fold{fold}_best.joblib"
            copy_or_link(src_model, dst_best)

            # Also create fold{fold}_done.json
            fold_row = fold_metrics_df[(fold_metrics_df["model"] == m_id) & (fold_metrics_df["fold"] == fold)].iloc[0]
            hp_row = selected_hp_df[(selected_hp_df["model"] == m_id) & (selected_hp_df["outer_fold"] == fold)].iloc[0]

            val_samples = 72 if fold in (0, 1) else 71
            train_samples = 357 - val_samples

            done_data = {
                "fold": fold,
                "best_r2": float(fold_row["weighted_r2"]),
                "selected_config": str(hp_row["selected_candidate_id"]),
                "train_samples": train_samples,
                "val_samples": val_samples,
            }
            done_file = dest_dir / f"fold{fold}_done.json"
            with open(done_file, "w", encoding="utf-8") as f:
                json.dump(done_data, f, indent=2)

        # 2. OOF predictions (NPZ and CSV)
        m_oof = oof_df[oof_df["model"] == m_id].sort_values("image_id").copy()
        csv_oof_path = dest_dir / "oof_predictions.csv"
        m_oof.to_csv(csv_oof_path, index=False)

        npz_oof_path = dest_dir / "oof_predictions.npz"
        np.savez_compressed(
            npz_oof_path,
            predictions=m_oof[pred_cols].values.astype(np.float32),
            targets=m_oof[target_cols].values.astype(np.float32),
            image_ids=m_oof["image_id"].values.astype(str),
            fold_assignments=m_oof["fold"].values.astype(np.int32),
        )

        # 3. Config JSON
        src_cfg = C_SRC_DIR / "configs" / f"{m_id}.json"
        dst_cfg = dest_dir / "config.json"
        copy_or_link(src_cfg, dst_cfg)

        # 4. Training summary JSON
        agg_row = agg_metrics_df[agg_metrics_df["model"] == m_id].iloc[0]

        fold_results = []
        for fold in range(5):
            f_row = fold_metrics_df[(fold_metrics_df["model"] == m_id) & (fold_metrics_df["fold"] == fold)].iloc[0]
            hp_row = selected_hp_df[(selected_hp_df["model"] == m_id) & (selected_hp_df["outer_fold"] == fold)].iloc[0]
            val_samples = 72 if fold in (0, 1) else 71
            train_samples = 357 - val_samples

            f_targets = {
                "Dry_Green_g": float(f_row["r2_green"]),
                "Dry_Dead_g": float(f_row["r2_dead"]),
                "Dry_Clover_g": float(f_row["r2_clover"]),
                "GDM_g": float(f_row["r2_gdm"]),
                "Dry_Total_g": float(f_row["r2_total"]),
            }

            fold_results.append({
                "fold": fold,
                "selected_config": str(hp_row["selected_candidate_id"]),
                "outer_weighted_r2": float(f_row["weighted_r2"]),
                "outer_per_target_r2": f_targets,
                "train_samples": train_samples,
                "val_samples": val_samples,
            })

        pooled_per_target = {
            "Dry_Green_g": float(agg_row["pooled_green_r2"]),
            "Dry_Dead_g": float(agg_row["pooled_dead_r2"]),
            "Dry_Clover_g": float(agg_row["pooled_clover_r2"]),
            "GDM_g": float(agg_row["pooled_gdm_r2"]),
            "Dry_Total_g": float(agg_row["pooled_total_r2"]),
        }

        summary_data = {
            "status": "completed",
            "model_identifier": m_id,
            "title": info["title"],
            "model_type": info["model_type"],
            "description": info["description"],
            "protocol": "Preregistered C-series classical baseline, inner 3-fold CV tuning, locked 5-fold outer test, raw gram targets",
            "seed": 17,
            "mean_outer_weighted_r2": float(agg_row["mean_fold_raw_weighted_r2"]),
            "std_outer_weighted_r2_ddof0": float(agg_row["std_fold_raw_weighted_r2_ddof0"]),
            "pooled_outer_weighted_r2": float(agg_row["pooled_oof_raw_weighted_r2"]),
            "pooled_per_target_r2": pooled_per_target,
            "fold_results": fold_results,
        }

        summary_path = dest_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Record into manifest
        for p in sorted(dest_dir.glob("*")):
            rel_path = f"{subfolder_name}/{p.name}"
            manifest[rel_path] = {
                "model": subfolder_name,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
                "verified_at": verified_at,
            }

        # Create alias symlink C1 -> C1_Metadata_Ridge
        alias_link = TARGET_DIR / m_id
        if alias_link.exists() or alias_link.is_symlink():
            alias_link.unlink()
        try:
            alias_link.symlink_to(dest_dir.name, target_is_directory=True)
        except OSError:
            pass

    # Save manifest
    manifest_path = TARGET_DIR / "checkpoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved checkpoint manifest to {manifest_path} ({len(manifest)} files)")

    # Create root symlink in checkpoints/
    chk_c_series = CHECKPOINTS_DIR / "c-series-biomass-models"
    if chk_c_series.exists() or chk_c_series.is_symlink():
        chk_c_series.unlink()
    try:
        chk_c_series.symlink_to(TARGET_DIR, target_is_directory=True)
        print("Created symlink checkpoints/c-series-biomass-models -> c-series-biomass-models")
    except OSError:
        pass

    # Generate README.md
    readme_content = """# C-Series Biomass Models:

## Overview:

This directory contains trained checkpoints, configurations, and out of fold predictions for the classical C series baseline study (C1 through C8) on the CSIRO Pasture Biomass benchmark.

All models evaluated under locked five fold cross validation with master seed 17.

## Master Performance Summary:

| Model ID | Subfolder Name | Architecture Family | Mean Fold Raw R2 | Pooled OOF Raw R2 |
| :--- | :--- | :--- | :---: | :---: |
| **C1** | `C1_Metadata_Ridge` | Linear Ridge Regression | 0.6827 +/- 0.0312 | 0.6829 |
| **C2** | `C2_Metadata_XGBoost` | Gradient Boosted Decision Trees | 0.7199 +/- 0.0289 | 0.7195 |
| **C3** | `C3_Handcrafted_Image_Ridge` | Linear Ridge Regression | 0.3503 +/- 0.0817 | 0.3433 |
| **C4** | `C4_Handcrafted_Image_XGBoost` | Gradient Boosted Decision Trees | 0.4242 +/- 0.0814 | 0.4202 |
| **C5** | `C5_Handcrafted_Image_Meta_XGBoost` | Multimodal Decision Trees | 0.6150 +/- 0.0527 | 0.6129 |
| **C6** | `C6_Frozen_DINOv2_Base_Ridge` | Linear Feature Probe | 0.6773 +/- 0.0252 | 0.6800 |
| **C7** | `C7_Frozen_DINOv3_Large_Ridge` | Linear Feature Probe | 0.7400 +/- 0.0203 | 0.7399 |
| **C8** | `C8_Frozen_DINOv3_Large_PCA_XGBoost` | PCA + Gradient Boosted Trees | 0.6550 +/- 0.0601 | 0.6509 |

## Folder Structure:

Each model directory contains:
- `fold{0..4}_best.joblib`: fitted model checkpoint for each outer fold.
- `fold{0..4}_done.json`: fold metric and selected configuration metadata.
- `oof_predictions.npz`: compressed predictions, targets, image IDs, and fold assignments.
- `oof_predictions.csv`: tabular out of fold predictions.
- `training_summary.json`: complete performance record and per target breakdowns.
- `config.json`: pipeline specification and candidate hyperparameters.
"""
    with open(TARGET_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\nPackage generation completed successfully!")


if __name__ == "__main__":
    main()
