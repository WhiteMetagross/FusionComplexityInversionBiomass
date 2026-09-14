#!/usr/bin/env python3
"""
Evaluate additional Kaggle checkpoints on the fixed 5-fold split:
- E6_GatedDWConv1x_NoMeta (DINOv3-ViT-L + 1x GatedDWConv, No Metadata)
- E7_GatedDWConv4x_NoMeta (DINOv3-ViT-L + 4x GatedDWConv, No Metadata)
- E5_FullMamba_NoMeta (DINOv3-ViT-L + Full Mamba, No Metadata)
- B6_VMamba_Mamba_Meta (VMamba-Base + 2x Mamba, With Metadata)

Outputs full per-fold metrics, pooled OOF predictions, and JSON summary.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / 'src'))

import engine
_encode_metadata = engine._encode_metadata
weighted_r2_score = engine.weighted_r2_score

from models import BiomassModelTimm, BiomassModelVMamba, GatedDepthwiseConvBlock

TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
EVALUATION_PROTOCOL = 'unified_common_v1'

MODEL_SPECS = {
    'E6_GatedDWConv1x_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'fusion_blocks': 1,
        'use_mamba_ssm': False,
        'ckpt_dir': 'checkpoints/E6_GatedDWConv1x_NoMeta',
    },
    'E7_GatedDWConv4x_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'fusion_blocks': 4,
        'use_mamba_ssm': False,
        'ckpt_dir': 'checkpoints/E7_GatedDWConv4x_NoMeta',
    },
    'E5_FullMamba_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'fusion_blocks': 2,
        'use_mamba_ssm': True,
        'ckpt_dir': 'checkpoints/E5_FullMamba_NoMeta',
    },
    'B9_VMamba_Base_Mamba_NoMeta': {
        'model_type': 'vmamba',
        'use_metadata': False,
        'fusion_blocks': 2,
        'use_mamba_ssm': True,
        'ckpt_dir': 'checkpoints/B9_VMamba_Base_Mamba_NoMeta',
    },
    'A1_SSM_Scale': {
        'model_type': 'vmamba',
        'use_metadata': False,
        'fusion_blocks': 2,
        'use_mamba_ssm': True,
        'ckpt_dir': 'checkpoints/A1_SSM_Scale',
    },
}


def build_eval_model(model_name: str, spec: dict, meta_dim: int):
    mtype = spec['model_type']
    if mtype == 'timm':
        model = BiomassModelTimm(
            model_name='vit_large_patch16_dinov3.lvd1689m',
            dropout=0.2,
            use_mamba_ssm=spec['use_mamba_ssm'],
            pretrained=False,
            use_metadata=spec['use_metadata'],
            meta_input_dim=meta_dim if spec['use_metadata'] else 0,
            img_size=512,
        )
        if spec['fusion_blocks'] == 1:
            nf = model.backbone.num_features
            model.fusion = nn.Sequential(GatedDepthwiseConvBlock(nf, dropout=0.2))
        elif spec['fusion_blocks'] == 4:
            nf = model.backbone.num_features
            model.fusion = nn.Sequential(
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2),
            )
        return model
    elif mtype == 'vmamba':
        model = BiomassModelVMamba(
            pretrained_path='',
            dropout=0.2,
            use_mamba_ssm=spec['use_mamba_ssm'],
            use_metadata=spec['use_metadata'],
            meta_input_dim=meta_dim if spec['use_metadata'] else 0,
            variant='vmamba_base',
        )
        return model
    else:
        raise ValueError(f"Unknown model type {mtype}")


class CommonEvaluatorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, meta_cols: list[str], transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.meta_cols = meta_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self.image_dir / f"{image_id}.jpg"
        img = np.array(Image.open(img_path).convert('RGB'))
        w = img.shape[1]
        left = img[:, :w // 2]
        right = img[:, w // 2:]

        aug_l = self.transform(image=left)['image']
        aug_r = self.transform(image=right)['image']

        meta_vec = torch.tensor(row[self.meta_cols].values.astype(np.float32), dtype=torch.float32)
        target_vec = torch.tensor(row[TARGET_COLS].values.astype(np.float32), dtype=torch.float32)

        return aug_l, aug_r, meta_vec, target_vec, image_id


def evaluate_model_fold(model, loader, device, use_metadata=False):
    model.eval()
    preds = []
    targets = []
    image_ids = []

    with torch.inference_mode():
        for left, right, meta, target, img_id in loader:
            left = left.to(device)
            right = right.to(device)
            meta = meta.to(device) if use_metadata else None

            out = model(left, right, metadata=meta)
            preds.append(out.cpu().numpy())
            targets.append(target.numpy())
            image_ids.extend(img_id)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return preds, targets, image_ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate additional Kaggle models on fixed 5-fold split.")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to evaluate (default: all available)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT.parent / "csiro-biomass"))
    train_csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train"
    fold_file = PROJ_ROOT / "output" / "reruns_2026_09_13" / "folds_seed17.csv"
    output_dir = PROJ_ROOT / "output" / "metadata_repair_2026_09_13" / "additional_kaggle_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_raw = pd.read_csv(train_csv_path)
    df_raw['image_id'] = df_raw['sample_id'].str.split('__').str[0]
    pivoted = df_raw.pivot_table(index=['image_id'], columns='target_name', values='target', aggfunc='first').reset_index()

    folds_df = pd.read_csv(fold_file)[['image_id', 'fold']]
    df_full = pivoted.merge(folds_df, on='image_id', how='left')
    assert df_full['fold'].notna().all(), "Some samples missing fold assignments!"

    df_full, encoded_meta_cols = _encode_metadata(df_raw, df_full)
    print(f"Dataset ready: {len(df_full)} samples, {len(encoded_meta_cols)} metadata features.")

    transform = A.Compose([
        A.Resize(512, 512, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    summary_path = output_dir / "additional_models_evaluation.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    npz_path = output_dir / "additional_models_oof_predictions.npz"
    oof_predictions_dict = {}
    if npz_path.exists():
        existing_npz = np.load(npz_path)
        for k in existing_npz.files:
            if k not in ['targets', 'image_ids', 'fold_assignments']:
                oof_predictions_dict[k] = existing_npz[k]

    models_to_eval = args.models if args.models else list(MODEL_SPECS.keys())

    for model_name in models_to_eval:
        if model_name not in MODEL_SPECS:
            print(f"[ERROR] Unknown model name: {model_name}. Available: {list(MODEL_SPECS.keys())}")
            continue

        spec = MODEL_SPECS[model_name]
        ckpt_dir = PROJ_ROOT / spec['ckpt_dir']
        print(f"\n{'='*60}\nEvaluating model: {model_name} from {ckpt_dir}\n{'='*60}")

        if not ckpt_dir.exists():
            print(f"[SKIP] Checkpoint directory not found: {ckpt_dir}")
            continue

        missing_folds = [f for f in range(5) if not (ckpt_dir / f"fold{f}_best.pth").exists()]
        if missing_folds:
            print(f"[SKIP] Model {model_name} missing folds {missing_folds} in {ckpt_dir}")
            continue

        model = build_eval_model(model_name, spec, len(encoded_meta_cols)).to(device)

        fold_scores = []
        all_preds = np.zeros((len(df_full), 5), dtype=np.float32)
        all_targets = np.zeros((len(df_full), 5), dtype=np.float32)
        fold_details = []

        for fold in range(5):
            ckpt_path = ckpt_dir / f"fold{fold}_best.pth"
            print(f"Loading {ckpt_path.name}...")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=True)

            val_mask = (df_full['fold'] == fold)
            val_df = df_full[val_mask].copy()

            val_dataset = CommonEvaluatorDataset(val_df, image_dir, encoded_meta_cols, transform)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

            preds, targets, img_ids = evaluate_model_fold(model, val_loader, device, use_metadata=spec['use_metadata'])

            w_r2, per_target = weighted_r2_score(targets, preds)
            fold_scores.append(float(w_r2))

            all_preds[val_mask] = preds
            all_targets[val_mask] = targets

            fold_details.append({
                'fold': fold,
                'weighted_r2': float(w_r2),
                'per_target_r2': [float(x) for x in per_target],
            })
            print(f"Fold {fold} Weighted R2: {w_r2:.4f}")

        pooled_w_r2, pooled_per_target = weighted_r2_score(all_targets, all_preds)
        mean_fold_r2 = float(np.mean(fold_scores))
        std_fold_r2 = float(np.std(fold_scores))

        print(f"\n{model_name} Summary:")
        print(f"  Mean Fold Weighted R2: {mean_fold_r2:.4f} +- {std_fold_r2:.4f}")
        print(f"  Pooled OOF Weighted R2: {pooled_w_r2:.4f}")

        results[model_name] = {
            'evaluation_protocol': EVALUATION_PROTOCOL,
            'mean_fold_r2': mean_fold_r2,
            'std_fold_r2': std_fold_r2,
            'pooled_oof_r2': float(pooled_w_r2),
            'pooled_per_target': [float(x) for x in pooled_per_target],
            'fold_results': fold_details,
        }
        oof_predictions_dict[model_name] = all_preds

        # Incrementally save summary and predictions
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        np.savez_compressed(
            npz_path,
            targets=all_targets,
            image_ids=df_full['image_id'].values,
            fold_assignments=df_full['fold'].values,
            **oof_predictions_dict,
        )
        print(f"Incremental results saved to {summary_path} and {npz_path}")

    print(f"\nAll requested evaluations complete.")


if __name__ == '__main__':
    main()
