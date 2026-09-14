#!/usr/bin/env python3
"""
Evaluate metadata modes (present, zero, shuffled) and matched no-metadata controls.

Evaluates the 4 matched architecture pairs across 5 folds:
- E3_GatedDWConv_Meta vs B5_DINOv3_L_GatedDWConv
- B7_CVGA_Meta vs E2_CVGA_NoMeta
- B8_BidirMamba_Meta vs E1_BidirMamba_NoMeta
- E8_Identity_Meta vs E4_Identity_NoMeta

Modes:
- present: true sample metadata
- zero: metadata replaced with zeros (passes through meta_mlp and meta_proj)
- shuffled: metadata permuted across rows within fold using seed 17

Computes:
- Per-target R2 and weighted R2 per fold and mode
- Pooled OOF weighted R2
- Paired contrasts:
  1. Information use: R2_present - R2_shuffled
  2. Missing metadata penalty: R2_present - R2_zero
  3. Visual damage: R2_zero_metadata_trained - R2_no_metadata_control

Outputs saved to:
  output/metadata_repair_2026_09_13/
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from models import BiomassModelTimm, BiomassModelVMamba
import engine
_encode_metadata = engine._encode_metadata
weighted_r2_score = engine.weighted_r2_score

TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
EVALUATION_PROTOCOL = 'unified_common_v1'

# Model configuration specifications
MODEL_SPECS = {
    'E3_GatedDWConv_Meta': {
        'model_type': 'timm',
        'use_metadata': True,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'no_fusion': False,
        'ckpt_dir': 'checkpoints/E3_GatedDWConv_Meta',
        'is_control': False,
        'control_pair': 'B5_DINOv3_L_GatedDWConv',
        'img_size': 512,
    },
    'B5_DINOv3_L_GatedDWConv': {
        'model_type': 'timm',
        'use_metadata': False,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'no_fusion': False,
        'ckpt_dir': 'output/metadata_repair_2026_09_13/B5_DINOv3_L_GatedDWConv',
        'is_control': True,
        'control_pair': None,
        'img_size': 512,
    },
    'B7_CVGA_Meta': {
        'model_type': 'timm',
        'use_metadata': True,
        'use_cvga': True,
        'use_mamba_ssm': False,
        'no_fusion': False,
        'ckpt_dir': 'checkpoints/B7_CVGA_Meta',
        'is_control': False,
        'control_pair': 'E2_CVGA_NoMeta',
        'img_size': 512,
    },
    'E2_CVGA_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'use_cvga': True,
        'use_mamba_ssm': False,
        'no_fusion': False,
        'ckpt_dir': 'checkpoints/E2_CVGA_NoMeta',
        'is_control': True,
        'control_pair': None,
        'img_size': 512,
    },
    'B8_BidirMamba_Meta': {
        'model_type': 'timm',
        'use_metadata': True,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'use_biobabymamba': True,
        'no_fusion': False,
        'ckpt_dir': 'checkpoints/B8_BidirMamba_Meta',
        'is_control': False,
        'control_pair': 'E1_BidirMamba_NoMeta',
        'img_size': 512,
    },
    'E1_BidirMamba_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'use_biobabymamba': True,
        'no_fusion': False,
        'ckpt_dir': 'checkpoints/E1_BidirMamba_NoMeta',
        'is_control': True,
        'control_pair': None,
        'img_size': 512,
    },
    'E8_Identity_Meta': {
        'model_type': 'timm',
        'use_metadata': True,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'use_biobabymamba': False,
        'no_fusion': True,
        'ckpt_dir': 'checkpoints/E8_Identity_Meta',
        'is_control': False,
        'control_pair': 'E4_Identity_NoMeta',
        'img_size': 512,
    },
    'E4_Identity_NoMeta': {
        'model_type': 'timm',
        'use_metadata': False,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'use_biobabymamba': False,
        'no_fusion': True,
        'ckpt_dir': 'checkpoints/E4_Identity_NoMeta',
        'is_control': True,
        'control_pair': None,
        'img_size': 512,
    },
    'B6_VMamba_Base_Mamba_Meta': {
        'model_type': 'vmamba',
        'use_metadata': True,
        'variant': 'vmamba_base',
        'ckpt_dir': 'output/metadata_repair_2026_09_13/B6_VMamba_Base_Mamba_Meta',
        'is_control': False,
        'control_pair': 'B9_VMamba_Base_Mamba_NoMeta',
        'img_size': 512,
    },
    'B9_VMamba_Base_Mamba_NoMeta': {
        'model_type': 'vmamba',
        'use_metadata': False,
        'variant': 'vmamba_base',
        'ckpt_dir': 'output/metadata_repair_2026_09_13/B9_VMamba_Base_Mamba_NoMeta',
        'is_control': True,
        'control_pair': None,
        'img_size': 512,
    },
    'B4_DINOv2_L_GatedDWConv_NoMeta': {
        'model_type': 'timm',
        'model_name': 'vit_large_patch14_dinov2.lvd142m',
        'use_metadata': False,
        'use_cvga': False,
        'use_mamba_ssm': False,
        'use_biobabymamba': False,
        'no_fusion': False,
        'ckpt_dir': 'output/metadata_repair_2026_09_13/B4_DINOv2_L_GatedDWConv_NoMeta',
        'is_control': True,
        'control_pair': None,
        'img_size': 518,
    },
}


def build_model(spec: dict, meta_dim: int = 23):
    mtype = spec.get('model_type', 'timm')
    if mtype == 'vmamba':
        return BiomassModelVMamba(
            pretrained_path='',
            dropout=0.2,
            use_mamba_ssm=True,
            use_metadata=spec['use_metadata'],
            meta_input_dim=meta_dim if spec['use_metadata'] else 0,
            variant=spec.get('variant', 'vmamba_base'),
        )
    model = BiomassModelTimm(
        model_name=spec.get('model_name', 'vit_large_patch16_dinov3.lvd1689m'),
        dropout=0.2,
        use_mamba_ssm=spec.get('use_mamba_ssm', False),
        use_biobabymamba=spec.get('use_biobabymamba', False),
        use_cvga=spec.get('use_cvga', False),
        pretrained=False,
        use_metadata=spec['use_metadata'],
        meta_input_dim=meta_dim if spec['use_metadata'] else 0,
        img_size=spec.get('img_size', 512),
    )
    if spec.get('no_fusion', False):
        model.fusion = nn.Identity()
    return model


class EvalBiomassDataset(Dataset):
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

        t_left = self.transform(image=left)['image']
        t_right = self.transform(image=right)['image']

        targets = row[TARGET_COLS].values.astype(np.float32)
        meta_present = row[self.meta_cols].values.astype(np.float32)

        return {
            'image_id': image_id,
            'left': t_left,
            'right': t_right,
            'targets': torch.tensor(targets, dtype=torch.float32),
            'meta_present': torch.tensor(meta_present, dtype=torch.float32),
        }


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_val_transform(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.environ.get('BIOMASS_DATA_DIR', repo_root / 'csiro-biomass'))
    fold_csv_path = repo_root / 'output' / 'reruns_2026_09_13' / 'folds_seed17.csv'
    out_dir = repo_root / 'output' / 'metadata_repair_2026_09_13'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print("Evaluating Metadata Modes and Matched Controls Across 5 Folds", flush=True)
    print("=================================================================", flush=True)

    # Load data and encode metadata
    train_csv = data_dir / 'train.csv'
    df_raw = pd.read_csv(train_csv)
    df_raw['image_id'] = df_raw['sample_id'].str.split('__').str[0]

    # Pivot wide to match engine.py
    df_wide = df_raw.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    # Merge fold assignments
    folds_df = pd.read_csv(fold_csv_path)[['image_id', 'fold']]
    df_wide = df_wide.merge(folds_df, on='image_id', how='left')
    assert df_wide['fold'].notna().all(), "Some samples missing fold assignments!"

    # Encode metadata
    df_wide, meta_feature_cols = _encode_metadata(df_raw, df_wide)
    meta_dim = len(meta_feature_cols)
    assert meta_dim == 23, f"Expected 23 metadata features, got {meta_dim}"
    print(f"Encoded {meta_dim} metadata features for {len(df_wide)} images.", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}", flush=True)
    all_predictions = []  # will collect per-sample records
    metrics_summary = {}

    # Check for existing evaluated predictions to avoid recomputing unchanged models
    existing_preds_path = out_dir / 'metadata_modes_oof_predictions.csv'
    existing_df = None
    if existing_preds_path.exists():
        try:
            existing_df = pd.read_csv(existing_preds_path)
            print(f"Loaded existing predictions from {existing_preds_path} ({len(existing_df)} rows).", flush=True)
        except Exception as e:
            print(f"Could not load existing predictions: {e}", flush=True)

    for model_name, spec in MODEL_SPECS.items():
        print(f"\n-------------------------------------------------------------", flush=True)
        print(f"Evaluating Model: {model_name} (use_metadata={spec['use_metadata']})", flush=True)
        print(f"-------------------------------------------------------------", flush=True)

        ckpt_dir = repo_root / spec['ckpt_dir']
        modes_to_eval = ['present', 'zero', 'shuffled'] if spec['use_metadata'] else ['no_meta']

        # Fast path: re-use existing predictions if all modes are present and complete
        if existing_df is not None:
            model_exist = existing_df[existing_df['model'] == model_name]
            protocol_matches = (
                'evaluation_protocol' in model_exist.columns
                and (model_exist['evaluation_protocol'] == EVALUATION_PROTOCOL).all()
            )
            all_modes_present = protocol_matches and all(
                (model_exist['mode'] == m).sum() == 357 for m in modes_to_eval
            )
            if all_modes_present:
                print(f"Re-using complete cached predictions for {model_name} (357 samples x {len(modes_to_eval)} modes).", flush=True)
                for rec in model_exist.to_dict('records'):
                    all_predictions.append(rec)
                continue

        # Check if checkpoints exist before evaluating
        all_ckpts_exist = all((ckpt_dir / f"fold{f}_best.pth").exists() for f in range(5))
        if not all_ckpts_exist:
            print(f"Skipping {model_name}: not all 5 fold checkpoints exist in {ckpt_dir}.", flush=True)
            continue

        model_results = {mode: {'fold_r2': {}, 'fold_target_r2': {}} for mode in modes_to_eval}
        model = build_model(spec, meta_dim=meta_dim).to(device)

        img_size = spec.get('img_size', 512)
        model_val_transform = get_val_transform(img_size=img_size)

        for fold in range(5):
            val_df = df_wide[df_wide['fold'] == fold].copy()
            n_val = len(val_df)
            print(f"  Fold {fold}: {n_val} validation samples", flush=True)

            ckpt_path = ckpt_dir / f"fold{fold}_best.pth"
            ckpt_sha = compute_file_sha256(ckpt_path)

            raw_state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(raw_state, dict) and 'model_state_dict' in raw_state:
                sd = raw_state['model_state_dict']
            elif isinstance(raw_state, dict) and 'state_dict' in raw_state:
                sd = raw_state['state_dict']
            else:
                sd = raw_state
            model.load_state_dict(sd, strict=True)
            model.eval()

            ds = EvalBiomassDataset(val_df, data_dir / 'train', meta_feature_cols, model_val_transform)
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

            # Pre-compute whole-fold deterministic shuffled metadata (seed 17)
            rng = np.random.RandomState(17 + fold)
            shuffled_indices = rng.permutation(n_val)
            val_df_shuffled = val_df.iloc[shuffled_indices].reset_index(drop=True)
            shuffled_meta_all = torch.tensor(
                val_df_shuffled[meta_feature_cols].values.astype(np.float32)
            )

            # Store predictions per mode for this fold
            fold_preds = {m: [] for m in modes_to_eval}
            fold_targets = []
            fold_image_ids = []

            with torch.inference_mode():
                sample_idx_offset = 0
                for batch in loader:
                    bs = len(batch['image_id'])
                    left = batch['left'].to(device, non_blocking=True)
                    right = batch['right'].to(device, non_blocking=True)
                    targets = batch['targets']  # CPU

                    fold_targets.append(targets.numpy())
                    fold_image_ids.extend(batch['image_id'])

                    if not spec['use_metadata']:
                        out = model(left, right)
                        fold_preds['no_meta'].append(out.cpu().numpy())
                    else:
                        # 1. present mode
                        meta_present = batch['meta_present'].to(device, non_blocking=True)
                        out_pres = model(left, right, metadata=meta_present)
                        fold_preds['present'].append(out_pres.cpu().numpy())

                        # 2. zero mode (zeros through meta_mlp)
                        meta_zero = torch.zeros_like(meta_present)
                        out_zero = model(left, right, metadata=meta_zero)
                        fold_preds['zero'].append(out_zero.cpu().numpy())

                        # 3. shuffled mode (whole-fold row permutation)
                        meta_shuf = shuffled_meta_all[sample_idx_offset:sample_idx_offset + bs].to(device, non_blocking=True)
                        out_shuf = model(left, right, metadata=meta_shuf)
                        fold_preds['shuffled'].append(out_shuf.cpu().numpy())

                    sample_idx_offset += bs

            y_true_fold = np.concatenate(fold_targets, axis=0)

            for mode in modes_to_eval:
                y_pred_fold = np.concatenate(fold_preds[mode], axis=0)
                # Calculate metrics
                w_r2, t_r2 = weighted_r2_score(y_true_fold, y_pred_fold)
                model_results[mode]['fold_r2'][fold] = float(w_r2)
                model_results[mode]['fold_target_r2'][fold] = [float(x) for x in t_r2]
                print(f"    Mode [{mode:8s}] Fold {fold} Weighted R2: {w_r2:.4f}", flush=True)

                # Collect row-level records
                for i, img_id in enumerate(fold_image_ids):
                    rec = {
                        'image_id': img_id,
                        'model': model_name,
                        'fold': fold,
                        'mode': mode,
                        'evaluation_protocol': EVALUATION_PROTOCOL,
                        'checkpoint_sha256': ckpt_sha,
                    }
                    for t_idx, t_name in enumerate(TARGET_COLS):
                        rec[f"target_{t_name}"] = float(y_true_fold[i, t_idx])
                        rec[f"pred_{t_name}"] = float(y_pred_fold[i, t_idx])
                    all_predictions.append(rec)

        # Clear GPU cache after model evaluation
        del model
        torch.cuda.empty_cache()

    # Compute aggregate metrics across folds and pooled OOF from all_predictions
    evaluated_models = sorted(list(set(r['model'] for r in all_predictions)))
    for model_name in evaluated_models:
        spec = MODEL_SPECS[model_name]
        modes_to_eval = ['present', 'zero', 'shuffled'] if spec['use_metadata'] else ['no_meta']
        metrics_summary[model_name] = {}

        for mode in modes_to_eval:
            mode_records = [r for r in all_predictions if r['model'] == model_name and r['mode'] == mode]
            if len(mode_records) != 357:
                print(f"Warning: {model_name} [{mode}] has {len(mode_records)} samples instead of 357", flush=True)
                continue

            # Per fold metrics
            fold_r2_dict = {}
            fold_target_r2_dict = {}
            for f in range(5):
                f_recs = [r for r in mode_records if r['fold'] == f]
                y_t = np.array([[r[f"target_{t}"] for t in TARGET_COLS] for r in f_recs])
                y_p = np.array([[r[f"pred_{t}"] for t in TARGET_COLS] for r in f_recs])
                w, t = weighted_r2_score(y_t, y_p)
                fold_r2_dict[f] = float(w)
                fold_target_r2_dict[f] = [float(x) for x in t]

            fold_scores = [fold_r2_dict[f] for f in range(5)]
            mean_fold_r2 = float(np.mean(fold_scores))
            std_fold_r2 = float(np.std(fold_scores))

            # Pooled OOF
            y_true_all = np.array([[r[f"target_{t}"] for t in TARGET_COLS] for r in mode_records])
            y_pred_all = np.array([[r[f"pred_{t}"] for t in TARGET_COLS] for r in mode_records])
            pooled_w_r2, pooled_t_r2 = weighted_r2_score(y_true_all, y_pred_all)

            metrics_summary[model_name][mode] = {
                'mean_fold_r2': mean_fold_r2,
                'std_fold_r2': std_fold_r2,
                'fold_r2': fold_r2_dict,
                'fold_target_r2': fold_target_r2_dict,
                'pooled_oof_r2': float(pooled_w_r2),
                'pooled_target_r2': [float(x) for x in pooled_t_r2],
            }
            print(f"  Summary [{model_name} - {mode:8s}]: Mean Fold R2 = {mean_fold_r2:.4f} ± {std_fold_r2:.4f} | Pooled OOF R2 = {pooled_w_r2:.4f}", flush=True)

    # Compute paired statistical contrasts
    contrasts = {}
    pairs = [
        ('E3_GatedDWConv_Meta', 'B5_DINOv3_L_GatedDWConv'),
        ('B7_CVGA_Meta', 'E2_CVGA_NoMeta'),
        ('B8_BidirMamba_Meta', 'E1_BidirMamba_NoMeta'),
        ('E8_Identity_Meta', 'E4_Identity_NoMeta'),
        ('B6_VMamba_Base_Mamba_Meta', 'B9_VMamba_Base_Mamba_NoMeta'),
    ]

    for meta_model, control_model in pairs:
        if meta_model not in metrics_summary or control_model not in metrics_summary:
            if meta_model == 'B6_VMamba_Base_Mamba_Meta' and control_model in metrics_summary:
                contrasts[f"{meta_model}_vs_{control_model}"] = {
                    'status': 'omitted_missing_metadata_weights',
                    'reason': 'Author checkpoints contain 0 metadata parameters (436 tensors vs 442 expected). Checkpoint is architecturally identical to B9 no-metadata model; strict loading into BiomassModelVMamba(use_metadata=True) fails with 6 missing keys.',
                }
            continue
        meta_summary = metrics_summary[meta_model]
        ctrl_summary = metrics_summary[control_model]['no_meta']

        fold_info_use = [meta_summary['present']['fold_r2'][f] - meta_summary['shuffled']['fold_r2'][f] for f in range(5)]
        fold_missing_penalty = [meta_summary['present']['fold_r2'][f] - meta_summary['zero']['fold_r2'][f] for f in range(5)]
        fold_visual_damage = [meta_summary['zero']['fold_r2'][f] - ctrl_summary['fold_r2'][f] for f in range(5)]
        fold_ordinary_diff = [meta_summary['present']['fold_r2'][f] - ctrl_summary['fold_r2'][f] for f in range(5)]

        pooled_info_use = meta_summary['present']['pooled_oof_r2'] - meta_summary['shuffled']['pooled_oof_r2']
        pooled_missing_penalty = meta_summary['present']['pooled_oof_r2'] - meta_summary['zero']['pooled_oof_r2']
        pooled_visual_damage = meta_summary['zero']['pooled_oof_r2'] - ctrl_summary['pooled_oof_r2']
        pooled_ordinary_diff = meta_summary['present']['pooled_oof_r2'] - ctrl_summary['pooled_oof_r2']

        contrasts[f"{meta_model}_vs_{control_model}"] = {
            'information_use': {
                'definition': 'present - shuffled',
                'fold_differences': fold_info_use,
                'mean_diff': float(np.mean(fold_info_use)),
                'std_diff': float(np.std(fold_info_use)),
                'pooled_oof_diff': float(pooled_info_use),
            },
            'missing_metadata_penalty': {
                'definition': 'present - zero',
                'fold_differences': fold_missing_penalty,
                'mean_diff': float(np.mean(fold_missing_penalty)),
                'std_diff': float(np.std(fold_missing_penalty)),
                'pooled_oof_diff': float(pooled_missing_penalty),
            },
            'visual_predictor_damage': {
                'definition': 'zero_metadata_trained - no_metadata_control',
                'fold_differences': fold_visual_damage,
                'mean_diff': float(np.mean(fold_visual_damage)),
                'std_diff': float(np.std(fold_visual_damage)),
                'pooled_oof_diff': float(pooled_visual_damage),
            },
            'ordinary_metadata_difference': {
                'definition': 'present - no_meta',
                'fold_differences': fold_ordinary_diff,
                'mean_diff': float(np.mean(fold_ordinary_diff)),
                'std_diff': float(np.std(fold_ordinary_diff)),
                'pooled_oof_diff': float(pooled_ordinary_diff),
            },
        }

    # Save outputs
    preds_df = pd.DataFrame(all_predictions)
    preds_csv_path = out_dir / 'metadata_modes_oof_predictions.csv'
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"\nSaved {len(preds_df)} predictions to: {preds_csv_path}", flush=True)

    preds_npz_path = out_dir / 'metadata_modes_oof_predictions.npz'
    np.savez_compressed(
        preds_npz_path,
        evaluation_protocol=np.array(EVALUATION_PROTOCOL),
        image_ids=preds_df['image_id'].values,
        models=preds_df['model'].values,
        folds=preds_df['fold'].values,
        modes=preds_df['mode'].values,
        targets=preds_df[[f"target_{t}" for t in TARGET_COLS]].values,
        predictions=preds_df[[f"pred_{t}" for t in TARGET_COLS]].values,
        target_names=np.array(TARGET_COLS),
    )
    print(f"Saved compressed predictions to: {preds_npz_path}", flush=True)

    summary_file = out_dir / 'metadata_modes_evaluation.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_protocol': EVALUATION_PROTOCOL,
            'dataset_train_csv_sha256': compute_file_sha256(train_csv),
            'fold_assignment_csv_sha256': compute_file_sha256(fold_csv_path),
            'environment': {
                'python': sys.version,
                'torch': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'albumentations': A.__version__,
            },
            'model_specs': MODEL_SPECS,
            'metadata_features': meta_feature_cols,
            'metrics_summary': metrics_summary,
            'paired_contrasts': contrasts,
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_file}", flush=True)


if __name__ == '__main__':
    main()
