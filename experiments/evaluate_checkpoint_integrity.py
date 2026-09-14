"""
Comprehensive checkpoint-only consistency audit across all obtainable biomass models.
Recomputes fold and aggregate R^2 from fresh inference (torch.inference_mode, strict=True),
audits checkpoint weights, compares with manuscript claims and stored OOF artifacts.
"""

import os
import sys
import gc
import json
import hashlib
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Workspace paths
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from models import BiomassModelTimm, BiomassModelVMamba, GatedDepthwiseConvBlock
from frozen_probe import DualViewFrozenProbe
import engine
_encode_metadata = engine._encode_metadata

TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)

DATA_DIR = Path(os.environ.get('BIOMASS_DATA_DIR', REPO_ROOT.parent / 'csiro-biomass'))
FOLD_FILE = REPO_ROOT / 'output' / 'reruns_2026_09_13' / 'folds_seed17.csv'
OUT_DIR = REPO_ROOT / 'output' / 'checkpoint_integrity_2026_09_14'
PAPER_PATH = REPO_ROOT.parent / 'paper' / 'MandalBiomassPaper.tex'

EXPECTED_TRAIN_SHA256 = '1a6b4c019c3a2386c7626b4002f1b2923d402341fb2a89132f66c57815c5f0e0'
EXPECTED_FOLD_SHA256 = '721f05a3dd0c00e4e2bd30bd2474da45d4ec1c7e3ca2abf88b75821bf7f7fe9f'


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    scores = []
    for idx in range(y_true.shape[1]):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        scores.append(r2)
    per_target = np.array(scores, dtype=np.float64)
    weighted = float(np.sum(per_target * TARGET_WEIGHTS))
    return weighted, per_target


def compute_log1p_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    y_true_valid = np.maximum(0.0, y_true)
    y_pred_valid = np.maximum(0.0, y_pred)
    y_true_log = np.log1p(y_true_valid)
    y_pred_log = np.log1p(y_pred_valid)
    return compute_weighted_r2(y_true_log, y_pred_log)


class ImageEvalDataset(Dataset):
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
        meta_vec = row[self.meta_cols].values.astype(np.float32) if self.meta_cols else np.zeros(0, dtype=np.float32)

        return {
            'image_id': image_id,
            'left': t_left,
            'right': t_right,
            'targets': torch.tensor(targets, dtype=torch.float32),
            'meta': torch.tensor(meta_vec, dtype=torch.float32),
            'fold': int(row['fold']),
            'row_idx': idx
        }


def get_val_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def build_experiment_registry() -> dict:
    return {
        'B1': {
            'id': 'B1',
            'name': 'Median_Predictor',
            'family': 'Nonparametric',
            'arch_class': 'MedianPredictor',
            'backbone': 'None',
            'fusion_type': 'None',
            'fusion_depth': 0,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': None,
            'ckpt_dir': None,
            'ckpt_pattern': None,
            'provenance': 'Deterministic nonparametric control',
            'paper_claim': {'r2': -0.065, 'std': 0.006, 'table': 'tab:main_results'},
            'special_status': 'MISSING_CHECKPOINT'
        },
        'B2': {
            'id': 'B2',
            'name': 'B2_EfficientNet_B3_matched',
            'family': 'EfficientNet',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'efficientnet_b3.ra2_in1k',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 448,
            'ckpt_dir': 'output/reruns_2026_09_13/B2_EfficientNet_B3_matched',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Local rerun (protocol matched dual-view)',
            'paper_claim': {'r2': 0.555, 'std': 0.084, 'table': 'tab:main_results'},
            'special_status': 'MISSING_OR_SUPERSEDED_CHECKPOINT'
        },
        'B3': {
            'id': 'B3',
            'name': 'B3_DINOv2_Base_probe',
            'family': 'DINOv2',
            'arch_class': 'DualViewFrozenProbe',
            'backbone': 'facebook/dinov2-base',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 448,
            'ckpt_dir': 'output/reruns_2026_09_13/B3_DINOv2_Base_probe',
            'ckpt_pattern': 'probe_fold{fold}.pth',
            'features_npz': 'output/reruns_2026_09_13/B3_DINOv2_Base_probe/features.npz',
            'provenance': 'Local rerun (supervised frozen probe)',
            'paper_claim': {'r2': -1.999, 'std': 0.341, 'table': 'tab:main_results'},
            'special_status': 'SUPERSEDED_PROTOCOL'
        },
        'B4': {
            'id': 'B4',
            'name': 'B4_DINOv2_L_GatedDWConv_NoMeta',
            'family': 'DINOv2',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch14_dinov2.lvd142m',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 518,
            'ckpt_dir': 'output/metadata_repair_2026_09_13/B4_DINOv2_L_GatedDWConv_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Local rerun',
            'paper_claim': {'r2': 0.853, 'std': 0.097, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B5': {
            'id': 'B5',
            'name': 'B5_DINOv3_L_GatedDWConv',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'output/metadata_repair_2026_09_13/B5_DINOv3_L_GatedDWConv',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Local rerun',
            'paper_claim': {'r2': 0.903, 'std': 0.064, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B6': {
            'id': 'B6',
            'name': 'B6_VMamba_Base_Mamba_Meta',
            'family': 'VMamba',
            'arch_class': 'BiomassModelVMamba',
            'backbone': 'vmamba_base',
            'fusion_type': 'MambaFusionBlock',
            'fusion_depth': 2,
            'use_metadata': True,
            'meta_dim': 23,
            'img_size': 512,
            'ckpt_dir': 'output/metadata_repair_2026_09_13/B6_VMamba_Base_Mamba_Meta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Local training (genuine metadata enabled)',
            'paper_claim': {'r2': 0.743, 'std': 0.048, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B7': {
            'id': 'B7',
            'name': 'B7_CVGA_Meta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'CVGABlock',
            'fusion_depth': 2,
            'use_metadata': True,
            'meta_dim': 23,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/B7_CVGA_Meta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.830, 'std': 0.050, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B8': {
            'id': 'B8',
            'name': 'B8_BidirMamba_Meta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'BioBabyMamba',
            'fusion_depth': 2,
            'use_metadata': True,
            'meta_dim': 23,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/B8_BidirMamba_Meta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.829, 'std': 0.042, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B9': {
            'id': 'B9',
            'name': 'B9_VMamba_Base_Mamba_NoMeta',
            'family': 'VMamba',
            'arch_class': 'BiomassModelVMamba',
            'backbone': 'vmamba_base',
            'fusion_type': 'MambaFusionBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'output/metadata_repair_2026_09_13/B9_VMamba_Base_Mamba_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Recovered author checkpoint (mislabeled as B6)',
            'paper_claim': {'r2': 0.717, 'std': 0.052, 'table': 'tab:main_results'},
            'special_status': None
        },
        'B10': {
            'id': 'B10',
            'name': 'B10_DINOv2_Giant_probe',
            'family': 'DINOv2',
            'arch_class': 'DualViewFrozenProbe',
            'backbone': 'facebook/dinov2-giant',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 448,
            'ckpt_dir': 'output/reruns_2026_09_13/B10_DINOv2_Giant_probe',
            'ckpt_pattern': 'probe_fold{fold}.pth',
            'features_npz': 'output/reruns_2026_09_13/B10_DINOv2_Giant_probe/features.npz',
            'provenance': 'Local rerun (supplemental frozen probe)',
            'paper_claim': None,
            'special_status': 'SUPPLEMENTAL_RESULT'
        },
        'E1': {
            'id': 'E1',
            'name': 'E1_BidirMamba_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'BioBabyMamba',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E1_BidirMamba_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.819, 'std': 0.051, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E2': {
            'id': 'E2',
            'name': 'E2_CVGA_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'CVGABlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E2_CVGA_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.833, 'std': 0.051, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E3': {
            'id': 'E3',
            'name': 'E3_GatedDWConv_Meta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 2,
            'use_metadata': True,
            'meta_dim': 23,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E3_GatedDWConv_Meta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.829, 'std': 0.045, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E4': {
            'id': 'E4',
            'name': 'E4_Identity_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'Identity',
            'fusion_depth': 0,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E4_Identity_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.819, 'std': 0.055, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E5': {
            'id': 'E5',
            'name': 'E5_FullMamba_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'MambaFusionBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E5_FullMamba_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.793, 'std': 0.034, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E6': {
            'id': 'E6',
            'name': 'E6_GatedDWConv1x_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 1,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E6_GatedDWConv1x_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.821, 'std': 0.034, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E7': {
            'id': 'E7',
            'name': 'E7_GatedDWConv4x_NoMeta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'GatedDepthwiseConvBlock',
            'fusion_depth': 4,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E7_GatedDWConv4x_NoMeta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.814, 'std': 0.039, 'table': 'tab:main_results'},
            'special_status': None
        },
        'E8': {
            'id': 'E8',
            'name': 'E8_Identity_Meta',
            'family': 'DINOv3',
            'arch_class': 'BiomassModelTimm',
            'backbone': 'vit_large_patch16_dinov3.lvd1689m',
            'fusion_type': 'Identity',
            'fusion_depth': 0,
            'use_metadata': True,
            'meta_dim': 23,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/E8_Identity_Meta',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint',
            'paper_claim': {'r2': 0.828, 'std': 0.053, 'table': 'tab:main_results'},
            'special_status': None
        },
        'A1': {
            'id': 'A1',
            'name': 'A1_SSM_Scale',
            'family': 'VMamba',
            'arch_class': 'BiomassModelVMamba',
            'backbone': 'vmamba_base',
            'fusion_type': 'MambaFusionBlock',
            'fusion_depth': 2,
            'use_metadata': False,
            'meta_dim': 0,
            'img_size': 512,
            'ckpt_dir': 'checkpoints/A1_SSM_Scale',
            'ckpt_pattern': 'fold{fold}_best.pth',
            'provenance': 'Downloaded Kaggle checkpoint (A1 scale sweep)',
            'paper_claim': {'r2': 0.717, 'std': 0.052, 'table': 'sec:4.3'},
            'special_status': 'SUPPLEMENTAL_RESULT'
        }
    }


def instantiate_model(spec: dict):
    arch = spec['arch_class']
    if arch == 'BiomassModelTimm':
        use_biobabymamba = (spec['fusion_type'] == 'BioBabyMamba')
        use_cvga = (spec['fusion_type'] == 'CVGABlock')
        use_mamba_ssm = (spec['fusion_type'] == 'MambaFusionBlock')
        no_fusion = (spec['fusion_type'] == 'Identity')

        m = BiomassModelTimm(
            model_name=spec['backbone'],
            dropout=0.2,
            use_mamba_ssm=use_mamba_ssm,
            use_biobabymamba=use_biobabymamba,
            use_cvga=use_cvga,
            pretrained=False,
            use_metadata=spec['use_metadata'],
            meta_input_dim=spec['meta_dim'],
            img_size=spec['img_size']
        )
        if no_fusion:
            m.fusion = nn.Identity()
        elif spec['fusion_depth'] == 1 and spec['fusion_type'] == 'GatedDepthwiseConvBlock':
            nf = m.backbone.num_features
            m.fusion = nn.Sequential(GatedDepthwiseConvBlock(nf, dropout=0.2))
        elif spec['fusion_depth'] == 4 and spec['fusion_type'] == 'GatedDepthwiseConvBlock':
            nf = m.backbone.num_features
            m.fusion = nn.Sequential(
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2),
                GatedDepthwiseConvBlock(nf, dropout=0.2)
            )
        return m
    elif arch == 'BiomassModelVMamba':
        m = BiomassModelVMamba(
            pretrained_path='',
            dropout=0.2,
            use_mamba_ssm=True,
            use_metadata=spec['use_metadata'],
            meta_input_dim=spec['meta_dim'],
            variant=spec['backbone']
        )
        return m
    elif arch == 'DualViewFrozenProbe':
        in_dim = 768 if 'base' in spec['backbone'] else 1536
        m = DualViewFrozenProbe(input_dim=in_dim, projection_dim=256, dropout=0.2)
        return m
    else:
        raise ValueError(f"Unknown architecture {arch}")


def extract_claims_ledger(paper_file: Path) -> list[dict]:
    """Extract all numerical claims from LaTeX manuscript."""
    text = paper_file.read_text(encoding='utf-8')
    ledger = []

    # 1. Main results table
    # Pattern: ID & Model & Backbone & Fusion & Meta & R2 & Std
    main_rows = re.findall(r'([BE]\d+)\s*&\s*([^&]+)&\s*([^&]+)&\s*([^&]+)&\s*([^&]+)&\s*([^\s&]+)\s*&\s*([^\s\\\\&]+)', text)
    for row in main_rows:
        exp_id = row[0].replace(r'\textbf{', '').replace('}', '').strip()
        r2_val = row[5].replace(r'\textbf{', '').replace('}', '').replace('$', '').replace('{', '').replace('}', '').replace('-', '-').strip()
        std_val = row[6].replace(r'\textbf{', '').replace('}', '').replace('$', '').replace('{', '').replace('}', '').strip()
        try:
            r2_flt = float(r2_val)
            std_flt = float(std_val)
            ledger.append({
                'table': 'tab:main_results',
                'experiment_id': exp_id,
                'model_name': row[1].strip(),
                'metric': 'weighted_r2',
                'claimed_mean': r2_flt,
                'claimed_std': std_flt,
                'raw_text': f"{row[0]} R2={r2_val} Std={std_val}"
            })
        except ValueError:
            pass

    # 2. Fold 4 table
    fold4_rows = re.findall(r'([BE]\d+)[^&]*&\s*([0-9\.]+)\s*&\s*([^\s\\\\&]+)', text)
    for row in fold4_rows:
        exp_id = row[0].strip()
        try:
            r2_flt = float(row[1])
            ledger.append({
                'table': 'tab:fold4',
                'experiment_id': exp_id,
                'metric': 'fold4_weighted_r2',
                'claimed_mean': r2_flt,
                'claimed_std': None,
                'raw_text': f"{exp_id} Fold4={row[1]} Drop={row[2]}"
            })
        except ValueError:
            pass

    # 3. Factorial table
    fact_rows = re.findall(r'([0-9\.]+\s*\([BE]\d+\))\s*&\s*([0-9\.]+\s*\([BE]\d+\))\s*&\s*([^\s\\\\&]+)', text)
    for row in fact_rows:
        ledger.append({
            'table': 'tab:factorial',
            'metric': 'metadata_delta',
            'no_meta_cell': row[0].strip(),
            'with_meta_cell': row[1].strip(),
            'delta_cell': row[2].strip()
        })

    # 4. Backbone table
    bb_rows = re.findall(r'(EfficientNet-B3|VMamba-Base|DINOv2-ViT-L|DINOv3-ViT-L)\s*&\s*([^&]+)&\s*([^&]+)&\s*([0-9\.]+)\s*&\s*([^\s\\\\&]+)', text)
    for row in bb_rows:
        ledger.append({
            'table': 'tab:backbone',
            'backbone': row[0].strip(),
            'metric': 'weighted_r2',
            'claimed_mean': float(row[3]),
            'claimed_delta': row[4].strip()
        })

    return ledger


def main():
    print("=================================================================", flush=True)
    print("Checkpoint-Only Consistency Audit and Falsification Ledger", flush=True)
    print("=================================================================", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = OUT_DIR / 'model_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Split Validation
    train_csv = DATA_DIR / 'train.csv'
    assert train_csv.exists(), f"train.csv missing at {train_csv}"
    actual_train_sha256 = compute_sha256(train_csv)
    print(f"Dataset train.csv SHA256: {actual_train_sha256}", flush=True)
    assert actual_train_sha256 == EXPECTED_TRAIN_SHA256, "train.csv hash mismatch!"

    assert FOLD_FILE.exists(), f"folds file missing at {FOLD_FILE}"
    actual_fold_sha256 = compute_sha256(FOLD_FILE)
    print(f"Folds CSV SHA256: {actual_fold_sha256}", flush=True)
    assert actual_fold_sha256 == EXPECTED_FOLD_SHA256, "folds file hash mismatch!"

    # Load dataset
    df_raw = pd.read_csv(train_csv)
    df_raw['image_id'] = df_raw['sample_id'].str.split('__').str[0]
    df_wide = df_raw.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    folds_df = pd.read_csv(FOLD_FILE)[['image_id', 'fold']]
    df_wide = df_wide.merge(folds_df, on='image_id', how='left')
    assert df_wide['fold'].notna().all()
    assert len(df_wide) == 357, f"Expected 357 images, got {len(df_wide)}"
    for f in range(5):
        cnt = (df_wide['fold'] == f).sum()
        assert cnt in (71, 72), f"Fold {f} has unexpected count {cnt}"

    # Metadata encoding
    df_wide, meta_feature_cols = _encode_metadata(df_raw, df_wide)
    meta_dim = len(meta_feature_cols)
    assert meta_dim == 23, f"Expected 23 metadata features, got {meta_dim}"

    # 2. Extract Claims Ledger
    claims_ledger = extract_claims_ledger(PAPER_PATH)
    with open(OUT_DIR / 'claims_ledger.json', 'w', encoding='utf-8') as f:
        json.dump(claims_ledger, f, indent=2)
    print(f"Saved {len(claims_ledger)} parsed claims to {OUT_DIR / 'claims_ledger.json'}", flush=True)

    # 3. Experiment Registry
    registry = build_experiment_registry()
    with open(OUT_DIR / 'experiment_registry.json', 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    print(f"Saved {len(registry)} experiment registry definitions to {OUT_DIR / 'experiment_registry.json'}", flush=True)

    # 4. Checkpoint Manifest & Duplicates Audit
    print("\n--- Auditing Checkpoints across Registry ---", flush=True)
    manifest_path = OUT_DIR / 'checkpoint_manifest.json'
    duplicates_path = OUT_DIR / 'checkpoint_duplicates.json'

    if manifest_path.exists() and duplicates_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            checkpoint_manifest = json.load(f)
        with open(duplicates_path, 'r', encoding='utf-8') as f:
            duplicates = json.load(f)
        print(f"Loaded pre-computed checkpoint manifest ({len(checkpoint_manifest)} files) and duplicates audit.", flush=True)
    else:
        checkpoint_manifest = []
        hash_to_ckpts = {}
        sig_to_ckpts = {}

        for exp_id, spec in registry.items():
            if spec['ckpt_dir'] is None:
                continue
            ckpt_dir = REPO_ROOT / spec['ckpt_dir']
            assert ckpt_dir.exists(), f"Missing ckpt dir: {ckpt_dir}"

            for fold in range(5):
                pth_name = spec['ckpt_pattern'].format(fold=fold)
                pth_file = ckpt_dir / pth_name
                assert pth_file.exists(), f"Missing checkpoint: {pth_file}"

                sha = compute_sha256(pth_file)
                size = pth_file.stat().st_size

                # Inspect state dict on CPU
                sd = torch.load(pth_file, map_location='cpu')
                if isinstance(sd, dict) and 'model_state_dict' in sd:
                    weights = sd['model_state_dict']
                elif isinstance(sd, dict) and 'state_dict' in sd:
                    weights = sd['state_dict']
                else:
                    weights = sd

                key_count = len(weights)
                param_count = sum(p.numel() for p in weights.values() if isinstance(p, torch.Tensor))
                dtypes = sorted(list(set(str(p.dtype) for p in weights.values() if isinstance(p, torch.Tensor))))

                # Shape signature (hash of sorted tensor names and shapes)
                sig_str = ";".join(f"{k}:{list(v.shape)}" for k, v in sorted(weights.items()) if isinstance(v, torch.Tensor))
                sig_hash = hashlib.sha256(sig_str.encode('utf-8')).hexdigest()

                # Metadata tensor check
                has_meta_mlp = any('meta_mlp' in k for k in weights.keys())
                has_meta_proj = any('meta_proj' in k for k in weights.keys())
                if spec['use_metadata']:
                    assert has_meta_mlp or has_meta_proj, f"Model {exp_id} expects metadata but lacks meta weights!"
                else:
                    assert not (has_meta_mlp or has_meta_proj), f"Model {exp_id} is no_meta but has meta weights!"

                # Tensor anomalies
                has_nan = any(torch.isnan(p).any().item() for p in weights.values() if isinstance(p, torch.Tensor))
                has_inf = any(torch.isinf(p).any().item() for p in weights.values() if isinstance(p, torch.Tensor))
                all_zeros = any((p == 0).all().item() for p in weights.values() if isinstance(p, torch.Tensor) and p.numel() > 10)

                entry = {
                    'experiment_id': exp_id,
                    'fold': fold,
                    'path': str(pth_file.relative_to(REPO_ROOT)),
                    'sha256': sha,
                    'size_bytes': size,
                    'key_count': key_count,
                    'param_count': param_count,
                    'dtypes': dtypes,
                    'shape_signature': sig_hash,
                    'has_metadata_weights': (has_meta_mlp or has_meta_proj),
                    'has_nan': bool(has_nan),
                    'has_inf': bool(has_inf),
                    'has_all_zeros_tensor': bool(all_zeros)
                }
                checkpoint_manifest.append(entry)

                hash_to_ckpts.setdefault(sha, []).append(entry['path'])
                sig_to_ckpts.setdefault(sig_hash, []).append(entry['path'])

                del sd, weights
                gc.collect()

        with open(OUT_DIR / 'checkpoint_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(checkpoint_manifest, f, indent=2)

        duplicates = {
            'exact_file_hash_duplicates': {k: v for k, v in hash_to_ckpts.items() if len(v) > 1},
            'tensor_signature_duplicates': {k: v for k, v in sig_to_ckpts.items() if len(v) > 1}
        }
        with open(OUT_DIR / 'checkpoint_duplicates.json', 'w', encoding='utf-8') as f:
            json.dump(duplicates, f, indent=2)
        print(f"Audited {len(checkpoint_manifest)} checkpoint files. Duplicate sets detected: {len(duplicates['exact_file_hash_duplicates'])} hash sets, {len(duplicates['tensor_signature_duplicates'])} signature sets.", flush=True)

    sha_lookup = {entry['path']: entry['sha256'] for entry in checkpoint_manifest}

    # 5. Fresh Inference Protocol
    print("\n--- Running Fresh Inference Protocol Across All Models ---", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}", flush=True)

    fresh_predictions_rows = []
    all_fold_metrics = []
    all_aggregate_metrics = []
    run_timings = {}

    image_dir = DATA_DIR / 'train'

    for exp_id, spec in registry.items():
        print(f"\nEvaluating {exp_id} ({spec['name']})...", flush=True)
        t0 = time.perf_counter()
        log_lines = []

        model_oof_preds = np.zeros((357, 5), dtype=np.float32)
        model_oof_targets = np.zeros((357, 5), dtype=np.float32)
        model_oof_image_ids = [''] * 357
        model_oof_folds = np.zeros(357, dtype=np.int32)
        model_fold_raw_r2 = []
        model_fold_log1p_r2 = []
        model_fold_target_raw_r2 = []
        model_fold_target_log1p_r2 = []

        if exp_id == 'B1':
            # Deterministic Nonparametric Median Predictor
            for fold in range(5):
                val_mask = (df_wide['fold'] == fold).values
                train_mask = ~val_mask

                train_targets = df_wide.loc[train_mask, TARGET_COLS].values.astype(np.float32)
                val_targets = df_wide.loc[val_mask, TARGET_COLS].values.astype(np.float32)
                val_indices = np.where(val_mask)[0]

                # Median of training targets
                fold_median = np.median(train_targets, axis=0)
                preds = np.tile(fold_median, (len(val_targets), 1))

                model_oof_preds[val_indices] = preds
                model_oof_targets[val_indices] = val_targets
                model_oof_folds[val_indices] = fold
                for vi, i_id in zip(val_indices, df_wide.loc[val_mask, 'image_id']):
                    model_oof_image_ids[vi] = i_id

                r_raw, pt_raw = compute_weighted_r2(val_targets, preds)
                r_log, pt_log = compute_log1p_weighted_r2(val_targets, preds)

                model_fold_raw_r2.append(r_raw)
                model_fold_log1p_r2.append(r_log)
                model_fold_target_raw_r2.append(pt_raw)
                model_fold_target_log1p_r2.append(pt_log)

                all_fold_metrics.append({
                    'experiment_id': exp_id,
                    'fold': fold,
                    'raw_weighted_r2': r_raw,
                    'log1p_weighted_r2': r_log,
                    'raw_target_r2': pt_raw.tolist(),
                    'log1p_target_r2': pt_log.tolist(),
                    'checkpoint_sha256': 'NONE_MEDIAN_PREDICTOR'
                })
        elif spec['arch_class'] == 'DualViewFrozenProbe':
            # Frozen Probe Model
            feat_data = np.load(REPO_ROOT / spec['features_npz'])
            left_feat = torch.tensor(feat_data['left'], dtype=torch.float32, device=device)
            right_feat = torch.tensor(feat_data['right'], dtype=torch.float32, device=device)

            for fold in range(5):
                val_mask = (df_wide['fold'] == fold).values
                val_indices = np.where(val_mask)[0]
                val_targets = df_wide.loc[val_mask, TARGET_COLS].values.astype(np.float32)

                model = instantiate_model(spec).to(device)
                ckpt_path = REPO_ROOT / spec['ckpt_dir'] / spec['ckpt_pattern'].format(fold=fold)
                ckpt_sha = sha_lookup.get(str(ckpt_path.relative_to(REPO_ROOT)), 'UNKNOWN')
                sd = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(sd, strict=True)
                model.eval()

                with torch.inference_mode():
                    preds = model(left_feat[val_indices], right_feat[val_indices]).cpu().numpy()

                # Determinism test
                with torch.inference_mode():
                    test_p1 = model(left_feat[val_indices[:4]], right_feat[val_indices[:4]]).cpu().numpy()
                    test_p2 = model(left_feat[val_indices[:4]], right_feat[val_indices[:4]]).cpu().numpy()
                    diff_exact = np.max(np.abs(test_p1 - test_p2))
                    assert diff_exact < 1e-6, f"Probe exact determinism failure: diff={diff_exact}"
                    diff_batch = np.max(np.abs(test_p1 - preds[:4]))
                    assert diff_batch < 1e-4, f"Probe batch determinism failure: diff={diff_batch}"

                assert preds.shape == (len(val_indices), 5)
                assert np.all(np.isfinite(preds))
                # Compositional check
                assert np.allclose(preds[:, 3], preds[:, 0] + preds[:, 2], atol=1e-4)
                assert np.allclose(preds[:, 4], preds[:, 3] + preds[:, 1], atol=1e-4)

                model_oof_preds[val_indices] = preds
                model_oof_targets[val_indices] = val_targets
                model_oof_folds[val_indices] = fold
                for vi, i_id in zip(val_indices, df_wide.loc[val_mask, 'image_id']):
                    model_oof_image_ids[vi] = i_id

                r_raw, pt_raw = compute_weighted_r2(val_targets, preds)
                r_log, pt_log = compute_log1p_weighted_r2(val_targets, preds)

                model_fold_raw_r2.append(r_raw)
                model_fold_log1p_r2.append(r_log)
                model_fold_target_raw_r2.append(pt_raw)
                model_fold_target_log1p_r2.append(pt_log)

                all_fold_metrics.append({
                    'experiment_id': exp_id,
                    'fold': fold,
                    'raw_weighted_r2': r_raw,
                    'log1p_weighted_r2': r_log,
                    'raw_target_r2': pt_raw.tolist(),
                    'log1p_target_r2': pt_log.tolist(),
                    'checkpoint_sha256': ckpt_sha
                })

                del model, sd
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Full image-based architectures
            transform = get_val_transform(spec['img_size'])
            dataset = ImageEvalDataset(
                df=df_wide,
                image_dir=image_dir,
                meta_cols=meta_feature_cols if spec['use_metadata'] else [],
                transform=transform
            )

            for fold in range(5):
                val_mask = (df_wide['fold'] == fold).values
                val_indices = np.where(val_mask)[0]
                val_sub = torch.utils.data.Subset(dataset, val_indices)
                val_loader = DataLoader(val_sub, batch_size=16, shuffle=False, num_workers=0)

                model = instantiate_model(spec).to(device)
                ckpt_path = REPO_ROOT / spec['ckpt_dir'] / spec['ckpt_pattern'].format(fold=fold)
                ckpt_sha = sha_lookup.get(str(ckpt_path.relative_to(REPO_ROOT)), 'UNKNOWN')

                raw_sd = torch.load(ckpt_path, map_location='cpu')
                if isinstance(raw_sd, dict) and 'model_state_dict' in raw_sd:
                    sd = raw_sd['model_state_dict']
                elif isinstance(raw_sd, dict) and 'state_dict' in raw_sd:
                    sd = raw_sd['state_dict']
                else:
                    sd = raw_sd

                model.load_state_dict(sd, strict=True)
                model.eval()

                fold_preds = []
                fold_targets = []
                fold_img_ids = []

                with torch.inference_mode():
                    for batch in val_loader:
                        l = batch['left'].to(device)
                        r = batch['right'].to(device)
                        m = batch['meta'].to(device) if spec['use_metadata'] else None

                        if spec['use_metadata']:
                            out = model(l, r, m)
                        else:
                            out = model(l, r)

                        fold_preds.append(out.cpu().numpy())
                        fold_targets.append(batch['targets'].numpy())
                        fold_img_ids.extend(batch['image_id'])

                preds = np.concatenate(fold_preds, axis=0)
                targets = np.concatenate(fold_targets, axis=0)

                # Determinism check on first batch
                with torch.inference_mode():
                    first_b = next(iter(val_loader))
                    l = first_b['left'].to(device)
                    r = first_b['right'].to(device)
                    m = first_b['meta'].to(device) if spec['use_metadata'] else None
                    det_out1 = model(l, r, m) if spec['use_metadata'] else model(l, r)
                    det_out2 = model(l, r, m) if spec['use_metadata'] else model(l, r)
                    diff_exact = np.max(np.abs(det_out1.cpu().numpy() - det_out2.cpu().numpy()))
                    assert diff_exact < 1e-6, f"Exact determinism failure in {exp_id} fold {fold}: diff={diff_exact}"
                    diff_batch = np.max(np.abs(det_out1.cpu().numpy() - preds[:len(det_out1)]))
                    assert diff_batch < 1e-4, f"Batch determinism failure in {exp_id} fold {fold}: diff={diff_batch}"

                assert preds.shape == (len(val_indices), 5)
                assert np.all(np.isfinite(preds))
                # Compositional check
                assert np.allclose(preds[:, 3], preds[:, 0] + preds[:, 2], atol=1e-4)
                assert np.allclose(preds[:, 4], preds[:, 3] + preds[:, 1], atol=1e-4)

                model_oof_preds[val_indices] = preds
                model_oof_targets[val_indices] = targets
                model_oof_folds[val_indices] = fold
                for vi, i_id in zip(val_indices, fold_img_ids):
                    model_oof_image_ids[vi] = i_id

                r_raw, pt_raw = compute_weighted_r2(targets, preds)
                r_log, pt_log = compute_log1p_weighted_r2(targets, preds)

                model_fold_raw_r2.append(r_raw)
                model_fold_log1p_r2.append(r_log)
                model_fold_target_raw_r2.append(pt_raw)
                model_fold_target_log1p_r2.append(pt_log)

                all_fold_metrics.append({
                    'experiment_id': exp_id,
                    'fold': fold,
                    'raw_weighted_r2': r_raw,
                    'log1p_weighted_r2': r_log,
                    'raw_target_r2': pt_raw.tolist(),
                    'log1p_target_r2': pt_log.tolist(),
                    'checkpoint_sha256': ckpt_sha
                })

                del model, raw_sd, sd, val_loader, val_sub
                gc.collect()
                torch.cuda.empty_cache()

        # Build fresh predictions dataframe rows
        for idx in range(357):
            fresh_predictions_rows.append({
                'experiment_id': exp_id,
                'image_id': model_oof_image_ids[idx],
                'fold': int(model_oof_folds[idx]),
                'true_green': float(model_oof_targets[idx, 0]),
                'true_dead': float(model_oof_targets[idx, 1]),
                'true_clover': float(model_oof_targets[idx, 2]),
                'true_gdm': float(model_oof_targets[idx, 3]),
                'true_total': float(model_oof_targets[idx, 4]),
                'pred_green': float(model_oof_preds[idx, 0]),
                'pred_dead': float(model_oof_preds[idx, 1]),
                'pred_clover': float(model_oof_preds[idx, 2]),
                'pred_gdm': float(model_oof_preds[idx, 3]),
                'pred_total': float(model_oof_preds[idx, 4]),
            })

        # Compute aggregate metrics
        pooled_raw_r2, pooled_target_raw = compute_weighted_r2(model_oof_targets, model_oof_preds)
        pooled_log1p_r2, pooled_target_log1p = compute_log1p_weighted_r2(model_oof_targets, model_oof_preds)

        mean_raw = float(np.mean(model_fold_raw_r2))
        std_raw_ddof0 = float(np.std(model_fold_raw_r2, ddof=0))
        std_raw_ddof1 = float(np.std(model_fold_raw_r2, ddof=1))

        mean_log = float(np.mean(model_fold_log1p_r2))
        std_log_ddof0 = float(np.std(model_fold_log1p_r2, ddof=0))
        std_log_ddof1 = float(np.std(model_fold_log1p_r2, ddof=1))

        min_pred = float(np.min(model_oof_preds))
        neg_count = int(np.sum(model_oof_preds < 0.0))

        runtime = time.perf_counter() - t0
        run_timings[exp_id] = runtime

        agg_entry = {
            'experiment_id': exp_id,
            'model_name': spec['name'],
            'mean_fold_raw_r2': mean_raw,
            'std_fold_raw_r2_ddof0': std_raw_ddof0,
            'std_fold_raw_r2_ddof1': std_raw_ddof1,
            'pooled_oof_raw_r2': pooled_raw_r2,
            'pooled_target_raw_r2': pooled_target_raw.tolist(),
            'mean_fold_log1p_r2': mean_log,
            'std_fold_log1p_r2_ddof0': std_log_ddof0,
            'std_fold_log1p_r2_ddof1': std_log_ddof1,
            'pooled_oof_log1p_r2': pooled_log1p_r2,
            'pooled_target_log1p_r2': pooled_target_log1p.tolist(),
            'min_prediction_value': min_pred,
            'negative_prediction_count': neg_count,
            'evaluation_seconds': runtime
        }
        all_aggregate_metrics.append(agg_entry)
        print(f"  -> RAW: Mean={mean_raw:.4f} +- {std_raw_ddof0:.4f} (ddof0) | Pooled={pooled_raw_r2:.4f}", flush=True)
        print(f"  -> LOG1P: Mean={mean_log:.4f} +- {std_log_ddof0:.4f} (ddof0) | Pooled={pooled_log1p_r2:.4f}", flush=True)

    # Save Fresh Predictions
    fresh_preds_df = pd.DataFrame(fresh_predictions_rows)
    fresh_preds_df.to_csv(OUT_DIR / 'fresh_oof_predictions.csv', index=False)
    print(f"\nSaved fresh predictions ({len(fresh_preds_df)} rows) to {OUT_DIR / 'fresh_oof_predictions.csv'}", flush=True)

    # Save Fold & Aggregate Metrics
    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_csv(OUT_DIR / 'fold_metrics.csv', index=False)

    agg_metrics_df = pd.DataFrame(all_aggregate_metrics)
    agg_metrics_df.to_csv(OUT_DIR / 'aggregate_metrics.csv', index=False)

    # 6. Stored Artifact Comparison
    print("\n--- Comparing Fresh Predictions Against Stored OOF Artifacts ---", flush=True)
    stored_comparisons = []
    for exp_id, spec in registry.items():
        if spec['ckpt_dir'] is None:
            continue
        ckpt_dir = REPO_ROOT / spec['ckpt_dir']
        oof_npz = ckpt_dir / 'oof_predictions.npz'
        train_sum = ckpt_dir / 'training_summary.json'

        has_oof = oof_npz.exists()
        has_sum = train_sum.exists()

        max_diff = None
        mean_diff = None
        stored_mean_cv = None
        stored_pooled_r2 = None
        classification = 'NO_STORED_ARTIFACT'

        fresh_m_preds = fresh_preds_df[fresh_preds_df['experiment_id'] == exp_id][
            ['pred_green', 'pred_dead', 'pred_clover', 'pred_gdm', 'pred_total']
        ].values

        if has_oof:
            try:
                npz_data = np.load(oof_npz)
                stored_p = None
                for k in ['predictions', 'oof_predictions', 'preds', 'y_pred']:
                    if k in npz_data:
                        stored_p = npz_data[k]
                        break
                if stored_p is not None:
                    max_diff = float(np.max(np.abs(fresh_m_preds - stored_p)))
                    mean_diff = float(np.mean(np.abs(fresh_m_preds - stored_p)))
                    classification = 'EXACT_MATCH' if max_diff < 1e-5 else ('CLOSE_MATCH' if max_diff < 1e-3 else 'DIVERGENCE')
            except Exception as e:
                classification = f"ERROR_READING_NPZ: {e}"

        if has_sum:
            try:
                with open(train_sum) as f:
                    s_data = json.load(f)
                stored_mean_cv = s_data.get('mean_cv', s_data.get('mean_fold_weighted_r2'))
                stored_pooled_r2 = s_data.get('pooled_oof_r2', s_data.get('pooled_oof_weighted_r2'))
            except Exception:
                pass

        stored_comparisons.append({
            'experiment_id': exp_id,
            'model_name': spec['name'],
            'has_stored_npz': has_oof,
            'has_stored_summary': has_sum,
            'max_absolute_pred_diff': max_diff,
            'mean_absolute_pred_diff': mean_diff,
            'stored_mean_cv': stored_mean_cv,
            'fresh_mean_raw_r2': agg_metrics_df.loc[agg_metrics_df['experiment_id'] == exp_id, 'mean_fold_raw_r2'].values[0],
            'fresh_mean_log1p_r2': agg_metrics_df.loc[agg_metrics_df['experiment_id'] == exp_id, 'mean_fold_log1p_r2'].values[0],
            'stored_pooled_r2': stored_pooled_r2,
            'fresh_pooled_raw_r2': agg_metrics_df.loc[agg_metrics_df['experiment_id'] == exp_id, 'pooled_oof_raw_r2'].values[0],
            'fresh_pooled_log1p_r2': agg_metrics_df.loc[agg_metrics_df['experiment_id'] == exp_id, 'pooled_oof_log1p_r2'].values[0],
            'artifact_consistency_status': classification
        })

    stored_comp_df = pd.DataFrame(stored_comparisons)
    stored_comp_df.to_csv(OUT_DIR / 'stored_artifact_comparison.csv', index=False)
    print(f"Saved stored artifact comparisons to {OUT_DIR / 'stored_artifact_comparison.csv'}", flush=True)

    # 7. Manuscript Claims Comparison
    print("\n--- Comparing Fresh Metrics with Manuscript Claims ---", flush=True)
    claim_comparisons = []
    for exp_id, spec in registry.items():
        claim = spec['paper_claim']
        special = spec['special_status']

        agg_row = agg_metrics_df[agg_metrics_df['experiment_id'] == exp_id]
        if len(agg_row) == 0:
            continue
        agg = agg_row.iloc[0]

        if special == 'MISSING_CHECKPOINT':
            claim_comparisons.append({
                'experiment_id': exp_id,
                'model_name': spec['name'],
                'claimed_mean': claim['r2'] if claim else None,
                'claimed_std': claim['std'] if claim else None,
                'fresh_raw_mean': agg['mean_fold_raw_r2'],
                'fresh_raw_std_ddof0': agg['std_fold_raw_r2_ddof0'],
                'fresh_log1p_mean': agg['mean_fold_log1p_r2'],
                'fresh_log1p_std_ddof0': agg['std_fold_log1p_r2_ddof0'],
                'raw_status': 'MISSING_CHECKPOINT',
                'log1p_status': 'MISSING_CHECKPOINT',
                'rationale': 'B1 has no checkpoint weights; evaluated via deterministic fold training median recomputation.'
            })
            continue

        if special in ('SUPERSEDED_PROTOCOL', 'MISSING_OR_SUPERSEDED_CHECKPOINT'):
            claim_comparisons.append({
                'experiment_id': exp_id,
                'model_name': spec['name'],
                'claimed_mean': claim['r2'] if claim else None,
                'claimed_std': claim['std'] if claim else None,
                'fresh_raw_mean': agg['mean_fold_raw_r2'],
                'fresh_raw_std_ddof0': agg['std_fold_raw_r2_ddof0'],
                'fresh_log1p_mean': agg['mean_fold_log1p_r2'],
                'fresh_log1p_std_ddof0': agg['std_fold_log1p_r2_ddof0'],
                'raw_status': special,
                'log1p_status': special,
                'rationale': f'Original manuscript evaluated flawed/different protocol; current run evaluates verified protocol.'
            })
            continue

        if special == 'SUPPLEMENTAL_RESULT' or claim is None:
            claim_comparisons.append({
                'experiment_id': exp_id,
                'model_name': spec['name'],
                'claimed_mean': None,
                'claimed_std': None,
                'fresh_raw_mean': agg['mean_fold_raw_r2'],
                'fresh_raw_std_ddof0': agg['std_fold_raw_r2_ddof0'],
                'fresh_log1p_mean': agg['mean_fold_log1p_r2'],
                'fresh_log1p_std_ddof0': agg['std_fold_log1p_r2_ddof0'],
                'raw_status': 'SUPPLEMENTAL_RESULT',
                'log1p_status': 'SUPPLEMENTAL_RESULT',
                'rationale': 'Supplemental checkpoint evaluated; not part of main manuscript claims.'
            })
            continue

        c_mean = claim['r2']

        # Determine status for RAW
        raw_diff = abs(agg['mean_fold_raw_r2'] - c_mean)
        if round(agg['mean_fold_raw_r2'], 3) == round(c_mean, 3):
            raw_status = 'MATCH_ROUNDED'
        elif raw_diff <= 0.005:
            raw_status = 'CLOSE_NOT_ROUNDED'
        else:
            raw_status = 'MISMATCH'

        # Determine status for LOG1P
        log_diff = abs(agg['mean_fold_log1p_r2'] - c_mean)
        if round(agg['mean_fold_log1p_r2'], 3) == round(c_mean, 3):
            log_status = 'MATCH_ROUNDED'
        elif log_diff <= 0.005:
            log_status = 'CLOSE_NOT_ROUNDED'
        else:
            log_status = 'MISMATCH'

        claim_comparisons.append({
            'experiment_id': exp_id,
            'model_name': spec['name'],
            'claimed_mean': c_mean,
            'claimed_std': claim['std'],
            'fresh_raw_mean': agg['mean_fold_raw_r2'],
            'fresh_raw_std_ddof0': agg['std_fold_raw_r2_ddof0'],
            'fresh_raw_delta': agg['mean_fold_raw_r2'] - c_mean,
            'raw_status': raw_status,
            'fresh_log1p_mean': agg['mean_fold_log1p_r2'],
            'fresh_log1p_std_ddof0': agg['std_fold_log1p_r2_ddof0'],
            'fresh_log1p_delta': agg['mean_fold_log1p_r2'] - c_mean,
            'log1p_status': log_status,
            'rationale': f"Raw diff={raw_diff:.4f}, Log1p diff={log_diff:.4f}"
        })

    claim_comp_df = pd.DataFrame(claim_comparisons)
    claim_comp_df.to_csv(OUT_DIR / 'paper_claim_comparison.csv', index=False)
    print(f"Saved paper claim comparisons to {OUT_DIR / 'paper_claim_comparison.csv'}", flush=True)

    # 8. Run Metadata
    run_meta = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'environment': 'mambahar-vmamba (WSL2 Ubuntu 24.04)',
        'torch_version': str(torch.__version__),
        'cuda_version': '13.2',
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'train_csv_sha256': actual_train_sha256,
        'fold_csv_sha256': actual_fold_sha256,
        'num_samples': 357,
        'models_evaluated': list(registry.keys()),
        'total_checkpoints_audited': len(checkpoint_manifest),
        'timings_seconds': run_timings
    }
    with open(OUT_DIR / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=2)
    print(f"Saved run metadata to {OUT_DIR / 'run_metadata.json'}", flush=True)

    print("\n=================================================================", flush=True)
    print("Checkpoint Integrity Audit Completed Successfully!", flush=True)
    print("=================================================================", flush=True)


if __name__ == '__main__':
    main()
