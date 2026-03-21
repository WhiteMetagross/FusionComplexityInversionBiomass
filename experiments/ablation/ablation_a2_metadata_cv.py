"""
A2: Metadata Ablation — 5-Fold CV (Local)

Sweeps both conditions:
  1. Image-only (USE_METADATA=False)   → baseline
  2. Image+Metadata (USE_METADATA=True) → full model

Measures the performance delta from the metadata MLP stream.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/ablation/ablation_a2_metadata_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import run_cv
from models import BiomassModelTimm

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'  # DINOv2-Large (~304M params)
    MODEL_DIR = ''   # set per condition
    OUTPUT_DIR = ''  # set per condition

    USE_METADATA = False  # toggled in sweep
    _META_COLS = None
    _META_DIM = 0

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    IMG_SIZE = 518          # DINOv2 native (patch14, 37×37 grid)
    BATCH_SIZE = 3          # Fits 8 GB VRAM
    GRAD_ACCUM_STEPS = 2   # effective batch = 3×2 = 6
    NUM_WORKERS = 4

    EPOCHS = 50
    WARMUP_EPOCHS = 5
    LR_BACKBONE = 1e-5
    LR_HEAD = 5e-4
    WD = 1e-2

    CLIP_GRAD_NORM = 1.0
    DROPOUT = 0.2
    EARLY_STOPPING_PATIENCE = 10

    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def model_factory(cfg):
    use_meta = getattr(cfg, 'USE_METADATA', False)
    meta_dim = getattr(cfg, '_META_DIM', 0)
    return BiomassModelTimm(cfg.MODEL_NAME, cfg.DROPOUT, use_mamba_ssm=True,
                            use_metadata=use_meta, meta_input_dim=meta_dim)


if __name__ == '__main__':
    conditions = [
        ('image_only',     False),
        ('image_metadata', True),
    ]

    for label, use_meta in conditions:
        print(f"\n{'#'*60}")
        print(f"  A2 ABLATION: {label.upper()} (USE_METADATA={use_meta})")
        print(f"{'#'*60}")

        CFG.USE_METADATA = use_meta
        base_dir = PROJ_ROOT / 'local_training_cv' / 'output' / f'A2_Metadata_{label}'
        CFG.MODEL_DIR = str(base_dir)
        CFG.OUTPUT_DIR = str(base_dir)

        run_cv(CFG, model_factory, use_compile=True)

    print(f"\n{'='*60}")
    print("A2 METADATA ABLATION COMPLETE — both conditions trained.")
    print(f"{'='*60}")
