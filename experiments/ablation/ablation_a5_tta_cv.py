"""
A5: Test-Time Augmentation (TTA) Ablation — 5-Fold CV (Local)

Sweeps both conditions:
  1. No TTA (single-pass inference)     → baseline
  2. TTA enabled (hflip + vflip ensemble) → augmented

Measures TTA's impact on inference-time robustness. The training protocol
is identical for both — TTA only affects validation-time prediction.

Usage (WSL):
    conda activate mambahar
    python experiments/ablation/ablation_a5_tta_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from engine import run_cv
from models import BiomassModelTimm

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'  # DINOv2-Large
    MODEL_DIR = ''   # set per condition
    OUTPUT_DIR = ''  # set per condition

    USE_TTA = False  # toggled in sweep

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    IMG_SIZE = 518          # DINOv2 native
    BATCH_SIZE = 3
    GRAD_ACCUM_STEPS = 2
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
    return BiomassModelTimm(cfg.MODEL_NAME, cfg.DROPOUT, use_mamba_ssm=True)


if __name__ == '__main__':
    conditions = [
        ('no_tta', False),
        ('tta',    True),
    ]

    for label, use_tta in conditions:
        print(f"\n{'#'*60}")
        print(f"  A5 ABLATION: USE_TTA={use_tta}")
        print(f"{'#'*60}")

        CFG.USE_TTA = use_tta
        base_dir = PROJ_ROOT / 'output' / f'A5_TTA_{label}'
        CFG.MODEL_DIR = str(base_dir)
        CFG.OUTPUT_DIR = str(base_dir)

        run_cv(CFG, model_factory, use_compile=True)

    print(f"\n{'='*60}")
    print("A5 TTA ABLATION COMPLETE — both conditions trained.")
    print(f"{'='*60}")
