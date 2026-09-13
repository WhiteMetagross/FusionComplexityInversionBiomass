"""
A4: Loss Function Ablation (MSE vs Huber) — 5-Fold CV (Local)

Sweeps both conditions:
  1. MSE loss (L2, sensitive to outliers)
  2. Huber / SmoothL1 loss (robust to outlier annotations)

All other components remain identical.

Usage (WSL):
    conda activate mambahar
    python experiments/ablation/ablation_a4_loss_function_cv.py
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

    LOSS_FN = 'huber'       # toggled in sweep
    HUBER_BETA = 5.0

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
        ('mse',   'mse'),
        ('huber', 'huber'),
    ]

    for label, loss_fn in conditions:
        print(f"\n{'#'*60}")
        print(f"  A4 ABLATION: LOSS_FN={loss_fn}")
        print(f"{'#'*60}")

        CFG.LOSS_FN = loss_fn
        base_dir = PROJ_ROOT / 'output' / f'A4_Loss_{label}'
        CFG.MODEL_DIR = str(base_dir)
        CFG.OUTPUT_DIR = str(base_dir)

        run_cv(CFG, model_factory, use_compile=True)

    print(f"\n{'='*60}")
    print("A4 LOSS FUNCTION ABLATION COMPLETE — both losses trained.")
    print(f"{'='*60}")
