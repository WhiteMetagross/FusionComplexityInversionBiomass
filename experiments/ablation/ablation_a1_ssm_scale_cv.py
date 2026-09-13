"""
A1: SSM Scale Ablation (VMamba-Tiny / Small / Base) — 5-Fold CV (Local)

Sweeps all three VMamba backbone scales to study how SSM model scale
affects biomass estimation.  All other components (Mamba SSM fusion,
heads, training protocol) remain identical.

Uses mamba_ssm + triton for hardware-accelerated SSM fusion.

Usage (WSL):
    conda activate mambahar
    python experiments/ablation/ablation_a1_ssm_scale_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from engine import run_cv
from models import BiomassModelVMamba

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vmamba_base'          # Updated per-variant in the sweep loop
    VMAMBA_VARIANT = 'vmamba_base'      # Updated per-variant in the sweep loop
    MODEL_DIR = str(PROJ_ROOT / 'output' / 'A1_SSM_Scale')
    OUTPUT_DIR = str(PROJ_ROOT / 'output' / 'A1_SSM_Scale')

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    IMG_SIZE = 512
    BATCH_SIZE = 4          # Fits 8 GB VRAM with gradient checkpointing
    GRAD_ACCUM_STEPS = 2   # effective batch = 4×2 = 8
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
    return BiomassModelVMamba(
        pretrained_path='',
        dropout=cfg.DROPOUT,
        use_mamba_ssm=True,
        variant=cfg.VMAMBA_VARIANT,
    )


if __name__ == '__main__':
    variants = ['vmamba_tiny', 'vmamba_small', 'vmamba_base']

    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"  A1 ABLATION: {variant.upper()}")
        print(f"{'#'*60}")

        # Update CFG for this variant
        CFG.MODEL_NAME = variant
        CFG.VMAMBA_VARIANT = variant
        base_dir = PROJ_ROOT / 'output' / f'A1_SSM_Scale_{variant}'
        CFG.MODEL_DIR = str(base_dir)
        CFG.OUTPUT_DIR = str(base_dir)

        run_cv(CFG, model_factory, use_compile=True)

    print(f"\n{'='*60}")
    print("A1 SCALE SWEEP COMPLETE — all three variants trained.")
    print(f"{'='*60}")
