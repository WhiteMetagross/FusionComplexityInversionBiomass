"""
Baseline 6: VMamba-Base + Mamba SSM Fusion — 5-Fold CV (Local)

VMamba-Base (~89M params) with 2D Selective State Space Model backbone
and real Mamba SSM fusion blocks. All other components (dual-view split,
compositional heads) match B4/B5 baselines for fair comparison.

Uses mamba_ssm + triton + torch.compile for acceleration.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/baselines/B6_VMamba_Base_Metadata_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import run_cv
from models import BiomassModelVMamba

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vmamba_base'
    MODEL_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'B6_VMamba_Base')
    OUTPUT_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'B6_VMamba_Base')

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    USE_METADATA = True

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
    use_meta = getattr(cfg, 'USE_METADATA', False)
    meta_dim = getattr(cfg, '_META_DIM', 0)
    return BiomassModelVMamba(pretrained_path='', dropout=cfg.DROPOUT, use_mamba_ssm=True,
                              use_metadata=use_meta, meta_input_dim=meta_dim)


if __name__ == '__main__':
    run_cv(CFG, model_factory, use_compile=True)
