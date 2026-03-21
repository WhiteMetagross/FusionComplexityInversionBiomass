"""
A6: VMamba-Base vs DINOv2-Large Head-to-Head — 5-Fold CV (Local)

Core scientific ablation comparing SSM linear complexity (VMamba-Base)
against ViT quadratic self-attention (DINOv2-Large).
Uses mamba_ssm + triton for hardware-accelerated SSM fusion.

Change CFG.MODEL_NAME to switch between architectures:
  'vmamba_base'                     → BiomassModelVMamba (VSSM backbone)
  'vit_large_patch14_dinov2.lvd142m' → BiomassModelTimm  (timm backbone)

Usage (WSL):
    conda activate mambahar
    python local_training_cv/ablation/ablation_a6_vmamba_vs_dinov2_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import run_cv
from models import BiomassModelVMamba, BiomassModelTimm

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    # Switch between 'vmamba_base' and DINOv2 for the ablation
    MODEL_NAME = 'vmamba_base'
    VMAMBA_VARIANT = 'vmamba_base'
    MODEL_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'A6_VMamba_vs_DINOv2')
    OUTPUT_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'A6_VMamba_vs_DINOv2')

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
    if 'vmamba' in cfg.MODEL_NAME.lower():
        return BiomassModelVMamba(
            pretrained_path='',
            dropout=cfg.DROPOUT,
            use_mamba_ssm=True,
            variant=getattr(cfg, 'VMAMBA_VARIANT', 'vmamba_base'),
        )
    else:
        return BiomassModelTimm(cfg.MODEL_NAME, cfg.DROPOUT, use_mamba_ssm=True)


if __name__ == '__main__':
    run_cv(CFG, model_factory, use_compile=True)
