"""
B4: DINOv2-Large + GatedDepthwiseConv Fusion — Resume from Kaggle Checkpoints

Continues training folds 2-4 that couldn't finish on Kaggle due to GPU limits.
Folds 0-1 are already done (best R² loaded from Kaggle done.json).

Architecture (matches Kaggle exactly):
  - Backbone: vit_large_patch14_dinov2.lvd142m (DINOv2-Large, 1024-d)
  - Fusion: 2× GatedDepthwiseConvBlock (no Mamba SSM CUDA kernels)
  - No metadata MLP

torch.compile is DISABLED — gradient checkpointing on DINOv2-Large creates
too many graph breaks that make compile 8x slower than plain PyTorch.

Usage (WSL):
    conda activate mambahar
    python experiments/baselines/B4_DINOv2_Metadata_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from engine import run_cv
from models import BiomassModelTimm

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'

# Kaggle checkpoint directory (folds 0-1 are done, 2-4 need training)
KAGGLE_CKPT_DIR = PROJ_ROOT / 'output' / 'B4Checkpoints' / 'models'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'

    # Point MODEL_DIR at the Kaggle checkpoint directory
    MODEL_DIR = str(KAGGLE_CKPT_DIR)
    OUTPUT_DIR = str(KAGGLE_CKPT_DIR)

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    # Must match Kaggle model: no metadata (checkpoint has no meta weights)
    USE_METADATA = False

    IMG_SIZE = 518          # DINOv2 native (patch14, 37×37 grid)
    BATCH_SIZE = 4          # Increased from 3 — fits 8.6 GB VRAM with grad_ckpt + AMP
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
    return BiomassModelTimm(cfg.MODEL_NAME, cfg.DROPOUT, use_mamba_ssm=False,
                            use_metadata=False, meta_input_dim=0)


if __name__ == '__main__':
    print(f"MODEL_DIR: {CFG.MODEL_DIR}")
    print(f"Kaggle checkpoints present:")
    for f in sorted(Path(CFG.MODEL_DIR).glob('fold*')):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)" if f.is_file() else f"  {f.name}/")
    print()

    run_cv(CFG, model_factory, use_compile=False)
