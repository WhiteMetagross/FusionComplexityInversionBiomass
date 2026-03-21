"""
B5: DINOv3-ViT-L + GatedDepthwiseConv Fusion — Resume from Kaggle Checkpoints

Continues training fold 4 that couldn't finish on Kaggle due to GPU limits.
Folds 0-3 are already done (done.json markers created from Kaggle best checkpoints).

Architecture (matches Kaggle exactly):
  - Backbone: vit_large_patch16_dinov3.lvd1689m (DINOv3-ViT-L, 1024-d)
  - Fusion: 2× GatedDepthwiseConvBlock (no Mamba SSM CUDA kernels)
  - No metadata MLP

torch.compile is DISABLED — gradient checkpointing on DINOv3-ViT-L creates
too many graph breaks that make compile 8x slower than plain PyTorch.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/baselines/B5_DINOv3_ViT_L_Metadata_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import run_cv
from models import BiomassModelTimm

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'

# Kaggle checkpoint directory (folds 0-3 have best models, fold 4 needs training)
KAGGLE_CKPT_DIR = PROJ_ROOT / 'output' / 'B5Checkpoints' / 'models'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'

    # Point MODEL_DIR at the Kaggle checkpoint directory
    MODEL_DIR = str(KAGGLE_CKPT_DIR)
    OUTPUT_DIR = str(KAGGLE_CKPT_DIR)

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    # Must match Kaggle model: no metadata (checkpoint has no meta weights)
    USE_METADATA = False

    IMG_SIZE = 512
    BATCH_SIZE = 5          # DINOv3-L is slightly smaller than DINOv2-L, fits bs=5
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


def _ensure_done_markers(ckpt_dir):
    """Create fold*_done.json for completed folds that only have best.pth."""
    ckpt_dir = Path(ckpt_dir)
    for fold in range(5):
        best_path = ckpt_dir / f"fold{fold}_best.pth"
        done_path = ckpt_dir / f"fold{fold}_done.json"
        if best_path.exists() and not done_path.exists():
            done_data = {"fold": fold, "best_r2": -1.0, "best_epoch": -1,
                         "note": "Completed on Kaggle, R² recovered on local validation"}
            with open(done_path, 'w') as f:
                json.dump(done_data, f, indent=2)
            print(f"[Resume] Created {done_path.name} (Kaggle fold {fold} complete)")


if __name__ == '__main__':
    print(f"MODEL_DIR: {CFG.MODEL_DIR}")

    # Create done markers for folds 0-3 (completed on Kaggle)
    _ensure_done_markers(CFG.MODEL_DIR)

    print(f"Checkpoint status:")
    for f in sorted(Path(CFG.MODEL_DIR).glob('fold*')):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)" if f.is_file() else f"  {f.name}/")
    print()

    run_cv(CFG, model_factory, use_compile=False)
