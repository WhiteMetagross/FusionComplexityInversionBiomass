"""
E4: DINOv3-ViT-L + Mean Pooling (NO Fusion) — 5-Fold CV

Removes ALL fusion blocks. Simply concatenates left/right backbone features
and pools directly. Tests whether ANY fusion module helps beyond baseline
pooling of DINOv3 features.

If E4 ≈ Proposed (0.903) → fusion blocks are unnecessary.
If E4 << Proposed → even simple local fusion adds value.

Usage (WSL):
    conda activate mambahar
    python local_training_cv/ablation/E4_NoFusion_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys, os
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

    MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'

    MODEL_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'E4_NoFusion')
    OUTPUT_DIR = str(PROJ_ROOT / 'local_training_cv' / 'output' / 'E4_NoFusion')

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    USE_METADATA = False

    IMG_SIZE = 512
    BATCH_SIZE = 5
    NUM_WORKERS = 2

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
    import torch.nn as nn
    model = BiomassModelTimm(
        model_name=cfg.MODEL_NAME,
        dropout=cfg.DROPOUT,
        use_mamba_ssm=False,
        pretrained=True,
        use_metadata=False,
        meta_input_dim=0,
    )
    # Replace fusion with identity — pure mean pooling of concatenated features
    model.fusion = nn.Identity()
    print("Fusion: REPLACED with nn.Identity (no fusion)")
    return model


if __name__ == '__main__':
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    print(f"E4: NO FUSION (Mean Pool Only)")
    print(f"MODEL_DIR: {CFG.MODEL_DIR}")
    run_cv(CFG, model_factory, use_compile=False)
