"""
E7: DINOv3 ViT-L with four gated depthwise blocks, five fold CV.

Provides the four block point in the matched fusion depth comparison. The
verified mean fold R2 is 0.8136.

Usage (WSL):
    conda activate mambahar
    python experiments/ablation/E7_QuadBlock_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from engine import run_cv
from models import BiomassModelTimm, GatedDepthwiseConvBlock

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / 'csiro-biomass'


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / 'train.csv')
    TRAIN_IMAGE_DIR = str(DATA_DIR / 'train')

    MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'

    MODEL_DIR = str(PROJ_ROOT / 'output' / 'E7_QuadBlock')
    OUTPUT_DIR = str(PROJ_ROOT / 'output' / 'E7_QuadBlock')

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
    # Replace 2-block fusion with 4-block fusion
    nf = model.backbone.num_features
    model.fusion = nn.Sequential(
        GatedDepthwiseConvBlock(nf, dropout=cfg.DROPOUT),
        GatedDepthwiseConvBlock(nf, dropout=cfg.DROPOUT),
        GatedDepthwiseConvBlock(nf, dropout=cfg.DROPOUT),
        GatedDepthwiseConvBlock(nf, dropout=cfg.DROPOUT),
    )
    print(f"Fusion: 4× GatedDepthwiseConvBlock(dim={nf})")
    return model


if __name__ == '__main__':
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    print(f"E7: 4× GatedDepthwiseConv (quadruple block)")
    print(f"MODEL_DIR: {CFG.MODEL_DIR}")
    run_cv(CFG, model_factory, use_compile=False)
