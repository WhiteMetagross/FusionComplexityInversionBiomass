"""
Proposed Model 2: DINOv3-ViT-L + Cross-View Gated Attention (CVGA) — 5-Fold CV (Local)

Based on B5 backbone, replaces GatedDepthwiseConv with CVGABlock
(weight-tied bidirectional cross-view attention with sigmoid gating).

Key advantages over GatedDepthwiseConv:
  - 100% cross-view interaction (every left token attends to every right token)
  - Stereo correspondence via QK^T similarity
  - Fully AMP and torch.compile compatible (no fp32 forcing)

Usage (WSL):
    conda activate mambahar
    python experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py
"""

# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

import sys, os, json
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

    MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'

    MODEL_DIR = str(PROJ_ROOT / 'output' / 'Proposed_CVGA')
    OUTPUT_DIR = str(PROJ_ROOT / 'output' / 'Proposed_CVGA')

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

    USE_METADATA = True

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
    use_meta = getattr(cfg, 'USE_METADATA', True)
    meta_dim = getattr(cfg, '_META_DIM', 23) if use_meta else 0
    model = BiomassModelTimm(
        model_name=cfg.MODEL_NAME,
        dropout=cfg.DROPOUT,
        use_cvga=True,
        pretrained=True,
        use_metadata=cfg.USE_METADATA,
        meta_input_dim=meta_dim
    )
    return model


if __name__ == '__main__':
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    print(f"MODEL_DIR: {CFG.MODEL_DIR}")

    print(f"Checkpoint status:")
    for f in sorted(Path(CFG.MODEL_DIR).glob('fold*')):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)" if f.is_file() else f"  {f.name}/")
    print()

    # CVGA is AMP-safe but torch.compile on the full 304M-param model is
    # counterproductive: ~80s compilation overhead per fold isn't amortized on
    # 57 batches/epoch.  Flash attention via F.scaled_dot_product_attention
    # already gives the SDPA speedup without compile.
    run_cv(CFG, model_factory, use_compile=False)
