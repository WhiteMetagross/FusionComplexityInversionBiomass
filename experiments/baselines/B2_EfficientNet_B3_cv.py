"""B2 rerun: protocol matched EfficientNet B3, five fold CV."""

import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from engine import run_cv
from models import BiomassModelTimm


DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT / "csiro-biomass"))
RUN_ROOT = PROJ_ROOT / "output" / "reruns_2026_09_13"


class CFG:
    BASE_PATH = str(DATA_DIR)
    TRAIN_CSV = str(DATA_DIR / "train.csv")
    TRAIN_IMAGE_DIR = str(DATA_DIR / "train")
    MODEL_NAME = "efficientnet_b3.ra2_in1k"
    MODEL_DIR = str(RUN_ROOT / "B2_EfficientNet_B3_matched")
    OUTPUT_DIR = MODEL_DIR
    FOLD_FILE = str(RUN_ROOT / "folds_seed17.csv")

    SEED = 17
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]
    USE_METADATA = False
    CV_STRATEGY = "stratified_group"

    IMG_SIZE = 448
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    LR_BACKBONE = 1e-5
    LR_HEAD = 5e-4
    WD = 1e-2
    CLIP_GRAD_NORM = 1.0
    DROPOUT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    LOSS_FN = "huber"
    HUBER_BETA = 5.0
    VAL_EVERY_N_EPOCHS = 1
    TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


def model_factory(cfg):
    return BiomassModelTimm(
        cfg.MODEL_NAME,
        dropout=cfg.DROPOUT,
        use_mamba_ssm=False,
        pretrained=True,
        use_metadata=False,
        img_size=cfg.IMG_SIZE,
    )


if __name__ == "__main__":
    if not Path(CFG.TRAIN_CSV).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. Set BIOMASS_DATA_DIR to the extracted dataset."
        )
    if os.environ.get("BIOMASS_SMOKE_TEST") == "1":
        CFG.MODEL_DIR = str(RUN_ROOT / "B2_EfficientNet_B3_smoke")
        CFG.OUTPUT_DIR = CFG.MODEL_DIR
        CFG.FOLDS_TO_TRAIN = [0]
        CFG.EPOCHS = 1
        CFG.WARMUP_EPOCHS = 0
    run_cv(CFG, model_factory, use_compile=True)
