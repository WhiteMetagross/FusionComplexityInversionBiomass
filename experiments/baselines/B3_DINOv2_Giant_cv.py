"""B3 rerun: DINOv2 Base supervised frozen dual view probe."""

import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from frozen_probe import ProbeConfig, run_probe


DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT / "csiro-biomass"))
RUN_ROOT = PROJ_ROOT / "output" / "reruns_2026_09_13"


if __name__ == "__main__":
    config = ProbeConfig(
        run_name="B3_DINOv2_Base",
        model_id="facebook/dinov2-base",
        data_dir=str(DATA_DIR),
        output_dir=str(RUN_ROOT / "B3_DINOv2_Base_probe"),
        fold_file=str(RUN_ROOT / "folds_seed17.csv"),
        extraction_batch_size=8,
    )
    run_probe(config)
