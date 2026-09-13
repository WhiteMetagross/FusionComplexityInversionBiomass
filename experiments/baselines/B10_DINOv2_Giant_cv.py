"""B10 rerun: DINOv2 Giant probe, with an authorized Large fallback."""

import gc
import json
import os
import sys
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from frozen_probe import ProbeConfig, SlowModelError, run_probe


DATA_DIR = Path(os.environ.get("BIOMASS_DATA_DIR", PROJ_ROOT / "csiro-biomass"))
RUN_ROOT = PROJ_ROOT / "output" / "reruns_2026_09_13"
GIANT_ID = "facebook/dinov2-giant"
LARGE_ID = "facebook/dinov2-large"


def make_config(model_id: str, run_name: str, directory: str, batch_size: int) -> ProbeConfig:
    return ProbeConfig(
        run_name=run_name,
        model_id=model_id,
        data_dir=str(DATA_DIR),
        output_dir=str(RUN_ROOT / directory),
        fold_file=str(RUN_ROOT / "folds_seed17.csv"),
        extraction_batch_size=batch_size,
        max_extraction_hours=2.0,
    )


if __name__ == "__main__":
    giant_config = make_config(
        GIANT_ID, "B10_DINOv2_Giant", "B10_DINOv2_Giant_probe", 1
    )
    try:
        run_probe(giant_config)
    except (RuntimeError, SlowModelError) as error:
        message = str(error)
        is_resource_failure = isinstance(error, SlowModelError) or "out of memory" in message.lower()
        if not is_resource_failure:
            raise
        error.__traceback__ = None
        fallback_dir = RUN_ROOT / "B10_DINOv2_Large_probe"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        (fallback_dir / "fallback_reason.json").write_text(
            json.dumps(
                {
                    "requested_model": GIANT_ID,
                    "resolved_model": LARGE_ID,
                    "reason": message,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        gc.collect()
        torch.cuda.empty_cache()
        large_config = make_config(
            LARGE_ID, "B10_DINOv2_Large_fallback", "B10_DINOv2_Large_probe", 2
        )
        run_probe(large_config)
