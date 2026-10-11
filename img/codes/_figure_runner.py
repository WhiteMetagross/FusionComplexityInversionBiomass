"""Shared runner used by the per-figure generation entry points."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "analysis"
PNG_DIR = REPO_ROOT / "img" / "png"
SVG_DIR = REPO_ROOT / "img" / "svg"


def _load_results_generator():
    sys.path.insert(0, str(ANALYSIS_DIR))
    import generate_revised_results_figures as generator

    return generator


def generate_result_figure(function_name: str) -> None:
    """Generate one main result figure through its canonical function."""

    generator = _load_results_generator()
    generator.PNG_DIR = PNG_DIR
    generator.SVG_DIR = SVG_DIR
    results = generator.load_results()
    getattr(generator, function_name)(results)


def generate_dataset_figure(source_name: str, destination_name: str) -> None:
    """Run the dataset analysis and retain one named appendix figure."""

    with tempfile.TemporaryDirectory(prefix="biomass_figure_") as temporary:
        output_root = Path(temporary)
        previous = os.environ.get("PAPER_FIGURE_OUTPUT_DIR")
        previous_jitter = os.environ.get("PAPER_FIGURE_RECOMPUTE_UNSEEDED_JITTER")
        os.environ["PAPER_FIGURE_OUTPUT_DIR"] = str(output_root)
        os.environ["PAPER_FIGURE_RECOMPUTE_UNSEEDED_JITTER"] = "1"
        try:
            runpy.run_path(str(ANALYSIS_DIR / "dataset_analysis_v3.py"), run_name="__main__")
        finally:
            if previous is None:
                os.environ.pop("PAPER_FIGURE_OUTPUT_DIR", None)
            else:
                os.environ["PAPER_FIGURE_OUTPUT_DIR"] = previous
            if previous_jitter is None:
                os.environ.pop("PAPER_FIGURE_RECOMPUTE_UNSEEDED_JITTER", None)
            else:
                os.environ["PAPER_FIGURE_RECOMPUTE_UNSEEDED_JITTER"] = previous_jitter

        PNG_DIR.mkdir(parents=True, exist_ok=True)
        SVG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_root / "png" / f"{source_name}.png", PNG_DIR / f"{destination_name}.png")
        shutil.copyfile(output_root / "svg" / f"{source_name}.svg", SVG_DIR / f"{destination_name}.svg")

