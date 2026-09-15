#!/usr/bin/env python3
"""Canonical entry point for checkpoint only evaluation suites.

This command performs inference only. It never trains, updates checkpoints, or
uses optimizer state. Individual suites remain separate modules because their
model registries and outputs differ.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
SUITES = {
    "integrity": EXPERIMENTS_DIR / "evaluate_checkpoint_integrity.py",
    "metadata": EXPERIMENTS_DIR / "ablation" / "evaluate_metadata_modes.py",
    "additional": EXPERIMENTS_DIR / "ablation" / "evaluate_additional_kaggle_models.py",
}


def run_suite(name: str) -> None:
    """Execute one checkpoint only evaluation suite without forwarding CLI args."""
    path = SUITES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} evaluator: {path}")

    prior_argv = sys.argv[:]
    try:
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = prior_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run checkpoint only biomass evaluation suites."
    )
    parser.add_argument(
        "--suite",
        choices=[*SUITES, "all"],
        default="integrity",
        help="Evaluation suite. 'all' runs integrity, metadata, then additional.",
    )
    args = parser.parse_args()

    selected = list(SUITES) if args.suite == "all" else [args.suite]
    for suite in selected:
        print(f"\n=== checkpoint evaluation: {suite} ===", flush=True)
        run_suite(suite)


if __name__ == "__main__":
    main()
