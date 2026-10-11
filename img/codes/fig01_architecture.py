"""Regenerate Figure 1 from the reviewed source and raster masters."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original architecture SVG.")
    parser.add_argument("--canonical-png", type=Path, required=True, help="Original architecture PNG.")
    parser.add_argument("--rendered-png", type=Path, required=True, help="Rasterized revised SVG.")
    args = parser.parse_args()
    subprocess.run(
        [
            str(Path(__import__("sys").executable)),
            str(REPO_ROOT / "analysis" / "revise_architecture_asset.py"),
            "--source",
            str(args.source),
            "--canonical-png",
            str(args.canonical_png),
            "--rendered-png",
            str(args.rendered_png),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

