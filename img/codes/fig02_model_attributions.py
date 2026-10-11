"""Regenerate Figure 2 with the canonical attribution command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    if "--png" not in arguments:
        arguments.extend(["--png", str(REPO_ROOT / "img" / "png" / "fig02_model_attributions.png")])
    if "--svg" not in arguments:
        arguments.extend(["--svg", str(REPO_ROOT / "img" / "svg" / "fig02_model_attributions.svg")])
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "analysis" / "generate_model_attributions.py"), *arguments],
        check=True,
    )

