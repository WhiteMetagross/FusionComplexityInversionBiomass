"""Validate the curated figure set for the revised paper."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = REPO_ROOT / "img"
FORBIDDEN_VISIBLE_TEXT = ("r1", "x1", "proposed", "0.903", "complexity inversion")


def main() -> None:
    pngs = sorted((FIGURE_ROOT / "png").glob("*.png"))
    svgs = sorted((FIGURE_ROOT / "svg").glob("*.svg"))

    if len(pngs) != 12 or len(svgs) != 12:
        raise RuntimeError(f"Expected 12 PNG and 12 SVG files, found {len(pngs)} and {len(svgs)}")
    if [path.stem for path in pngs] != [path.stem for path in svgs]:
        raise RuntimeError("PNG and SVG figure names do not match")

    dimensions: dict[str, tuple[int, int]] = {}
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions[path.stem] = image.size

    violations: list[tuple[str, str]] = []
    for path in svgs:
        root = ET.parse(path).getroot()
        visible_text = " ".join("".join(root.itertext()).split()).lower()
        violations.extend(
            (path.name, forbidden)
            for forbidden in FORBIDDEN_VISIBLE_TEXT
            if forbidden in visible_text
        )
    if violations:
        raise RuntimeError(f"Forbidden visible labels found: {violations}")

    print(f"Validated {len(pngs)} PNG/SVG pairs.")
    for stem, size in dimensions.items():
        print(f"{stem}: {size[0]} x {size[1]} px")


if __name__ == "__main__":
    main()
