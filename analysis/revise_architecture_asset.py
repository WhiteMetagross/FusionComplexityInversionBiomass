"""Create the revised architecture asset from the canonical hand edited SVG.

The script performs only evidence required edits. It removes obsolete proposed
method emphasis and the unsupported complexity ordering while preserving every
other SVG element and transform from the canonical paper figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from lxml import etree
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def remove_by_id(root: etree._Element, element_id: str) -> None:
    nodes = root.xpath(f'//*[@id="{element_id}"]')
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one element with id={element_id!r}, found {len(nodes)}")
    parent = nodes[0].getparent()
    if parent is None:
        raise RuntimeError(f"Cannot remove root element id={element_id!r}")
    parent.remove(nodes[0])


def suppress_text_by_id(root: etree._Element, element_id: str) -> None:
    """Remove rendered glyph instances but retain globally reused path defs."""

    nodes = root.xpath(f'//*[@id="{element_id}"]')
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one element with id={element_id!r}, found {len(nodes)}")
    node = nodes[0]
    for use_node in node.xpath('.//*[local-name()="use"]'):
        parent = use_node.getparent()
        if parent is not None:
            parent.remove(use_node)
    for comment in node.xpath(".//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)


def replace_score(root: etree._Element) -> None:
    """Change the stale GatedDepthwiseConv score from 0.903 to 0.827."""

    score_group = root.xpath('//*[@id="text_86"]')
    if len(score_group) != 1:
        raise RuntimeError("Canonical score label text_86 was not found")

    comments = score_group[0].xpath(".//comment()")
    if len(comments) != 1 or "R²=0.903" not in comments[0].text:
        raise RuntimeError("Canonical 0.903 score comment was not found")
    comments[0].text = comments[0].text.replace("R²=0.903", "R²=0.827")

    replacements = {
        "use998": "#DejaVuSans-Oblique-38",
        "use999": "#DejaVuSans-Oblique-32",
        "use1000": "#DejaVuSans-Oblique-37",
    }
    for element_id, glyph in replacements.items():
        nodes = root.xpath(f'//*[@id="{element_id}"]')
        if len(nodes) != 1:
            raise RuntimeError(f"Canonical score glyph {element_id} was not found")
        nodes[0].set(XLINK_HREF, glyph)


def revise(source: Path, destination: Path) -> None:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(source), parser)
    root = tree.getroot()

    for element_id in (
        "patch_2",  # Yellow highlight border.
        "patch_14",  # Unsupported complexity arrow.
    ):
        remove_by_id(root, element_id)
    for element_id in (
        "text_93",  # Star and Proposed label.
        "text_18",  # Unsupported complexity caption.
        "text_19",  # Unsupported Best R2 label.
        "text_20",  # Unsupported Worst R2 label.
    ):
        suppress_text_by_id(root, element_id)
    replace_score(root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(destination),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )


def revise_raster(canonical_png: Path, rendered_png: Path, destination_png: Path) -> None:
    """Preserve canonical pixels outside the exact edited figure regions."""

    canonical = np.asarray(Image.open(canonical_png).convert("RGB"))
    rendered = np.asarray(Image.open(rendered_png).convert("RGB"))
    if rendered.shape[0] == canonical.shape[0] + 1 and rendered.shape[1] == canonical.shape[1]:
        rendered = rendered[: canonical.shape[0], :, :]
    if canonical.shape != rendered.shape:
        raise RuntimeError(f"Raster shape mismatch: {canonical.shape} vs {rendered.shape}")

    edit_mask = np.zeros(canonical.shape[:2], dtype=np.uint8)

    # Yellow border and gold star/Proposed label. Include antialiased edge pixels.
    region = canonical[1700:3340, 500:2200]
    colored = (
        (region[..., 0] > 145)
        & (region[..., 1] > 95)
        & (region[..., 2] < 135)
        & ((region[..., 0] - region[..., 2]) > 80)
    ).astype(np.uint8)
    colored = cv2.dilate(colored, np.ones((5, 5), dtype=np.uint8), iterations=1)
    edit_mask[1700:3340, 500:2200] |= colored

    # Corrected score, obsolete best/worst labels, and obsolete complexity arrow.
    edit_mask[1940:2085, 930:1900] = 1
    edit_mask[3160:3340, 570:1050] = 1
    edit_mask[3100:3380, 6900:canonical.shape[1]] = 1
    edit_mask[3340:canonical.shape[0], 400:canonical.shape[1]] = 1

    revised = canonical.copy()
    active = edit_mask.astype(bool)
    revised[active] = rendered[active]
    if not np.array_equal(revised[~active], canonical[~active]):
        raise RuntimeError("Pixels outside declared edit regions changed")

    destination_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(revised).save(destination_png, optimize=True)
    changed = int(np.any(revised != canonical, axis=2).sum())
    total = int(canonical.shape[0] * canonical.shape[1])
    print(f"Saved pixel preserving architecture PNG: {destination_png}")
    print(f"Changed pixels: {changed}/{total} ({100.0 * changed / total:.3f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "img" / "fig_architecture.svg")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "newPaper" / "img" / "svg" / "fig01_architecture.svg",
    )
    parser.add_argument("--canonical-png", type=Path, default=REPO_ROOT / "img" / "fig_architecture.png")
    parser.add_argument(
        "--rendered-png",
        type=Path,
        default=WORKSPACE_ROOT / "tmp" / "architecture_edge_final.png",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=WORKSPACE_ROOT / "newPaper" / "img" / "png" / "fig01_architecture.png",
    )
    parser.add_argument("--skip-png", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revise(args.source.resolve(), args.output.resolve())
    print(f"Saved revised canonical architecture: {args.output.resolve()}")
    if not args.skip_png:
        revise_raster(
            args.canonical_png.resolve(),
            args.rendered_png.resolve(),
            args.output_png.resolve(),
        )


if __name__ == "__main__":
    main()
