"""Verify and export the exact PNG and SVG assets used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("paper_figure_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def load_manifest() -> list[dict[str, object]]:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)["figures"]


def verify_assets(asset_dir: Path) -> list[dict[str, object]]:
    failures: list[str] = []
    figures = load_manifest()
    for figure in figures:
        name = str(figure["name"])
        png = asset_dir / f"{name}.png"
        svg = asset_dir / f"{name}.svg"
        for path in (png, svg):
            if not path.is_file():
                failures.append(f"Missing: {path}")
        if not png.is_file() or not svg.is_file():
            continue
        actual_size = png_size(png)
        expected_size = (int(figure["width"]), int(figure["height"]))
        if actual_size != expected_size:
            failures.append(f"Dimension mismatch: {png}: {actual_size} != {expected_size}")
        for path, key in ((png, "png_sha256"), (svg, "svg_sha256")):
            actual_hash = sha256(path)
            expected_hash = str(figure[key])
            if actual_hash != expected_hash:
                failures.append(f"SHA256 mismatch: {path}: {actual_hash} != {expected_hash}")
    if failures:
        raise RuntimeError("\n".join(failures))
    return figures


def export_assets(asset_dir: Path, output_dir: Path) -> None:
    figures = verify_assets(asset_dir)
    png_dir = output_dir / "png"
    svg_dir = output_dir / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        name = str(figure["name"])
        shutil.copyfile(asset_dir / f"{name}.png", png_dir / f"{name}.png")
        shutil.copyfile(asset_dir / f"{name}.svg", svg_dir / f"{name}.svg")
    verify_assets_in_split_output(figures, png_dir, svg_dir)
    print(f"Verified and exported {len(figures)} exact PNG and SVG figure pairs to {output_dir}")


def verify_assets_in_split_output(
    figures: list[dict[str, object]], png_dir: Path, svg_dir: Path
) -> None:
    failures: list[str] = []
    for figure in figures:
        name = str(figure["name"])
        png = png_dir / f"{name}.png"
        svg = svg_dir / f"{name}.svg"
        if sha256(png) != str(figure["png_sha256"]):
            failures.append(f"Exported PNG mismatch: {png}")
        if sha256(svg) != str(figure["svg_sha256"]):
            failures.append(f"Exported SVG mismatch: {svg}")
    if failures:
        raise RuntimeError("\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or export exact paper figure assets."
    )
    parser.add_argument("--asset-dir", type=Path, default=REPO_ROOT / "img")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "output" / "paper_figures"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = verify_assets(args.asset_dir)
    if args.verify_only:
        print(f"Verified {len(figures)} exact PNG and SVG figure pairs in {args.asset_dir}")
        return
    export_assets(args.asset_dir, args.output_dir)


if __name__ == "__main__":
    main()
