#!/usr/bin/env python3
"""
Create a clean, verified zip archive of c-series-biomass-models.
"""

import os
import sys
import shutil
import hashlib
import zipfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
SRC_DIR = CHECKPOINTS_DIR / "c-series-biomass-models"
ZIPS_DIR = REPO_ROOT / "zips"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=================================================================")
    print("Creating Zip Archive for c-series-biomass-models")
    print("=================================================================")

    assert SRC_DIR.exists(), f"Source directory does not exist: {SRC_DIR}"

    ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    target_zip = ZIPS_DIR / "c-series-biomass-models.zip"

    if target_zip.exists():
        target_zip.unlink()

    # Collect files to archive
    # Exclude symlinks so zip is clean and portable across Windows/Linux/Kaggle
    files_to_zip = []
    total_uncompressed_bytes = 0

    for root, dirs, files in os.walk(SRC_DIR, followlinks=False):
        # Exclude symlink directories
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for f in files:
            p = Path(root) / f
            if not p.is_symlink():
                rel = p.relative_to(SRC_DIR)
                arcname = f"c-series-biomass-models/{rel.as_posix()}"
                files_to_zip.append((p, arcname))
                total_uncompressed_bytes += p.stat().st_size

    print(f"Found {len(files_to_zip)} files ({total_uncompressed_bytes / (1024*1024):.2f} MB uncompressed)")

    t0 = time.perf_counter()
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p, arcname in files_to_zip:
            zf.write(p, arcname=arcname)
    elapsed = time.perf_counter() - t0

    zip_size = target_zip.stat().st_size
    sha256 = compute_sha256(target_zip)

    print(f"\nCreated archive: {target_zip}")
    print(f"Compressed size: {zip_size / (1024*1024):.2f} MB ({zip_size} bytes)")
    print(f"Compression ratio: {total_uncompressed_bytes / zip_size:.2f}x")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"SHA256: {sha256}")

    # Copy / hardlink to checkpoints/ as requested by user
    chk_zip = CHECKPOINTS_DIR / "c-series-biomass-models.zip"
    if chk_zip.exists():
        chk_zip.unlink()
    shutil.copy2(target_zip, chk_zip)
    print(f"Copied to checkpoints location: {chk_zip}")

    # Also copy to root for easy user discovery
    root_zip = REPO_ROOT / "c-series-biomass-models.zip"
    if root_zip.exists():
        root_zip.unlink()
    shutil.copy2(target_zip, root_zip)
    print(f"Copied to repo root: {root_zip}")

    # Test testzip()
    print("\nVerifying zip integrity with testzip()...")
    with zipfile.ZipFile(target_zip, "r") as zf:
        bad_file = zf.testzip()
        assert bad_file is None, f"Corrupted file in zip: {bad_file}"
        namelist = zf.namelist()
        print(f"Integrity check passed! {len(namelist)} entries in zip.")

    print("\nZip creation certified complete!")


if __name__ == "__main__":
    main()
