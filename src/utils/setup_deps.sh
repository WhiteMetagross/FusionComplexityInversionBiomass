#!/bin/bash
# ============================================================
# One-time setup: install VMamba + download all pretrained weights
#
# Usage (WSL):
#   conda activate mambahar
#   bash src/utils/setup_deps.sh
# ============================================================

set -e
PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CACHE_DIR="$PROJ_ROOT/pretrained"
mkdir -p "$CACHE_DIR"

echo "============================================================"
echo "  Setup: VMamba deps + pretrained weight cache"
echo "  Cache dir: $CACHE_DIR"
echo "============================================================"

# ── 1. Install VMamba package (single-file approach) ───────────
echo -e "\n>>> Installing VMamba module..."
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
TMPDIR=$(mktemp -d)
git clone --filter=blob:none --depth=1 https://github.com/MzeroMiko/VMamba.git "$TMPDIR/vmamba_repo" 2>&1
cp "$TMPDIR/vmamba_repo/vmamba.py" "$SITE_PKGS/vmamba.py"
rm -rf "$TMPDIR"
python -c "from vmamba import VSSM; print('[OK] VMamba VSSM import works')" || {
    echo "[WARN] VMamba install failed. Models will fall back to timm ViT backbone."
}

# ── 2. Ensure other deps ──────────────────────────────────────
echo -e "\n>>> Checking other Python deps..."
python -m pip install -q albumentations timm huggingface_hub safetensors tqdm scikit-learn

# ── 3. Download VMamba pretrained weights (GitHub releases) ────
echo -e "\n>>> Downloading VMamba pretrained weights..."
CACHE_DIR="${CACHE_DIR:-$PROJ_ROOT/pretrained}"
mkdir -p "$CACHE_DIR"

declare -A WEIGHTS
WEIGHTS[vssmtiny_dp01_ckpt_epoch_292.pth]="https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
WEIGHTS[vssmsmall_dp03_ckpt_epoch_238.pth]="https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmsmall_dp03_ckpt_epoch_238.pth"
WEIGHTS[vssmbase_dp06_ckpt_epoch_241.pth]="https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmbase_dp06_ckpt_epoch_241.pth"
WEIGHTS[vssm_base_0229_ckpt_epoch_237.pth]="https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"

for filename in "${!WEIGHTS[@]}"; do
    dest="$CACHE_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "  [OK] $filename already cached"
        continue
    fi
    url="${WEIGHTS[$filename]}"
    echo "  Downloading $filename..."
    wget -q --show-progress -O "$dest" "$url" || {
        echo "  [FAIL] $filename"
        rm -f "$dest"
    }
done

echo -e "\nVMamba weights in cache:"
ls -lh "$CACHE_DIR"/*.pth 2>/dev/null || echo "  (none)"

# ── 4. Pre-download timm models used by baselines ─────────────
echo -e "\n>>> Pre-caching timm models (first download only)..."
python - <<'PYEOF'
import timm, torch

models = [
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_base_patch14_dinov2.lvd142m",
    "convnext_base.fb_in22k_ft_in1k_384",
]

for name in models:
    print(f"  Caching {name}...", end=" ", flush=True)
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0)
        del m
        torch.cuda.empty_cache()
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")

print("\nAll timm models cached in ~/.cache/huggingface/hub/")
PYEOF

echo -e "\n============================================================"
echo "  Setup complete!"
echo "  VMamba weights: $CACHE_DIR/"
echo "  timm models:    ~/.cache/huggingface/hub/"
echo "============================================================"
