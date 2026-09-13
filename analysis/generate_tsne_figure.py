# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
t-SNE Feature Space Visualization: Four Backbone Architectures
===============================================================
Extracts global-pooled features from EfficientNet-B3, VMamba-Base,
DINOv2-ViT-L, and DINOv3-ViT-L for all 357 training images, runs
t-SNE, and produces a 2×2 panel colored by Dry Total quintile.

Run in WSL:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mambahar
    python analysis/generate_tsne_figure.py
"""

import os, sys, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.manifold import TSNE
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = str(REPO_ROOT / "csiro-biomass")
PROJ = str(REPO_ROOT)
TRAIN_DIR = os.path.join(BASE, "train")
WEIGHTS = str(REPO_ROOT / "pretrained")
# Output directories.
OUT_FIG_IMG = os.path.join(PROJ, "output", "analysis", "img")
OUT_PAPER_IMG = os.path.join(PROJ, "output", "analysis", "paper")
OUT_PNG = os.path.join(PROJ, "output", "analysis", "png")
OUT_SVG = os.path.join(PROJ, "output", "analysis", "svg")
for d in [OUT_FIG_IMG, OUT_PAPER_IMG, OUT_PNG, OUT_SVG]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Style (matching paper figures) ───────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Quintile color palette — diverging from low to high biomass
QUINTILE_COLORS = ['#3C5488', '#4DBBD5', '#00A087', '#F39B7F', '#E64B35']
QUINTILE_LABELS = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']

# ── Data ─────────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(os.path.join(BASE, "train.csv"))

# Pivot: one row per image with all targets as columns
pivot = df.pivot_table(
    index=["image_path", "Sampling_Date", "State", "Species",
           "Pre_GSHH_NDVI", "Height_Ave_cm"],
    columns="target_name", values="target"
).reset_index()

# Assign Dry Total quintile (1–5)
pivot["quintile"] = pd.qcut(pivot["Dry_Total_g"], q=5, labels=False)  # 0-indexed

image_paths = pivot["image_path"].tolist()
quintiles = pivot["quintile"].values
dry_total = pivot["Dry_Total_g"].values

N = len(image_paths)
print(f"  {N} images, quintile distribution: {np.bincount(quintiles)}")

# ── Transforms ───────────────────────────────────────────────────────────────
val_tfm = A.Compose([
    A.Resize(518, 518),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def load_tensor(img_rel_path):
    """Load image, apply val transforms, return tensor on device."""
    full_path = os.path.join(BASE, img_rel_path)
    img = np.array(Image.open(full_path).convert("RGB"))
    augmented = val_tfm(image=img)
    return augmented["image"].unsqueeze(0).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction functions (return pooled 1-D feature vector per image)
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(model_name, extract_fn, cleanup_fn=None):
    """Extract pooled features for all images, with numpy cache."""
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace(".", "_")
    cache_path = os.path.join("/tmp", f"tsne_cache_{safe_name}.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        features = np.load(cache_path)
        if cleanup_fn:
            cleanup_fn()
        return features
    features = []
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{N}]")
        tensor = load_tensor(img_path)
        feat = extract_fn(tensor)
        features.append(feat)
        del tensor
    if cleanup_fn:
        cleanup_fn()
    gc.collect()
    torch.cuda.empty_cache()
    result = np.stack(features)
    np.save(cache_path, result)
    print(f"  Cached features to {cache_path}")
    return result


# ── Backbone 1: EfficientNet-B3 ─────────────────────────────────────────────
print("\n[1/4] EfficientNet-B3 ...")
effnet = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
effnet = effnet.eval().to(DEVICE)

@torch.no_grad()
def get_effnet_pooled(tensor):
    """Global average pooled features from EfficientNet-B3."""
    feat = effnet(tensor)  # num_classes=0 → pooled features
    return feat.cpu().numpy().flatten()

effnet_features = extract_all_features(
    "EfficientNet-B3", get_effnet_pooled,
    cleanup_fn=lambda: (effnet.cpu(), gc.collect(), torch.cuda.empty_cache())
)
del effnet
print(f"  EfficientNet-B3 features: {effnet_features.shape}")

# ── Backbone 2: VMamba-Base ──────────────────────────────────────────────────
print("\n[2/4] VMamba-Base ...")

from vmamba import VSSM

_VSSM_V2_KWARGS = dict(
    ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
    ssm_conv=3, ssm_conv_bias=False, ssm_init="v0", forward_type="v05_noz",
    mlp_ratio=4.0, mlp_act_layer="gelu", patch_norm=True, norm_layer="ln2d",
    downsample_version="v3", patchembed_version="v2",
)
vmamba = VSSM(depths=[2, 2, 15, 2], dims=128, **_VSSM_V2_KWARGS)
ckpt_path = os.path.join(WEIGHTS, "vssm_base_0229_ckpt_epoch_237.pth")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if "model" in ckpt:
    ckpt = ckpt["model"]
ckpt = {k: v for k, v in ckpt.items() if not k.startswith("classifier.")}
vmamba.load_state_dict(ckpt, strict=False)
vmamba = vmamba.eval().to(DEVICE)

# Hook to capture last stage output
vmamba_feats = {}
def make_vmamba_hook(name):
    def hook(module, input, output):
        vmamba_feats[name] = output.detach().cpu()
    return hook
for i, layer in enumerate(vmamba.layers):
    layer.register_forward_hook(make_vmamba_hook(f"stage_{i}"))

@torch.no_grad()
def get_vmamba_pooled(tensor):
    """Global average pooled features from VMamba-Base last stage."""
    t = F.interpolate(tensor, size=(512, 512), mode="bilinear", align_corners=False)
    vmamba_feats.clear()
    x = vmamba.patch_embed(t)
    if vmamba.pos_embed is not None:
        pos_embed = vmamba.pos_embed.permute(0, 2, 3, 1) if not vmamba.channel_first else vmamba.pos_embed
        x = x + pos_embed
    for layer in vmamba.layers:
        x = layer(x)
    last_key = f"stage_{len(vmamba.layers)-1}"
    feat = vmamba_feats.get(last_key, list(vmamba_feats.values())[-1])
    # feat: (B, C, H, W) → global average pool → (C,)
    pooled = feat.mean(dim=[2, 3]).numpy().flatten()
    return pooled

vmamba_features = extract_all_features(
    "VMamba-Base", get_vmamba_pooled,
    cleanup_fn=lambda: (vmamba.cpu(), gc.collect(), torch.cuda.empty_cache())
)
del vmamba
print(f"  VMamba-Base features: {vmamba_features.shape}")

# ── Backbone 3: DINOv2-ViT-L ────────────────────────────────────────────────
print("\n[3/4] DINOv2-ViT-L ...")
dinov2 = timm.create_model(
    "vit_large_patch14_dinov2.lvd142m",
    pretrained=True, num_classes=0, global_pool="token"
)
dinov2 = dinov2.eval().to(DEVICE)

@torch.no_grad()
def get_dinov2_pooled(tensor):
    """CLS token features from DINOv2-ViT-L."""
    feat = dinov2(tensor)  # global_pool="token" → CLS token
    return feat.cpu().numpy().flatten()

dinov2_features = extract_all_features(
    "DINOv2-ViT-L", get_dinov2_pooled,
    cleanup_fn=lambda: (dinov2.cpu(), gc.collect(), torch.cuda.empty_cache())
)
del dinov2
print(f"  DINOv2-ViT-L features: {dinov2_features.shape}")

# ── Backbone 4: DINOv3-ViT-L ────────────────────────────────────────────────
print("\n[4/4] DINOv3-ViT-L ...")
dinov3 = timm.create_model(
    "vit_large_patch16_dinov3",
    pretrained=True, num_classes=0, global_pool="token"
)
dinov3 = dinov3.eval().to(DEVICE)

@torch.no_grad()
def get_dinov3_pooled(tensor):
    """CLS token features from DINOv3-ViT-L (resize to 512 for patch16)."""
    t = F.interpolate(tensor, size=(512, 512), mode="bilinear", align_corners=False)
    feat = dinov3(t)
    return feat.cpu().numpy().flatten()

dinov3_features = extract_all_features(
    "DINOv3-ViT-L", get_dinov3_pooled,
    cleanup_fn=lambda: (dinov3.cpu(), gc.collect(), torch.cuda.empty_cache())
)
del dinov3
print(f"  DINOv3-ViT-L features: {dinov3_features.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# t-SNE projection
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning t-SNE ...")
TSNE_PARAMS = dict(n_components=2, perplexity=30, random_state=42,
                   max_iter=1000, learning_rate="auto", init="pca")

all_backbones = [
    ("EfficientNet-B3\n(10.70M, ImageNet-1K)", effnet_features),
    ("VMamba-Base\n(88.56M, ImageNet-1K)", vmamba_features),
    ("DINOv2-ViT-L\n(304.37M, LVD-142M)", dinov2_features),
    ("DINOv3-ViT-L\n(303.08M, LVD-1.7B)", dinov3_features),
]

embeddings = []
for name, feats in all_backbones:
    print(f"  t-SNE for {name.split(chr(10))[0]} ({feats.shape}) ...")
    tsne = TSNE(**TSNE_PARAMS)
    emb = tsne.fit_transform(feats.astype(np.float64))
    embeddings.append(emb)

# ══════════════════════════════════════════════════════════════════════════════
# Plot: 2×2 panel
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figure ...")

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for ax, (name, _), emb in zip(axes, all_backbones, embeddings):
    for q in range(5):
        mask = quintiles == q
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=QUINTILE_COLORS[q],
            s=18, alpha=0.7, edgecolors='none',
            label=QUINTILE_LABELS[q], rasterized=True,
        )
    ax.set_title(name, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

# Shared legend
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                  markersize=8, label=l)
           for c, l in zip(QUINTILE_COLORS, QUINTILE_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("t-SNE Projections of Pooled Backbone Features\nColored by Dry Total Biomass Quintile",
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ── Save outputs ─────────────────────────────────────────────────────────────
fname = "fig08_feature_space_analysis"

# High-quality version for paper/img
paper_path = os.path.join(OUT_PAPER_IMG, f"{fname}.png")
fig.savefig(paper_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f"  Saved: {paper_path}")

# Compressed version for figures/img
fig_img_path = os.path.join(OUT_FIG_IMG, f"{fname}.png")
fig.savefig(fig_img_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
print(f"  Saved: {fig_img_path}")

# Also save to figures/png and figures/svg
png_path = os.path.join(OUT_PNG, f"{fname}.png")
fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f"  Saved: {png_path}")

svg_path = os.path.join(OUT_SVG, f"{fname}.svg")
fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.05)
print(f"  Saved: {svg_path}")

plt.close(fig)
print("\nDone!")
