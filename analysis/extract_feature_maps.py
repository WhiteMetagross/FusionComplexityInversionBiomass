# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
Backbone Feature Map Visualization: DINOv2-Large vs DINOv3-ViT-L vs VMamba-Base
================================================================================
Extracts feature norm maps from 3 backbones for 4 diverse images.
Uses feature norms (not CLS attention) for consistent, reliable activation maps.

Run in WSL:  source ~/miniconda3/etc/profile.d/conda.sh && conda activate mambahar
             python extract_feature_maps.py
"""

import os, sys, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# ── Paths (WSL mount) ───────────────────────────────────────────────────────
BASE     = "/mnt/c/Users/Xeron/Desktop/ProjectBioMass/csiro-biomass"
PROJ     = "/mnt/c/Users/Xeron/Desktop/ProjectBioMass"
TRAIN_DIR = os.path.join(BASE, "train")
WEIGHTS  = "/mnt/c/Users/Xeron/Desktop/ProjectBioMass/local_training_cv/pretrained"
OUT_PNG  = os.path.join(PROJ, "figures", "png")
OUT_SVG  = os.path.join(PROJ, "figures", "svg")
os.makedirs(OUT_PNG, exist_ok=True)
os.makedirs(OUT_SVG, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Transforms ───────────────────────────────────────────────────────────────
IMG_SIZE = 518

val_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_image(img_path):
    """Load image and return (raw_rgb, tensor)."""
    img = np.array(Image.open(img_path).convert("RGB"))
    raw = img.copy()
    augmented = val_tfm(image=img)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return raw, tensor


def normalize_map(fmap):
    """Normalize feature map to [0, 1] with percentile clipping for contrast."""
    vmin = np.percentile(fmap, 2)
    vmax = np.percentile(fmap, 98)
    if vmax - vmin < 1e-8:
        return np.zeros_like(fmap)
    return np.clip((fmap - vmin) / (vmax - vmin), 0, 1)


def resize_map(fmap, target_h, target_w):
    """Resize 2D feature map using scipy (reliable for float arrays)."""
    zoom_h = target_h / fmap.shape[0]
    zoom_w = target_w / fmap.shape[1]
    return ndimage_zoom(fmap, (zoom_h, zoom_w), order=1)


# ── Select 4 diverse images ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "train.csv"))
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
pivot = df.pivot_table(
    index=["image_path","Sampling_Date","State","Species","Pre_GSHH_NDVI","Height_Ave_cm"],
    columns="target_name", values="target"
).reset_index()

samples = []
for state, sort_col, ascending, label in [
    ("NSW", "Dry_Green_g", False, "High Green (NSW)"),
    ("Tas", "Dry_Dead_g",  False, "High Dead (Tas)"),
    ("Vic", "Dry_Total_g", False, "High Biomass (Vic)"),
    ("WA",  "Dry_Total_g", True,  "Low Biomass (WA)"),
]:
    sub = pivot[pivot["State"] == state].sort_values(sort_col, ascending=ascending)
    row = sub.iloc[len(sub)//10]
    samples.append((row, label))

print(f"Selected {len(samples)} images:")
for row, label in samples:
    print(f"  {label}: {os.path.basename(row['image_path'])} "
          f"Total={row['Dry_Total_g']:.1f}g NDVI={row['Pre_GSHH_NDVI']:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE 1: DINOv2-Large (ViT-L/14, patch14)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Loading DINOv2-Large ...")

dinov2 = timm.create_model(
    "vit_large_patch14_dinov2.lvd142m",
    pretrained=True, num_classes=0, global_pool=""
)
dinov2 = dinov2.eval().to(DEVICE)

@torch.no_grad()
def get_dinov2_features(tensor):
    """Extract feature norms from DINOv2-Large patch tokens."""
    B, C, H, W = tensor.shape
    x = dinov2.patch_embed(tensor)
    if x.dim() == 4:
        x = x.flatten(2).transpose(1, 2)
    cls_token = dinov2.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + dinov2.pos_embed
    x = dinov2.norm_pre(x)

    for blk in dinov2.blocks:
        x = blk(x)
    x = dinov2.norm(x)

    # Feature norms from patch tokens (skip CLS at index 0)
    patch_tokens = x[:, 1:, :]  # (B, N_patches, D)
    feat_norm = patch_tokens.norm(dim=-1)  # (B, N_patches)

    h = w = IMG_SIZE // 14  # 37
    feat_map = feat_norm.reshape(B, h, w).cpu().numpy()[0]
    return feat_map

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE 2: DINOv3-ViT-L (ViT-L/16, patch16)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/3] Loading DINOv3-ViT-L ...")

dinov3 = timm.create_model(
    "vit_large_patch16_dinov3",
    pretrained=True, num_classes=0, global_pool=""
)
dinov3 = dinov3.eval().to(DEVICE)

@torch.no_grad()
def get_dinov3_features(tensor):
    """Extract feature norms from DINOv3-ViT-L patch tokens."""
    t = F.interpolate(tensor, size=(512, 512), mode="bilinear", align_corners=False)
    B, C, H, W = t.shape

    x = dinov3.patch_embed(t)
    # patch_embed returns (B, H, W, C) for DINOv3
    if x.dim() == 4:
        if x.shape[-1] == dinov3.embed_dim:
            x = x.flatten(1, 2)  # (B, H*W, C)
        else:
            x = x.flatten(2).transpose(1, 2)

    has_cls = hasattr(dinov3, 'cls_token') and dinov3.cls_token is not None
    if has_cls:
        cls_token = dinov3.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

    if hasattr(dinov3, 'pos_embed') and dinov3.pos_embed is not None:
        if dinov3.pos_embed.shape[1] != x.shape[1]:
            pe = dinov3.pos_embed
            if has_cls:
                cls_pe, patch_pe = pe[:, :1, :], pe[:, 1:, :]
            else:
                cls_pe, patch_pe = None, pe
            side_orig = int(patch_pe.shape[1] ** 0.5)
            patch_pe = patch_pe.reshape(1, side_orig, side_orig, -1).permute(0, 3, 1, 2)
            side_new = 512 // 16
            patch_pe = F.interpolate(patch_pe, size=(side_new, side_new),
                                     mode="bilinear", align_corners=False)
            patch_pe = patch_pe.flatten(2).transpose(1, 2)
            pe = torch.cat([cls_pe, patch_pe], dim=1) if cls_pe is not None else patch_pe
            x = x + pe
        else:
            x = x + dinov3.pos_embed

    if hasattr(dinov3, 'norm_pre'):
        x = dinov3.norm_pre(x)

    for blk in dinov3.blocks:
        x = blk(x)

    if hasattr(dinov3, 'norm'):
        x = dinov3.norm(x)

    start_idx = 1 if has_cls else 0
    patch_tokens = x[:, start_idx:, :]
    feat_norm = patch_tokens.norm(dim=-1)

    h = w = 512 // 16  # 32
    n_patches = h * w
    feat_map = feat_norm[:, :n_patches].reshape(B, h, w).cpu().numpy()[0]
    return feat_map

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE 3: VMamba-Base (v2, depths=[2,2,15,2], dims=128)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/3] Loading VMamba-Base ...")
gc.collect()
torch.cuda.empty_cache()

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
def get_vmamba_features(tensor):
    """Extract feature norms from VMamba-Base last stage."""
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

    # feat shape: (B, C, H, W) for channel_first
    feat_norm = feat.norm(dim=1)  # (B, H, W)
    feat_map = feat_norm[0].numpy()
    return feat_map

# ══════════════════════════════════════════════════════════════════════════════
# Extract features for all samples
# ══════════════════════════════════════════════════════════════════════════════
print("\nExtracting features ...")

all_results = []
for (row, label) in samples:
    img_name = os.path.basename(row["image_path"])
    img_path = os.path.join(TRAIN_DIR, img_name)
    raw_img, tensor = load_image(img_path)

    print(f"  Processing {img_name} ({label}) ...")

    d2_feat = get_dinov2_features(tensor)
    torch.cuda.empty_cache()

    d3_feat = get_dinov3_features(tensor)
    torch.cuda.empty_cache()

    try:
        vm_feat = get_vmamba_features(tensor)
    except RuntimeError:
        with torch.cuda.amp.autocast():
            vm_feat = get_vmamba_features(tensor.half())
    torch.cuda.empty_cache()

    all_results.append({
        "raw": raw_img, "label": label, "row": row,
        "d2_feat": d2_feat, "d3_feat": d3_feat, "vm_feat": vm_feat,
    })

# Free GPU memory
del dinov2, dinov3, vmamba
gc.collect()
torch.cuda.empty_cache()

print("\nGenerating figures ...")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

TARGET_COLORS = {
    "Dry_Clover_g": "#3C5488", "Dry_Dead_g": "#B09C85",
    "Dry_Green_g": "#00A087", "Dry_Total_g": "#E64B35", "GDM_g": "#F39B7F",
}

def save(fig, name):
    fig.savefig(os.path.join(OUT_PNG, f"{name}.png"))
    fig.savefig(os.path.join(OUT_SVG, f"{name}.svg"))
    plt.close(fig)
    print(f"  [OK] {name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Backbone Feature Activation Comparison (merged, 4 rows x 5 cols)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 22))
gs = gridspec.GridSpec(4, 5, hspace=0.3, wspace=0.15,
                       width_ratios=[2.5, 1.5, 1.5, 1.5, 0.8])

col_titles = [
    "Original Image and Ground Truth",
    "DINOv2-Large\n(ViT-L/14, 1024-d)",
    "DINOv3-ViT-L\n(ViT-L/16, 1024-d)",
    "VMamba-Base\n(SSM, [2,2,15,2])",
    "Biomass Targets",
]

for row_idx, res in enumerate(all_results):
    raw = res["raw"]
    label = res["label"]
    r = res["row"]
    h_img, w_img = raw.shape[:2]

    # Col 0: Original image
    ax = fig.add_subplot(gs[row_idx, 0])
    ax.imshow(raw)
    ax.set_title(f"({chr(97+row_idx)}) {label}\n"
                 f"Total={r['Dry_Total_g']:.1f}g | GDM={r['GDM_g']:.1f}g | NDVI={r['Pre_GSHH_NDVI']:.2f}",
                 fontsize=9, fontweight="bold")
    ax.axis("off")
    if row_idx == 0:
        ax.text(0.5, 1.22, col_titles[0], transform=ax.transAxes,
                ha="center", fontsize=12, fontweight="bold", color="#3C5488")

    # Cols 1-3: Feature norm overlays
    backbone_maps = [
        (res["d2_feat"], col_titles[1]),
        (res["d3_feat"], col_titles[2]),
        (res["vm_feat"], col_titles[3]),
    ]
    for j, (fmap, title) in enumerate(backbone_maps):
        ax = fig.add_subplot(gs[row_idx, j+1])
        # Normalize and resize
        norm_map = normalize_map(fmap)
        resized = resize_map(norm_map, h_img, w_img)
        ax.imshow(raw)
        im = ax.imshow(resized, cmap="inferno", alpha=0.6, interpolation="bilinear",
                       vmin=0, vmax=1)
        ax.axis("off")
        if row_idx == 0:
            ax.text(0.5, 1.22, title, transform=ax.transAxes,
                    ha="center", fontsize=11, fontweight="bold", color="#3C5488")

    # Col 4: Biomass bar chart
    ax = fig.add_subplot(gs[row_idx, 4])
    tgt_vals = [r[t] for t in TARGETS]
    tgt_cols = [TARGET_COLORS[t] for t in TARGETS]
    tgt_short = ["Clover", "Dead", "Green", "Total", "GDM"]
    bars = ax.barh(range(5), tgt_vals, color=tgt_cols, edgecolor="white", height=0.55)
    ax.set_yticks(range(5))
    ax.set_yticklabels(tgt_short, fontsize=9)
    for bar, v in zip(bars, tgt_vals):
        if v > 0.1:
            ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f"{v:.1f}g",
                    va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Biomass (g)", fontsize=9)
    ax.invert_yaxis()
    if row_idx == 0:
        ax.text(0.5, 1.22, col_titles[4], transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="bold", color="#3C5488")

fig.suptitle("Backbone Feature Activation Comparison\n"
             "Feature Norms: DINOv2, DINOv3, and VMamba (ImageNet Pretrained)",
             fontweight="bold", fontsize=16, y=1.0)
save(fig, "fig11_backbone_feature_maps")

print(f"\nDone! Backbone feature figure saved to {OUT_PNG}")
