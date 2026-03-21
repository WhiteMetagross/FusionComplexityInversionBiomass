#!/usr/bin/env python3
"""
Count parameters for all model variants used in the paper.

Uses torch parameter counting (sum of p.numel()) which does not require
a forward pass, avoiding CUDA dependency issues with Mamba kernels.
Also uses thop for backbone FLOPs where possible.

Author: Mridankan Mandal
Paper: arXiv:2603.07819.

Run: conda activate mambahar && python src/utils/count_params.py
"""

import sys
import os
import torch
import torch.nn as nn
import timm

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    GatedDepthwiseConvBlock,
    MambaFusionBlock,
    BidirMamba,
    BabyMambaFusionBlock,
    CVGABlock,
    BiomassModelTimm,
    _make_head,
)


def count_params(module, label=""):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    """Format parameter count with commas and M suffix."""
    return f"{n:>12,d}  ({n/1e6:.2f}M)"


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────
# 1. Standalone Fusion Block Parameters (dim=1024, matching DINOv3 nf)
# ──────────────────────────────────────────────────────────────────────
print_separator("FUSION BLOCK PARAMETERS (dim=1024)")

dim = 1024

blocks = {
    "GatedDepthwiseConvBlock": GatedDepthwiseConvBlock(dim, dropout=0.2),
    "MambaFusionBlock (FullMamba)": MambaFusionBlock(dim, d_state=16, d_conv=4, expand=2, dropout=0.2),
    "BidirMamba": BidirMamba(dim, d_state=16, d_conv=4, expand=2, filter_size=5, dropout=0.2),
    "BabyMambaFusionBlock": BabyMambaFusionBlock(dim, d_state=8, d_conv=4, expand=2, dropout=0.2),
    "CVGABlock (n_heads=8)": CVGABlock(dim, n_heads=8, dropout=0.2),
}

for name, block in blocks.items():
    total, trainable = count_params(block)
    print(f"\n  {name}:")
    print(f"    1 block:  {fmt(total)}")
    print(f"    2 blocks: {fmt(total * 2)}")
    
    # Break down sub-modules
    for sub_name, sub_module in block.named_children():
        sub_total = sum(p.numel() for p in sub_module.parameters())
        if sub_total > 0:
            print(f"      .{sub_name:12s}: {fmt(sub_total)}")


# ──────────────────────────────────────────────────────────────────────
# 2. Head Parameters (nf=1024)
# ──────────────────────────────────────────────────────────────────────
print_separator("HEAD PARAMETERS (nf=1024)")

nf = 1024
head = _make_head(nf, 0.2)
head_total, _ = count_params(head)
print(f"  Single head:           {fmt(head_total)}")
print(f"  3 heads (G/D/C):       {fmt(head_total * 3)}")

# Break down head
for i, layer in enumerate(head):
    layer_total = sum(p.numel() for p in layer.parameters())
    if layer_total > 0:
        print(f"    [{i}] {layer.__class__.__name__:20s}: {fmt(layer_total)}")


# ──────────────────────────────────────────────────────────────────────
# 3. Metadata MLP Parameters
# ──────────────────────────────────────────────────────────────────────
print_separator("METADATA MLP (23 → 64, fused with 1024)")

meta_input_dim = 23
meta_hidden = 64
meta_mlp = nn.Sequential(
    nn.Linear(meta_input_dim, meta_hidden),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(meta_hidden, meta_hidden),
)
meta_proj = nn.Sequential(
    nn.Linear(nf + meta_hidden, nf),
    nn.GELU(),
)
meta_total, _ = count_params(meta_mlp)
proj_total, _ = count_params(meta_proj)
print(f"  meta_mlp:    {fmt(meta_total)}")
print(f"  meta_proj:   {fmt(proj_total)}")
print(f"  Total meta:  {fmt(meta_total + proj_total)}")


# ──────────────────────────────────────────────────────────────────────
# 4. Pool layer (AdaptiveAvgPool1d) — no parameters
# ──────────────────────────────────────────────────────────────────────
pool = nn.AdaptiveAvgPool1d(1)
pool_total, _ = count_params(pool)
print(f"\n  AdaptiveAvgPool1d:     {fmt(pool_total)}  (expected 0)")


# ──────────────────────────────────────────────────────────────────────
# 5. Full Models with DINOv3 backbone
# ──────────────────────────────────────────────────────────────────────
print_separator("FULL MODELS — DINOv3-ViT-L backbone")

model_name = 'vit_large_patch16_dinov3.lvd1689m'

configs = [
    ("B5: GatedDWConv (no meta)",   dict(use_metadata=False)),
    ("B5m: GatedDWConv (with meta)", dict(use_metadata=True)),
    ("E1: BidirMamba",              dict(use_biobabymamba=True)),
    ("E2: CVGA",                    dict(use_cvga=True)),
    ("E4: Identity (no fusion)",    dict()),  # default is GatedDWConv, we'll handle manually
    ("E5: FullMamba",               dict(use_mamba_ssm=True)),
    ("BabyMamba",                   dict(use_babymamba=True)),
]

for label, kwargs in configs:
    print(f"\n  {label}:")
    
    if "Identity" in label:
        # E4 uses Identity fusion — build default GatedDWConv then replace
        model = BiomassModelTimm(model_name, dropout=0.2, pretrained=False, **kwargs)
        model.fusion = nn.Sequential(nn.Identity(), nn.Identity())
    else:
        model = BiomassModelTimm(model_name, dropout=0.2, pretrained=False, **kwargs)
    
    total, trainable = count_params(model)
    backbone_total, _ = count_params(model.backbone)
    fusion_total, _ = count_params(model.fusion)
    
    head_g, _ = count_params(model.head_green)
    head_d, _ = count_params(model.head_dead)
    head_c, _ = count_params(model.head_clover)
    heads_total = head_g + head_d + head_c
    
    meta_total_model = 0
    if hasattr(model, 'meta_mlp') and model.use_metadata:
        mm, _ = count_params(model.meta_mlp)
        mp, _ = count_params(model.meta_proj)
        meta_total_model = mm + mp
    
    task_specific = total - backbone_total
    
    print(f"    Backbone:       {fmt(backbone_total)}")
    print(f"    Fusion (2×):    {fmt(fusion_total)}")
    print(f"    3 Heads:        {fmt(heads_total)}")
    if meta_total_model > 0:
        print(f"    Meta MLP+Proj:  {fmt(meta_total_model)}")
    print(f"    Task-specific:  {fmt(task_specific)}  ({task_specific/total*100:.1f}%)")
    print(f"    TOTAL:          {fmt(total)}")


# ──────────────────────────────────────────────────────────────────────
# 6. EfficientNet-B3 backbone
# ──────────────────────────────────────────────────────────────────────
print_separator("EfficientNet-B3 BACKBONE")

eff_backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0, global_pool='')
eff_total, _ = count_params(eff_backbone)
eff_nf = eff_backbone.num_features
print(f"  Params:        {fmt(eff_total)}")
print(f"  num_features:  {eff_nf}")


# ──────────────────────────────────────────────────────────────────────
# 7. VMamba-Base backbone (if vmamba available)
# ──────────────────────────────────────────────────────────────────────
print_separator("VMamba-Base BACKBONE")

try:
    from models import VMambaBackbone
    vmamba = VMambaBackbone(pretrained_path='', variant='vmamba_base')
    vmamba_total, _ = count_params(vmamba)
    print(f"  Params:        {fmt(vmamba_total)}")
    print(f"  num_features:  {vmamba.num_features}")
except Exception as e:
    print(f"  [SKIP] VMamba not available: {e}")


# ──────────────────────────────────────────────────────────────────────
# 8. DINOv2-ViT-L backbone for comparison
# ──────────────────────────────────────────────────────────────────────
print_separator("DINOv2-ViT-L BACKBONE")

try:
    dinov2_backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, 
                                         num_classes=0, global_pool='')
    dinov2_total, _ = count_params(dinov2_backbone)
    dinov2_nf = dinov2_backbone.num_features
    print(f"  Params:        {fmt(dinov2_total)}")
    print(f"  num_features:  {dinov2_nf}")
except Exception as e:
    print(f"  [SKIP] DINOv2 not available: {e}")


# ──────────────────────────────────────────────────────────────────────
# 9. Summary Table
# ──────────────────────────────────────────────────────────────────────
print_separator("SUMMARY FOR PAPER")
print("""
Use these verified counts to update the paper.
Key numbers to check:
  - DINOv3-ViT-L backbone params (paper says 304M)
  - GatedDWConv per-block and 2-block total (paper says ~3.1M/block, ~6.3M)
  - CVGA per-block and 2-block total (paper says ~6.3M/block, ~12.6M)
  - BidirMamba per-block and 2-block total (paper says ~3.6M/block, ~7.2M)
  - FullMamba per-block and 2-block total (paper says ~8.4M/block, ~16.8M)
  - Task-specific params (paper says ~7.9M, 2.5%)
  - Total model params (paper says ~312M)
  - EfficientNet-B3 (paper says 12M)
  - VMamba-Base (paper says 89M)
""")
