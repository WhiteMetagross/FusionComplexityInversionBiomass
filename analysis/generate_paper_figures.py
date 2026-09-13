# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
Publication-quality figures for CVPR Agriculture-Vision Workshop paper.

"Less is More: Foundation Model Adaptation for Pasture Biomass Estimation"

Generates 3 multi-panel figures:
  fig_main_results.png/svg     — (a) All models bar chart, (b) Backbone scale scatter
  fig_ablation_studies.png/svg — (a) Fusion type bars, (b) Depth curve, (c) Metadata interaction
  fig_fold_analysis.png/svg    — (a) Per-fold grouped bars, (b) Violin + strip plots

Usage:
    python analysis/generate_paper_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get(
    'PAPER_FIGURE_OUTPUT_DIR', REPO_ROOT / 'output' / 'analysis'
))
PNG_DIR = OUT_DIR / 'png'
SVG_DIR = OUT_DIR / 'svg'
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# ── Color palette ─────────────────────────────────────────────────────
C_PROPOSED  = '#1565C0'   # strong blue — proposed method
C_LOCAL     = '#43A047'   # green — local fusion family
C_ATTN      = '#8E24AA'   # purple — cross-attention
C_SSM_BIDIR = '#E64A19'   # deep orange — BidirMamba
C_SSM_FULL  = '#C62828'   # dark red — FullMamba
C_NONE      = '#78909C'   # blue-gray — identity / no fusion
C_META      = '#F9A825'   # amber — metadata ceiling marker
C_VMAMBA    = '#6D4C41'   # brown — VMamba backbone
C_EFFNET    = '#546E7A'   # steel — EfficientNet
C_DINOV2    = '#00838F'   # teal — DINOv2
C_DINOV3    = '#1565C0'   # blue — DINOv3
C_MEDIAN    = '#BDBDBD'   # light gray — median baseline

# ── Data ──────────────────────────────────────────────────────────────

# === FIGURE 1: Main Results — Proposed + All Baselines ===
main_models = {
    'B1: Median Predictor': {
        'r2': -0.065, 'std': 0.006, 'color': C_MEDIAN,
        'folds': [-0.0638, -0.0736, -0.0699, -0.0620, -0.0567],
    },
    'B2: EfficientNet-B3': {
        'r2': 0.555, 'std': 0.084, 'color': C_EFFNET,
        'folds': None,
    },
    'B4: DINOv2-L\n+ GatedDWConv': {
        'r2': 0.853, 'std': 0.097, 'color': C_DINOV2,
        'folds': [0.9628, 0.9619, 0.8513, 0.7499, 0.7403],
    },
    'B6: VMamba\n+ Mamba + Meta': {
        'r2': 0.743, 'std': 0.048, 'color': C_VMAMBA,
        'folds': [0.7465, 0.7593, 0.8154, 0.7285, 0.6663],
    },
    'B7: DINOv3-L\n+ CVGA + Meta': {
        'r2': 0.830, 'std': 0.050, 'color': C_ATTN,
        'folds': [0.8335, 0.8652, 0.8927, 0.8077, 0.7486],
    },
    'B8: DINOv3-L\n+ BidirMamba\n+ Meta': {
        'r2': 0.829, 'std': 0.042, 'color': C_SSM_BIDIR,
        'folds': [0.8321, 0.8417, 0.8920, 0.8175, 0.7611],
    },
    'Proposed: DINOv3-L\n+ 2× GatedDWConv': {
        'r2': 0.903, 'std': 0.064, 'color': C_PROPOSED,
        'folds': [0.9221, 0.9502, 0.9178, 0.9464, 0.7785],
    },
}

# === FIGURE 2a: Fusion Type Comparison (DINOv3, no metadata) ===
fusion_ablation = {
    'Identity\n(E4)': {
        'r2': 0.819, 'std': 0.055, 'color': C_NONE,
        'folds': [0.8400, 0.8437, 0.8926, 0.7848, 0.7324],
    },
    '1× GatedDWConv\n(E6)': {
        'r2': 0.821, 'std': 0.034, 'color': C_LOCAL,
        'folds': [0.8257, 0.8511, 0.8573, 0.8113, 0.7612],
    },
    '2× GatedDWConv\n(Proposed)': {
        'r2': 0.903, 'std': 0.064, 'color': C_PROPOSED,
        'folds': [0.9221, 0.9502, 0.9178, 0.9464, 0.7785],
    },
    '4× GatedDWConv\n(E7)': {
        'r2': 0.814, 'std': 0.039, 'color': '#66BB6A',
        'folds': [0.8230, 0.8559, 0.8465, 0.7971, 0.7489],
    },
    '2× CVGA\n(E2)': {
        'r2': 0.833, 'std': 0.051, 'color': C_ATTN,
        'folds': [0.8419, 0.8486, 0.9098, 0.8126, 0.7522],
    },
    '2× BidirMamba\n(E1)': {
        'r2': 0.819, 'std': 0.051, 'color': C_SSM_BIDIR,
        'folds': [0.8162, 0.8599, 0.8887, 0.7845, 0.7458],
    },
    '2× FullMamba\n(E5)': {
        'r2': 0.793, 'std': 0.034, 'color': C_SSM_FULL,
        'folds': [0.8088, 0.8445, 0.7763, 0.7913, 0.7418],
    },
}

# === Backbone scale data ===
backbone_data = {
    'EfficientNet-B3': {'r2': 0.555, 'std': 0.084, 'pretrain_size': 1.2e6,
                        'params': 12, 'color': C_EFFNET},
    'VMamba-Base':      {'r2': 0.717, 'std': 0.052, 'pretrain_size': 1.2e6,
                        'params': 89, 'color': C_VMAMBA},
    'DINOv2-ViT-L':    {'r2': 0.853, 'std': 0.097, 'pretrain_size': 142e6,
                        'params': 304, 'color': C_DINOV2},
    'DINOv3-ViT-L':    {'r2': 0.903, 'std': 0.064, 'pretrain_size': 169e6,
                        'params': 304, 'color': C_DINOV3},
}

# === 4×2 Factorial: Fusion × Metadata (with E8 = Identity + Meta) ===
factorial = {
    '2× GatedDWConv': {'no_meta': 0.903, 'with_meta': 0.829,
                       'no_meta_std': 0.064, 'with_meta_std': 0.045},
    '2× CVGA':        {'no_meta': 0.833, 'with_meta': 0.830,
                       'no_meta_std': 0.051, 'with_meta_std': 0.050},
    '2× BidirMamba':  {'no_meta': 0.819, 'with_meta': 0.829,
                       'no_meta_std': 0.051, 'with_meta_std': 0.042},
    'Identity':       {'no_meta': 0.819, 'with_meta': 0.828,
                       'no_meta_std': 0.055, 'with_meta_std': 0.053},
}

# === Fusion depth curve (0, 1, 2, 4 GatedDWConv blocks) ===
depth_curve = {
    0: {'r2': 0.819, 'std': 0.055},
    1: {'r2': 0.821, 'std': 0.034},
    2: {'r2': 0.903, 'std': 0.064},
    4: {'r2': 0.814, 'std': 0.039},
}

# === Fold data for analysis figure ===
fold_models = {
    'Proposed\n(2× GatedDWConv)': [0.9221, 0.9502, 0.9178, 0.9464, 0.7785],
    '2× CVGA\n(E2)':             [0.8419, 0.8486, 0.9098, 0.8126, 0.7522],
    '2× BidirMamba\n(E1)':       [0.8162, 0.8599, 0.8887, 0.7845, 0.7458],
    'Identity\n(E4)':            [0.8400, 0.8437, 0.8926, 0.7848, 0.7324],
    '4× GatedDWConv\n(E7)':      [0.8230, 0.8559, 0.8465, 0.7971, 0.7489],
    'Identity+Meta\n(E8)':       [0.8360, 0.8606, 0.9021, 0.7928, 0.7496],
}
FOLD_COLORS = [C_PROPOSED, C_ATTN, C_SSM_BIDIR, C_NONE, '#66BB6A', C_META]


def save_fig(fig, name):
    fig.savefig(PNG_DIR / f'{name}.png')
    fig.savefig(SVG_DIR / f'{name}.svg')
    print(f'  Saved: {name}.png, {name}.svg')


# =====================================================================
# FIGURE 1: Main Results (2 panels)
#   (a) All models horizontal bar chart  (b) Backbone pretraining scale
# =====================================================================
def make_fig_main_results():
    print('Figure 1: Main results...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Panel (a): Horizontal bar chart ───────────────────────────────
    names = list(main_models.keys())
    r2s   = [main_models[n]['r2'] for n in names]
    stds  = [main_models[n]['std'] for n in names]
    colors = [main_models[n]['color'] for n in names]

    # Sort ascending (best at top)
    order  = np.argsort(r2s)
    names  = [names[i] for i in order]
    r2s    = [r2s[i] for i in order]
    stds   = [stds[i] for i in order]
    colors = [colors[i] for i in order]

    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, r2s, color=colors, edgecolor='white',
                    linewidth=1.0, height=0.62, zorder=3)
    ax1.errorbar(r2s, y_pos, xerr=stds, fmt='none',
                 ecolor='#444444', elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)

    # Value labels
    for i, (bar, r2, std) in enumerate(zip(bars, r2s, stds)):
        is_proposed = 'Proposed' in names[i]
        x_pos = max(r2 + std + 0.015, r2 + 0.02)
        if r2 < 0:
            x_pos = 0.02
        ax1.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f'{r2:.3f}', ha='left', va='center', fontsize=8,
                 fontweight='bold' if is_proposed else 'normal',
                 color=C_PROPOSED if is_proposed else '#333333')

    # Highlight proposed bar
    proposed_idx = [i for i, n in enumerate(names) if 'Proposed' in n][0]
    bars[proposed_idx].set_edgecolor(C_PROPOSED)
    bars[proposed_idx].set_linewidth(2.0)
    ax1.text(r2s[proposed_idx] / 2,
             proposed_idx, '★', fontsize=12, color='#FFD600',
             ha='center', va='center', zorder=10)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=7.5)
    ax1.set_xlabel('Mean Weighted R²', fontweight='bold')
    ax1.set_xlim(-0.12, 1.08)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.set_title('(a) Overall Model Comparison', fontweight='bold', pad=10)
    ax1.grid(axis='x', zorder=0)
    ax1.axvline(x=0, color='#BDBDBD', linewidth=0.8, zorder=1)

    # ── Panel (b): Backbone Pretraining Scale ─────────────────────────
    for name, d in backbone_data.items():
        ax2.scatter(d['pretrain_size'], d['r2'],
                    s=d['params'] * 1.8 + 40,
                    c=d['color'], edgecolors='white', linewidth=1.5,
                    zorder=5, alpha=0.95)
        ax2.errorbar(d['pretrain_size'], d['r2'], yerr=d['std'],
                     fmt='none', ecolor=d['color'], elinewidth=1.0,
                     capsize=3, capthick=1.0, alpha=0.5, zorder=4)

    offsets = {
        'EfficientNet-B3': (55, 0),
        'VMamba-Base':      (55, 0),
        'DINOv2-ViT-L':    (-55, 0),
        'DINOv3-ViT-L':    (-55, 0),
    }
    for name, d in backbone_data.items():
        ox, oy = offsets[name]
        label = f"{name}\n({d['params']}M params)"
        ax2.annotate(label, (d['pretrain_size'], d['r2']),
                     textcoords='offset points', xytext=(ox, oy),
                     fontsize=7.5, ha='center', va='center',
                     arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#CCCCCC', alpha=0.9))

    # Log-linear trend
    x_vals = np.array([d['pretrain_size'] for d in backbone_data.values()])
    y_vals = np.array([d['r2'] for d in backbone_data.values()])
    z = np.polyfit(np.log10(x_vals), y_vals, 1)
    x_line = np.logspace(np.log10(0.5e6), np.log10(300e6), 100)
    ax2.plot(x_line, z[0] * np.log10(x_line) + z[1],
             '--', color='#999999', linewidth=1.0, alpha=0.5, zorder=1)

    ax2.set_xscale('log')
    ax2.set_xlabel('Pretraining Dataset Size (images)', fontweight='bold')
    ax2.set_ylabel('Mean Weighted R²', fontweight='bold')
    ax2.set_title('(b) Backbone Pretraining Scale', fontweight='bold', pad=10)
    ax2.set_ylim(0.45, 0.98)
    ax2.set_xlim(0.5e6, 400e6)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:.0f}M' if x >= 1e6 else f'{x / 1e3:.0f}K'))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.grid(True, zorder=0)

    for ps, lab in [(12, '12M'), (89, '89M'), (304, '304M')]:
        ax2.scatter([], [], s=ps * 1.8 + 40, c='gray', alpha=0.4,
                    edgecolors='white', label=lab)
    ax2.legend(title='Params', loc='lower right', framealpha=0.9,
               edgecolor='#CCCCCC', fontsize=7.5, title_fontsize=8)

    plt.tight_layout(w_pad=3)
    save_fig(fig, 'fig_main_results')
    plt.close()


# =====================================================================
# FIGURE 2: Ablation Studies (3 panels)
#   (a) Fusion type comparison   (b) Depth curve   (c) Metadata interaction
# =====================================================================
def make_fig_ablation_studies():
    print('Figure 2: Ablation studies...')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5),
                                         gridspec_kw={'width_ratios': [1.2, 0.8, 1.1]})

    # ── Panel (a): Fusion Type Comparison ─────────────────────────────
    names  = list(fusion_ablation.keys())
    r2s    = [fusion_ablation[n]['r2'] for n in names]
    stds   = [fusion_ablation[n]['std'] for n in names]
    colors = [fusion_ablation[n]['color'] for n in names]

    order   = np.argsort(r2s)[::-1]
    names_s  = [names[i] for i in order]
    r2s_s    = [r2s[i] for i in order]
    stds_s   = [stds[i] for i in order]
    colors_s = [colors[i] for i in order]

    bars = ax1.bar(range(len(names_s)), r2s_s, color=colors_s,
                   edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
    ax1.errorbar(range(len(names_s)), r2s_s, yerr=stds_s, fmt='none',
                 ecolor='#333333', elinewidth=1.0, capsize=4, capthick=1.0, zorder=4)

    # Highlight proposed
    for i, (bar, name) in enumerate(zip(bars, names_s)):
        if 'Proposed' in name:
            bar.set_edgecolor(C_PROPOSED)
            bar.set_linewidth(2.0)

    for i, (bar, r2, std) in enumerate(zip(bars, r2s_s, stds_s)):
        ax1.text(bar.get_x() + bar.get_width() / 2, r2 + std + 0.007,
                 f'{r2:.3f}', ha='center', va='bottom', fontsize=7.5,
                 fontweight='bold' if i == 0 else 'normal', color='#222222')

    # No-fusion baseline
    ax1.axhline(y=0.819, color=C_NONE, linestyle='--', linewidth=0.8,
                alpha=0.5, zorder=2)
    ax1.text(6.35, 0.850, 'No fusion\nbaseline', fontsize=6.5,
             color=C_NONE, ha='right', va='bottom', style='italic')

    ax1.set_xticks(range(len(names_s)))
    ax1.set_xticklabels(names_s, fontsize=5.25, rotation=30, ha='right')
    ax1.set_ylabel('Mean Weighted R²', fontweight='bold')
    ax1.set_ylim(0.74, 0.99)
    ax1.set_title('(a) Fusion Block Comparison\n(DINOv3-ViT-L, no metadata)',
                  fontweight='bold', pad=8)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax1.grid(axis='y', zorder=0)

    legend_patches = [
        mpatches.Patch(facecolor=C_PROPOSED, label='Local (Proposed)'),
        mpatches.Patch(facecolor=C_LOCAL, label='Local'),
        mpatches.Patch(facecolor='#66BB6A', label='Local (4×)'),
        mpatches.Patch(facecolor=C_ATTN, label='Cross-Attention'),
        mpatches.Patch(facecolor=C_SSM_BIDIR, label='Bidir. SSM'),
        mpatches.Patch(facecolor=C_SSM_FULL, label='Full SSM'),
        mpatches.Patch(facecolor=C_NONE, label='None (Identity)'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right', framealpha=0.9,
               edgecolor='#CCCCCC', fontsize=6.5, ncol=2)

    # ── Panel (b): Fusion Depth Curve ─────────────────────────────────
    blocks    = sorted(depth_curve.keys())
    r2_depth  = [depth_curve[b]['r2'] for b in blocks]
    std_depth = [depth_curve[b]['std'] for b in blocks]

    ax2.plot(blocks, r2_depth, 'o-', color=C_PROPOSED, linewidth=2.0,
             markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax2.fill_between(blocks,
                     [r - s for r, s in zip(r2_depth, std_depth)],
                     [r + s for r, s in zip(r2_depth, std_depth)],
                     color=C_PROPOSED, alpha=0.12, zorder=2)
    ax2.errorbar(blocks, r2_depth, yerr=std_depth, fmt='none',
                 ecolor=C_PROPOSED, elinewidth=1.0, capsize=4,
                 capthick=1.0, alpha=0.5, zorder=4)

    for b, r2, std in zip(blocks, r2_depth, std_depth):
        offset_y = 14 if r2 > 0.85 else -18
        ax2.annotate(f'{r2:.3f}', (b, r2),
                     textcoords='offset points', xytext=(0, offset_y),
                     fontsize=8.5, ha='center',
                     fontweight='bold' if b == 2 else 'normal',
                     color=C_PROPOSED if b == 2 else '#444444')

    ax2.annotate('★ Optimal', (2, 0.903),
                 textcoords='offset points', xytext=(22, 8),
                 fontsize=7.5, color='#FFD600', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#FFD600', lw=1.2))

    ax2.set_xticks(blocks)
    ax2.set_xticklabels(['0\n(Identity)', '1', '2', '4'], fontsize=8)
    ax2.set_xlabel('Number of GatedDWConv Blocks', fontweight='bold')
    ax2.set_ylabel('Mean Weighted R²', fontweight='bold')
    ax2.set_title('(b) Fusion Depth Curve', fontweight='bold', pad=8)
    ax2.set_ylim(0.76, 0.98)
    ax2.set_xlim(-0.3, 4.5)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax2.grid(True, zorder=0)

    # ── Panel (c): 4×2 Metadata × Fusion Interaction ─────────────────
    fusions = list(factorial.keys())
    x = np.arange(len(fusions))
    w = 0.32

    no_meta      = [factorial[f]['no_meta'] for f in fusions]
    with_meta    = [factorial[f]['with_meta'] for f in fusions]
    no_meta_std  = [factorial[f]['no_meta_std'] for f in fusions]
    with_meta_std = [factorial[f]['with_meta_std'] for f in fusions]

    bar_colors = [C_PROPOSED, C_ATTN, C_SSM_BIDIR, C_NONE]

    bars1 = ax3.bar(x - w / 2, no_meta, w, color=bar_colors,
                    edgecolor='white', linewidth=0.8, zorder=3)
    bars2 = ax3.bar(x + w / 2, with_meta, w, color=bar_colors,
                    edgecolor='white', linewidth=0.8, zorder=3,
                    alpha=0.40, hatch='///')

    ax3.errorbar(x - w / 2, no_meta, yerr=no_meta_std, fmt='none',
                 ecolor='#333333', elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    ax3.errorbar(x + w / 2, with_meta, yerr=with_meta_std, fmt='none',
                 ecolor='#333333', elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)

    for bars_set, vals in [(bars1, no_meta), (bars2, with_meta)]:
        for bar, val in zip(bars_set, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.007,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=6.5,
                     fontweight='bold' if val > 0.89 else 'normal')

    # ~0.829 ceiling line
    ax3.axhline(y=0.829, color=C_META, linestyle='--', linewidth=1.2,
                alpha=0.7, zorder=2)
    ax3.text(3.48, 0.843, '~0.829\nceiling', fontsize=6.5, color=C_META,
             ha='right', va='bottom', style='italic', fontweight='bold')

    # Delta arrow for GatedDWConv
    ax3.annotate('', xy=(-w / 2, 0.829), xytext=(-w / 2, 0.903),
                 arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5))
    ax3.text(-w / 2 - 0.22, 0.866, '−7.4%', fontsize=7.5, color='#C62828',
             fontweight='bold', ha='center', rotation=90)

    short_labels = ['GatedDWConv', 'CVGA', 'BidirMamba', 'Identity']
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, fontsize=8)
    ax3.set_ylabel('Mean Weighted R²', fontweight='bold')
    ax3.set_ylim(0.78, 0.97)
    ax3.set_title('(c) Metadata × Fusion Interaction', fontweight='bold', pad=8)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax3.grid(axis='y', zorder=0)

    solid_patch = mpatches.Patch(facecolor='#888888', label='No Metadata')
    hatch_patch = mpatches.Patch(facecolor='#888888', alpha=0.40, hatch='///',
                                 label='With Metadata')
    ax3.legend(handles=[solid_patch, hatch_patch], loc='upper right',
               framealpha=0.9, edgecolor='#CCCCCC', fontsize=7.5)

    plt.tight_layout(w_pad=2.5)
    save_fig(fig, 'fig_ablation_studies')
    plt.close()


# =====================================================================
# FIGURE 3: Fold Analysis (2 panels)
#   (a) Per-fold grouped bars  (b) Violin + strip plots
# =====================================================================
def make_fig_fold_analysis():
    print('Figure 3: Fold analysis...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    model_names = list(fold_models.keys())
    n_models = len(model_names)
    n_folds  = 5
    x = np.arange(n_folds)
    total_width = 0.78
    bar_w = total_width / n_models

    for i, (name, folds) in enumerate(fold_models.items()):
        offset = (i - n_models / 2 + 0.5) * bar_w
        ax1.bar(x + offset, folds, bar_w * 0.92, color=FOLD_COLORS[i],
                edgecolor='white', linewidth=0.5, label=name, zorder=3)

    # Highlight hardest fold
    ax1.axvspan(3.5, 4.5, alpha=0.07, color='#C62828', zorder=0)
    ax1.text(4, 0.84, 'Hardest\nFold', fontsize=7, ha='center', va='center',
             color='#C62828', fontweight='bold', style='italic')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i}' for i in range(5)], fontsize=9)
    ax1.set_ylabel('Weighted R²', fontweight='bold')
    ax1.set_ylim(0.70, 0.98)
    ax1.set_title('(a) Per-Fold Performance', fontweight='bold', pad=10)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax1.grid(axis='y', zorder=0)
    # L-shaped legend: first 4 items on top row (ncol=4), remaining 2 below
    handles, labels = ax1.get_legend_handles_labels()
    from matplotlib.legend import Legend
    # Use ncol=4 for the top-heavy L-shape so top row has 4 and bottom has 2
    ax1.legend(loc='upper right', fontsize=5.5, ncol=4, framealpha=0.9,
               edgecolor='#CCCCCC', columnspacing=0.6, handlelength=1.2,
               handleheight=0.7, labelspacing=0.3)

    # ── Panel (b): Violin + Strip Plots ───────────────────────────────
    model_order = sorted(fold_models.keys(),
                         key=lambda n: np.mean(fold_models[n]), reverse=True)
    color_map = dict(zip(fold_models.keys(), FOLD_COLORS))

    parts = ax2.violinplot(
        [fold_models[m] for m in model_order],
        positions=range(len(model_order)),
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, mn in zip(parts['bodies'], model_order):
        pc.set_facecolor(color_map[mn])
        pc.set_edgecolor('white')
        pc.set_alpha(0.30)
        pc.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for i, mn in enumerate(model_order):
        folds = fold_models[mn]
        jitter = rng.uniform(-0.08, 0.08, len(folds))
        ax2.scatter([i + j for j in jitter], folds,
                    s=36, c=color_map[mn], edgecolors='white',
                    linewidth=0.8, zorder=5, alpha=0.9)
        ax2.scatter(i, np.mean(folds), s=70, c=color_map[mn],
                    edgecolors='#333333', linewidth=1.2, zorder=6, marker='D')

    short_labels = [m.split('\n')[0] for m in model_order]
    ax2.set_xticks(range(len(model_order)))
    ax2.set_xticklabels(short_labels, fontsize=7, rotation=25, ha='right')
    ax2.set_ylabel('Weighted R²', fontweight='bold')
    ax2.set_ylim(0.70, 0.98)
    ax2.set_title('(b) Fold R² Distribution', fontweight='bold', pad=10)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax2.grid(axis='y', zorder=0)

    ax2.scatter([], [], s=36, c='gray', edgecolors='white', label='Individual fold')
    ax2.scatter([], [], s=70, c='gray', edgecolors='#333333', marker='D', label='Mean')
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout(w_pad=3)
    save_fig(fig, 'fig_fold_analysis')
    plt.close()


# =====================================================================
if __name__ == '__main__':
    print('Generating CVPR Agriculture-Vision figures...\n')
    make_fig_main_results()
    make_fig_ablation_studies()
    make_fig_fold_analysis()
    print(f'\nDone! All figures saved to {PNG_DIR} and {SVG_DIR}')
