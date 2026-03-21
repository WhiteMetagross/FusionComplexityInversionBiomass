# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
Publication-quality architecture diagram for CVPR Agriculture-Vision Workshop.

Fig: Full pipeline (top) + 4 fusion block internals (bottom).
Clean, professional research-grade aesthetic using matplotlib.

Usage:
    python figures/generate_architecture_diagram.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent
PNG_DIR = OUT_DIR / 'png'
SVG_DIR = OUT_DIR / 'svg'
PNG_DIR.mkdir(exist_ok=True)
SVG_DIR.mkdir(exist_ok=True)

# ── Vibrant, high-saturation palette ──────────────────────────────────
C_INPUT     = '#42A5F5'   # bright blue — inputs
C_INPUT_E   = '#1565C0'
C_BACKBONE  = '#1E88E5'   # vivid blue — backbone
C_BB_E      = '#0D47A1'
C_CONCAT    = '#26A69A'   # vivid teal — concat/merge
C_CONCAT_E  = '#00695C'
C_POOL      = '#7E57C2'   # vivid purple — pooling/projection
C_POOL_E    = '#4527A0'
C_META      = '#FF9800'   # bold amber — metadata
C_META_E    = '#E65100'
C_HEAD      = '#66BB6A'   # vivid green — prediction heads
C_HEAD_E    = '#2E7D32'
C_OUT       = '#EF5350'   # vivid red — outputs
C_OUT_E     = '#C62828'
C_SUM       = '#FFA726'   # bright orange — sum nodes
C_SUM_E     = '#E65100'

# Fusion block colors — bold & distinct
C_GDW       = '#43A047'   # bold green — GatedDWConv (proposed)
C_GDW_E     = '#1B5E20'
C_GDW_BG    = '#E8F5E9'
C_CVGA      = '#8E24AA'   # bold purple — CVGA
C_CVGA_E    = '#4A148C'
C_CVGA_BG   = '#F3E5F5'
C_BIDIR     = '#E64A19'   # bold deep orange — BidirMamba
C_BIDIR_E   = '#BF360C'
C_BIDIR_BG  = '#FBE9E7'
C_FULL      = '#D32F2F'   # bold red — FullMamba
C_FULL_E    = '#B71C1C'
C_FULL_BG   = '#FFEBEE'

C_ARROW     = '#263238'   # dark charcoal arrows
C_SKIP      = '#78909C'   # blue-gray skip connections
C_DIMTEXT   = '#555555'
C_LIGHTGRID = '#E0E0E0'

# Styles — increased padding for larger text
BLOCK_KW = dict(boxstyle='round,pad=0.22', linewidth=1.8)
SMALL_KW = dict(boxstyle='round,pad=0.15', linewidth=1.5)
CIRCLE_KW = dict(boxstyle='circle,pad=0.12', linewidth=1.5)


def draw_box(ax, x, y, w, h, text, fc, ec, fontsize=11, fontweight='bold',
             text_color='white', zorder=5, subtext=None, kw=None):
    if kw is None:
        kw = BLOCK_KW
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         facecolor=fc, edgecolor=ec,
                         zorder=zorder, **kw)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.22, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=zorder+1)
        ax.text(x, y - 0.24, subtext, ha='center', va='center',
                fontsize=max(fontsize - 2.5, 7), fontweight='normal', color=text_color,
                alpha=0.88, zorder=zorder+1)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=zorder+1)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.6, style='-|>',
               zorder=4, connectionstyle='arc3,rad=0', mutation_scale=13):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw, zorder=zorder,
        mutation_scale=mutation_scale, connectionstyle=connectionstyle)
    ax.add_patch(arrow)


def draw_dim(ax, x, y, text, fontsize=8, color=C_DIMTEXT):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, style='italic', zorder=10,
            bbox=dict(boxstyle='round,pad=0.14', facecolor='white',
                      edgecolor='#BDBDBD', alpha=0.95, linewidth=0.6))


# =====================================================================
# TOP: Full Pipeline
# =====================================================================
def draw_pipeline(ax):
    y_t = 9.5    # top view (left image path)
    y_b = 7.5    # bottom view (right image path)
    y_m = 8.5    # merge line
    y_meta = 6.0

    bh = 1.1     # box height for main pipeline

    # Input images
    draw_box(ax, 1.5, y_t, 2.2, bh, 'Left Image', C_INPUT, C_INPUT_E,
             fontsize=11, subtext='512 × 512')
    draw_box(ax, 1.5, y_b, 2.2, bh, 'Right Image', C_INPUT, C_INPUT_E,
             fontsize=11, subtext='512 × 512')

    # Backbone (weight-tied)
    draw_box(ax, 5.5, y_t, 2.6, bh, 'DINOv3-ViT-L', C_BACKBONE, C_BB_E,
             fontsize=11, subtext='Shared Weights')
    draw_box(ax, 5.5, y_b, 2.6, bh, 'DINOv3-ViT-L', C_BACKBONE, C_BB_E,
             fontsize=11, subtext='Shared Weights')
    # Weight-tie indicator
    ax.plot([5.5, 5.5], [y_t - 0.55, y_b + 0.55],
            linestyle=':', color=C_BB_E, linewidth=1.2, alpha=0.5, zorder=3)

    # Arrows: input → backbone
    draw_arrow(ax, 2.6, y_t, 4.2, y_t)
    draw_arrow(ax, 2.6, y_b, 4.2, y_b)

    # Dim labels — placed well above/below the path to avoid overlap
    draw_dim(ax, 8.0, y_t + 0.75, '[B, N, 1024]')
    draw_dim(ax, 8.0, y_b - 0.75, '[B, N, 1024]')

    # Concat
    draw_box(ax, 10.0, y_m, 1.8, 0.90, 'Concat', C_CONCAT, C_CONCAT_E, fontsize=11)
    draw_arrow(ax, 6.8, y_t, 9.1, y_m + 0.18, connectionstyle='arc3,rad=-0.15')
    draw_arrow(ax, 6.8, y_b, 9.1, y_m - 0.18, connectionstyle='arc3,rad=0.15')
    draw_dim(ax, 10.0, y_m + 0.75, '[B, 2N, 1024]')

    # Fusion region (dashed outline)
    fusion_rect = FancyBboxPatch(
        (11.7, y_m - 0.72), 5.6, 1.44,
        boxstyle='round,pad=0.14', facecolor='#F7F7F7', edgecolor='#AAAAAA',
        linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.add_patch(fusion_rect)
    ax.text(14.5, y_m + 1.0, 'Interchangeable Fusion', ha='center', va='center',
            fontsize=9, style='italic', color='#777777', zorder=3)

    draw_box(ax, 13.2, y_m, 1.8, 0.90, 'Fusion ×1', '#708090', '#4A5A6A',
             fontsize=10, subtext='See (a)–(d)')
    draw_box(ax, 15.8, y_m, 1.8, 0.90, 'Fusion ×2', '#708090', '#4A5A6A',
             fontsize=10, subtext='See (a)–(d)')
    draw_arrow(ax, 10.9, y_m, 12.3, y_m)
    draw_arrow(ax, 14.1, y_m, 14.9, y_m)

    # Avg Pool
    draw_box(ax, 18.5, y_m, 1.6, 0.90, 'Avg Pool', C_POOL, C_POOL_E,
             fontsize=11, subtext='Global')
    draw_arrow(ax, 16.7, y_m, 17.7, y_m)
    draw_dim(ax, 18.5, y_m + 0.75, '[B, 1024]')

    # Metadata branch
    draw_box(ax, 18.5, y_meta, 1.8, 0.80, 'Metadata', C_META, C_META_E,
             fontsize=10, subtext='23-dim')
    draw_box(ax, 21.0, y_meta, 1.6, 0.80, 'Meta MLP', C_META, C_META_E,
             fontsize=10, subtext='→ 64-d')
    draw_arrow(ax, 19.4, y_meta, 20.2, y_meta)

    # Concat node (image + meta)
    draw_box(ax, 21.2, y_m, 0.75, 0.60, '⊕', C_CONCAT, C_CONCAT_E,
             fontsize=14, fontweight='normal', kw=SMALL_KW)
    draw_arrow(ax, 19.3, y_m, 20.82, y_m)
    draw_arrow(ax, 21.0, y_meta + 0.40, 21.15, y_m - 0.30,
               connectionstyle='arc3,rad=0.20')
    ax.text(19.7, y_meta + 0.65, '(optional)', ha='center', va='center',
            fontsize=7.5, color='#999999', style='italic', zorder=10)

    # Projection
    draw_box(ax, 23.0, y_m, 1.5, 0.80, 'Project', C_POOL, C_POOL_E,
             fontsize=11, subtext='+ GELU')
    draw_arrow(ax, 21.57, y_m, 22.25, y_m)

    # Prediction Heads
    hx = 25.5
    hs = 1.5
    for dy, name in [(hs, 'Head_Green'), (0, 'Head_Dead'), (-hs, 'Head_Clover')]:
        draw_box(ax, hx, y_m + dy, 1.5, 0.60, name, C_HEAD, C_HEAD_E,
                 fontsize=9, subtext='Softplus')

    draw_arrow(ax, 23.75, y_m, 24.75, y_m + hs, connectionstyle='arc3,rad=-0.15')
    draw_arrow(ax, 23.75, y_m, 24.75, y_m)
    draw_arrow(ax, 23.75, y_m, 24.75, y_m - hs, connectionstyle='arc3,rad=0.15')

    # Outputs
    ox = 27.8
    for dy, name in [(hs, 'Green'), (0, 'Dead'), (-hs, 'Clover')]:
        draw_box(ax, ox, y_m + dy, 1.1, 0.50, name, C_OUT, C_OUT_E,
                 fontsize=9, kw=SMALL_KW)
        draw_arrow(ax, 26.25, y_m + dy, ox - 0.55, y_m + dy)

    # Sum: Green + Clover → GDM
    s1x = 29.5
    draw_box(ax, s1x, y_m + 0.55, 0.50, 0.45, '+', C_SUM, C_SUM_E,
             fontsize=13, fontweight='bold', kw=CIRCLE_KW)
    draw_arrow(ax, ox + 0.55, y_m + hs, s1x - 0.25, y_m + 0.75,
               connectionstyle='arc3,rad=-0.10', mutation_scale=9)
    draw_arrow(ax, ox + 0.55, y_m - hs, s1x - 0.25, y_m + 0.35,
               connectionstyle='arc3,rad=0.28', mutation_scale=9)

    # GDM
    gx = 30.5
    draw_box(ax, gx, y_m + 0.55, 0.90, 0.45, 'GDM', C_SUM, C_SUM_E,
             fontsize=9, kw=SMALL_KW)
    draw_arrow(ax, s1x + 0.25, y_m + 0.55, gx - 0.45, y_m + 0.55, mutation_scale=9)

    # Sum: GDM + Dead → Total
    s2x = 31.5
    draw_box(ax, s2x, y_m, 0.50, 0.45, '+', C_SUM, C_SUM_E,
             fontsize=13, fontweight='bold', kw=CIRCLE_KW)
    draw_arrow(ax, gx + 0.45, y_m + 0.55, s2x - 0.18, y_m + 0.18,
               connectionstyle='arc3,rad=-0.10', mutation_scale=9)
    draw_arrow(ax, ox + 0.55, y_m, s2x - 0.25, y_m, mutation_scale=9)

    # Total
    tx = 32.6
    draw_box(ax, tx, y_m, 0.95, 0.45, 'Total', C_SUM, C_SUM_E,
             fontsize=9, kw=SMALL_KW)
    draw_arrow(ax, s2x + 0.25, y_m, tx - 0.48, y_m, mutation_scale=9)

    # Compositional label
    ax.text(30.5, y_m - 1.0, 'Compositional Constraints',
            fontsize=8, ha='center', va='top', color='#999999', style='italic')


# =====================================================================
# FUSION BLOCK PANELS
# =====================================================================
def draw_fusion_panel(ax, cx, cy, title, subtitle, accent, accent_e, accent_bg,
                      panel_w=6.5, panel_h=6.0, layout='linear', steps=None,
                      is_proposed=False):
    """Draw a single fusion block internal diagrams."""
    # Panel background
    bg = FancyBboxPatch(
        (cx - panel_w/2, cy - panel_h/2), panel_w, panel_h,
        boxstyle='round,pad=0.20', facecolor=accent_bg, edgecolor=accent_e,
        linewidth=1.6, zorder=2, alpha=0.6)
    ax.add_patch(bg)

    # Proposed highlight border
    if is_proposed:
        hl = FancyBboxPatch(
            (cx - panel_w/2 - 0.10, cy - panel_h/2 - 0.10),
            panel_w + 0.20, panel_h + 0.20,
            boxstyle='round,pad=0.25', facecolor='none', edgecolor='#FFD600',
            linewidth=2.5, linestyle='-', zorder=1)
        ax.add_patch(hl)
        ax.text(cx + panel_w/2 - 0.4, cy + panel_h/2 + 0.10, '★ Proposed',
                fontsize=9, color='#B8860B', fontweight='bold',
                ha='right', va='bottom', zorder=15)

    # Title
    ty = cy + panel_h/2 - 0.12
    ax.text(cx, ty, title, ha='center', va='top',
            fontsize=11, fontweight='bold', color=accent_e, zorder=10)
    ax.text(cx, ty - 0.42, subtitle, ha='center', va='top',
            fontsize=8, color='#666666', style='italic', zorder=10)

    # I/O labels
    ax.text(cx, cy + panel_h/2 + 0.18, 'x ∈ ℝ^{B×S×D}', ha='center', va='bottom',
            fontsize=8, color=C_DIMTEXT, style='italic')
    ax.text(cx, cy - panel_h/2 - 0.18, 'x + residual', ha='center', va='top',
            fontsize=8, color=C_DIMTEXT, style='italic')

    content_top = ty - 0.90
    content_bot = cy - panel_h/2 + 0.45

    if layout == 'linear':
        _draw_linear(ax, cx, content_top, content_bot, steps, accent, accent_e, panel_w)
    elif layout == 'cvga':
        _draw_cvga(ax, cx, content_top, content_bot, accent, accent_e, panel_w)
    elif layout == 'bidir':
        _draw_bidir(ax, cx, content_top, content_bot, accent, accent_e, panel_w)
    elif layout == 'fullmamba':
        _draw_fullmamba(ax, cx, content_top, content_bot, accent, accent_e, panel_w)


def _draw_skip(ax, cx, y_top, y_bot, pw, accent_e):
    """Draw a skip connection on the left side."""
    sx = cx - pw * 0.42
    ax.annotate('', xy=(cx - pw * 0.37, y_bot),
                xytext=(cx - pw * 0.37, y_top),
                arrowprops=dict(arrowstyle='-|>', color=C_SKIP,
                                connectionstyle='arc3,rad=0.40',
                                lw=1.2, linestyle='--'))
    ax.text(sx - 0.15, (y_top + y_bot)/2, 'skip', fontsize=7, color=C_SKIP,
            rotation=90, ha='center', va='center', style='italic')


def _draw_linear(ax, cx, top, bot, steps, accent, accent_e, pw):
    """Linear vertical chain with skip connection."""
    n = len(steps)
    bw, bh = pw * 0.68, 0.55
    total_h = top - bot
    spacing = (total_h - bh) / (n - 1) if n > 1 else 0
    ys = []
    for i, (label, sub) in enumerate(steps):
        sy = top - bh/2 - spacing * i
        ys.append(sy)
        draw_box(ax, cx, sy, bw, bh, label, accent, accent_e,
                 fontsize=9, text_color='white', subtext=sub, zorder=6)
        if i > 0:
            draw_arrow(ax, cx, ys[i-1] - bh/2, cx, sy + bh/2,
                       color=accent_e, lw=1.1, mutation_scale=9)
    if n >= 2:
        _draw_skip(ax, cx, ys[0], ys[-1], pw, accent_e)


def _draw_cvga(ax, cx, top, bot, accent, accent_e, pw):
    """CVGA with cross-attention split."""
    bh = 0.50
    bw = pw * 0.68
    off = pw * 0.21
    total_h = top - bot
    sp = (total_h - bh) / 4.0  # 5 rows, 4 gaps

    y1 = top - bh/2
    draw_box(ax, cx, y1, bw, bh, 'LayerNorm', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)

    y2 = y1 - sp
    draw_box(ax, cx, y2, bw, bh, 'Sigmoid Gate', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx, y1 - bh/2, cx, y2 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    y3 = y2 - sp
    sbw = bw * 0.44
    draw_box(ax, cx - off, y3, sbw, bh, 'Q,K,V_L', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    draw_box(ax, cx + off, y3, sbw, bh, 'Q,K,V_R', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    draw_arrow(ax, cx - 0.30, y2 - bh/2, cx - off, y3 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)
    draw_arrow(ax, cx + 0.30, y2 - bh/2, cx + off, y3 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)

    y4 = y3 - sp
    draw_box(ax, cx - off, y4, sbw, bh, 'Attn(L→R)', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    draw_box(ax, cx + off, y4, sbw, bh, 'Attn(R→L)', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    # Crossing arrows
    draw_arrow(ax, cx - off + sbw*0.30, y3 - bh/2,
               cx + off - sbw*0.30, y4 + bh/2,
               color='#B39DDB', lw=1.5, mutation_scale=8, connectionstyle='arc3,rad=-0.15')
    draw_arrow(ax, cx + off - sbw*0.30, y3 - bh/2,
               cx - off + sbw*0.30, y4 + bh/2,
               color='#B39DDB', lw=1.5, mutation_scale=8, connectionstyle='arc3,rad=0.15')

    y5 = y4 - sp
    draw_box(ax, cx, y5, bw, bh, 'Concat + Linear', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx - off, y4 - bh/2, cx - 0.30, y5 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)
    draw_arrow(ax, cx + off, y4 - bh/2, cx + 0.30, y5 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)

    _draw_skip(ax, cx, y1, y5, pw, accent_e)


def _draw_bidir(ax, cx, top, bot, accent, accent_e, pw):
    """Bidirectional Mamba with forward/backward SSM paths."""
    bh = 0.48
    bw = pw * 0.68
    off = pw * 0.21
    total_h = top - bot
    sp = (total_h - bh) / 5.0  # 6 rows, 5 gaps

    y1 = top - bh/2
    draw_box(ax, cx, y1, bw, bh, 'LayerNorm', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)

    y2 = y1 - sp
    draw_box(ax, cx, y2, bw, bh, 'Sigmoid Gate', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx, y1 - bh/2, cx, y2 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    y3 = y2 - sp
    draw_box(ax, cx, y3, bw, bh, 'DWConv (k=5)', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx, y2 - bh/2, cx, y3 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    y4 = y3 - sp
    sbw = bw * 0.44
    draw_box(ax, cx - off, y4, sbw, bh, 'Mamba →', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    draw_box(ax, cx + off, y4, sbw, bh, '← Mamba', accent, accent_e,
             fontsize=7.5, text_color='white', zorder=6)
    draw_arrow(ax, cx - 0.30, y3 - bh/2, cx - off, y4 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)
    draw_arrow(ax, cx + 0.30, y3 - bh/2, cx + off, y4 + bh/2,
               color=accent_e, lw=1.0, mutation_scale=8)
    # Weight-tied indicator
    ax.plot([cx - off, cx + off], [y4 + bh/2 + 0.08, y4 + bh/2 + 0.08],
            linestyle=':', color='#D4A76A', linewidth=1.4, alpha=0.7, zorder=7)
    ax.text(cx, y4 + bh/2 + 0.16, 'weight-tied', fontsize=7, ha='center', va='bottom',
            color='#A07030', fontweight='bold', style='italic', zorder=8)

    y5 = y4 - sp
    draw_box(ax, cx, y5, bw*0.30, bh * 0.85, '+', accent, accent_e,
             fontsize=13, fontweight='bold', text_color='white', zorder=6, kw=SMALL_KW)
    draw_arrow(ax, cx - off, y4 - bh/2, cx - 0.22, y5 + bh*0.42,
               color=accent_e, lw=1.0, mutation_scale=8)
    draw_arrow(ax, cx + off, y4 - bh/2, cx + 0.22, y5 + bh*0.42,
               color=accent_e, lw=1.0, mutation_scale=8)

    y6 = y5 - sp
    draw_box(ax, cx, y6, bw, bh, 'Linear', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx, y5 - bh*0.42, cx, y6 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    _draw_skip(ax, cx, y1, y6, pw, accent_e)


def _draw_fullmamba(ax, cx, top, bot, accent, accent_e, pw):
    """FullMamba — simple 3-component chain."""
    bh = 0.55
    bw = pw * 0.68
    total_h = top - bot
    sp = (total_h - bh) / 2.0  # 3 rows, 2 gaps

    y1 = top - bh/2
    draw_box(ax, cx, y1, bw, bh, 'LayerNorm', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)

    y2 = y1 - sp
    draw_box(ax, cx, y2, bw, bh, 'Mamba SSM', accent, accent_e,
             fontsize=9, text_color='white', subtext='expand=2, full', zorder=6)
    draw_arrow(ax, cx, y1 - bh/2, cx, y2 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    y3 = y2 - sp
    draw_box(ax, cx, y3, bw, bh, 'Dropout', accent, accent_e,
             fontsize=9, text_color='white', zorder=6)
    draw_arrow(ax, cx, y2 - bh/2, cx, y3 + bh/2, color=accent_e, lw=1.1, mutation_scale=9)

    _draw_skip(ax, cx, y1, y3, pw, accent_e)

    ax.text(cx, y3 - 0.50, '3 components · ~8d² params',
            ha='center', va='center', fontsize=7.5, color='#999999', style='italic')


# =====================================================================
# MAIN FIGURE
# =====================================================================
def make_architecture_figure():
    print('Generating architecture diagram...')

    fig, ax = plt.subplots(1, 1, figsize=(26, 14))
    ax.set_xlim(-1.0, 35.0)
    ax.set_ylim(-3.0, 12.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Pipeline section header ───────────────────────────────────────
    ax.text(17.0, 12.0, 'Pipeline Overview', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2C2C2C',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8EDF2',
                      edgecolor='#1565C0', linewidth=1.4))

    draw_pipeline(ax)

    # ── Separator ─────────────────────────────────────────────────────
    ax.axhline(y=4.5, color='#CCCCCC', linewidth=0.8, linestyle='-',
               xmin=0.02, xmax=0.98)
    ax.text(17.0, 4.85, 'Fusion Block Variants', ha='center', va='center',
            fontsize=15, fontweight='bold', color='#2C2C2C',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5EDE0',
                      edgecolor='#E65100', linewidth=1.4))

    # ── Bottom: 4 Fusion Block Panels ─────────────────────────────────
    pw, ph = 6.5, 6.2
    py = 1.0
    gap = 1.5
    total = 4 * pw + 3 * gap
    sx = (34.0 - total) / 2 + pw / 2

    # (a) GatedDepthwiseConv — Proposed
    draw_fusion_panel(ax, sx, py,
                      '(a) GatedDepthwiseConv',
                      'Local fusion · k=5 · R²=0.903',
                      C_GDW, C_GDW_E, C_GDW_BG,
                      pw, ph, layout='linear', is_proposed=True,
                      steps=[('LayerNorm', None), ('Sigmoid Gate', None),
                             ('DWConv (k=5)', 'groups=d'), ('Linear', None)])

    # (b) CVGA
    draw_fusion_panel(ax, sx + pw + gap, py,
                      '(b) CVGA',
                      'Cross-view attention · 8 heads · R²=0.833',
                      C_CVGA, C_CVGA_E, C_CVGA_BG,
                      pw, ph, layout='cvga')

    # (c) Bidirectional Mamba
    draw_fusion_panel(ax, sx + 2*(pw + gap), py,
                      '(c) Bidirectional Mamba',
                      'Bidir SSM + local conv · R²=0.819',
                      C_BIDIR, C_BIDIR_E, C_BIDIR_BG,
                      pw, ph, layout='bidir')

    # (d) FullMamba
    draw_fusion_panel(ax, sx + 3*(pw + gap), py,
                      '(d) FullMamba',
                      'Full SSM · expand=2 · R²=0.793',
                      C_FULL, C_FULL_E, C_FULL_BG,
                      pw, ph, layout='fullmamba')

    # ── Complexity arrow ──────────────────────────────────────────────
    ay = py - ph/2 - 0.75
    ax_left = sx - pw/2
    ax_right = sx + 3*(pw + gap) + pw/2
    ax.annotate('', xy=(ax_right, ay), xytext=(ax_left, ay),
                arrowprops=dict(arrowstyle='-|>', color='#777777', lw=1.5))
    ax.text((ax_left + ax_right)/2, ay - 0.30,
            'Increasing Model Complexity →', ha='center', va='top',
            fontsize=10, color='#777777', fontweight='bold', style='italic')
    ax.text(ax_left + 0.5, ay + 0.25, 'Best R²',
            ha='left', va='bottom', fontsize=8.5, color='#43A047', fontweight='bold')
    ax.text(ax_right - 0.5, ay + 0.25, 'Worst R²',
            ha='right', va='bottom', fontsize=8.5, color='#D32F2F', fontweight='bold')

    plt.tight_layout(pad=0.5)
    fig.savefig(PNG_DIR / 'fig_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.20)
    fig.savefig(SVG_DIR / 'fig_architecture.svg', bbox_inches='tight',
                facecolor='white', pad_inches=0.20)
    print('  Saved: fig_architecture.png, fig_architecture.svg')
    plt.close()


if __name__ == '__main__':
    make_architecture_figure()
    print('\nDone!')
