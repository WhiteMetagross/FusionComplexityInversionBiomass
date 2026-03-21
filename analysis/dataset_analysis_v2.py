# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
CSIRO BioMass Dataset — Research-Grade Exploratory Data Analysis (v2)
=====================================================================
Informed by the Image2Biomass paper (Liao et al., arXiv:2510.22916).
Produces publication-quality figures for the research paper.

Key paper insights incorporated:
  - 70cm × 30cm quadrat framing, images from diverse consumer cameras
  - GDM = Dry_Green + Dry_Clover  (green dry matter)
  - Evaluation uses log(1+y) transform with weighted R²
  - NDVI from GreenSeeker handheld sensor (not satellite), saturates at high LAI
  - Height from falling plate meter (30cm diameter, 200g)
  - 19 locations across 4 Australian states, 2014-2017
  - Rigorous QC: 1162/3187 samples passed quality control

All plots saved as:
  - figures/png/<name>.png  (300 dpi)
  - figures/svg/<name>.svg
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)                          # csiro-biomass/
PROJECT = os.path.dirname(BASE)                           # ProjectBioMass/
FIGDIR_PNG = os.path.join(PROJECT, "figures", "png")
FIGDIR_SVG = os.path.join(PROJECT, "figures", "svg")
TRAIN_DIR  = os.path.join(BASE, "train")
os.makedirs(FIGDIR_PNG, exist_ok=True)
os.makedirs(FIGDIR_SVG, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ── Research-grade palettes ──────────────────────────────────────────────────
# Nature-style vibrant palette  (CB-safe)
STATE_COLORS = {"NSW": "#E64B35", "Tas": "#4DBBD5", "Vic": "#00A087", "WA": "#F39B7F"}
TARGET_COLORS = {
    "Dry_Clover_g": "#3C5488",
    "Dry_Dead_g":   "#B09C85",
    "Dry_Green_g":  "#00A087",
    "Dry_Total_g":  "#E64B35",
    "GDM_g":        "#F39B7F",
}
TARGET_LABELS = {
    "Dry_Clover_g": "Dry Clover (g)",
    "Dry_Dead_g":   "Dry Dead (g)",
    "Dry_Green_g":  "Dry Green (g)",
    "Dry_Total_g":  "Dry Total (g)",
    "GDM_g":        "GDM (g)",
}
COMP_COLORS = {"Dry_Green_g": "#00A087", "Dry_Dead_g": "#B09C85", "Dry_Clover_g": "#3C5488"}

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR_PNG, f"{name}.png"))
    fig.savefig(os.path.join(FIGDIR_SVG, f"{name}.svg"))
    plt.close(fig)
    print(f"  [OK] {name}")

# ── Load & prepare ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "train.csv"))
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
df["Month"] = df["Sampling_Date"].dt.month
df["MonthName"] = df["Sampling_Date"].dt.strftime("%b")

TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
pivot = df.pivot_table(
    index=["image_path","Sampling_Date","State","Species",
           "Pre_GSHH_NDVI","Height_Ave_cm","Month","MonthName"],
    columns="target_name", values="target"
).reset_index()

# Derived features (paper-informed)
pivot["Green_Fraction"]  = pivot["Dry_Green_g"]  / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["Dead_Fraction"]   = pivot["Dry_Dead_g"]   / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["Clover_Fraction"] = pivot["Dry_Clover_g"] / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["log_DryTotal"]    = np.log1p(pivot["Dry_Total_g"])
pivot["log_GDM"]         = np.log1p(pivot["GDM_g"])

# Image color features (precompute)
print("Precomputing image color statistics …")
color_records = []
for _, row in pivot.iterrows():
    img_name = os.path.basename(row["image_path"])
    img_path = os.path.join(TRAIN_DIR, img_name)
    if os.path.exists(img_path):
        img = np.array(Image.open(img_path))
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        brightness = img.mean()
        greenness = g.mean() / (r.mean() + g.mean() + b.mean() + 1e-8)
        # Excess Green Index (ExG) — vegetation detection
        R, G, B = r.mean()/255, g.mean()/255, b.mean()/255
        exg = 2*G - R - B
        color_records.append({
            "image_path": row["image_path"],
            "R_mean": r.mean(), "G_mean": g.mean(), "B_mean": b.mean(),
            "Brightness": brightness, "Greenness": greenness,
            "ExG": exg,
            "R_std": r.std(), "G_std": g.std(), "B_std": b.std(),
        })
color_df = pd.DataFrame(color_records)
pivot = pivot.merge(color_df, on="image_path", how="left")

month_order = sorted(pivot["Month"].unique())
month_labels = [pd.Timestamp(2015, m, 1).strftime("%b") for m in month_order]

N_IMAGES = len(pivot)
N_ROWS   = len(df)
print(f"Dataset: {N_IMAGES} images, {N_ROWS} rows\n")
print("Generating figures …\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Dataset Overview: Geographic & Species Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.4], wspace=0.35)

# 1a: State distribution (donut chart)
ax1 = fig.add_subplot(gs[0])
state_counts = pivot["State"].value_counts().reindex(["Tas","Vic","NSW","WA"])
wedges, texts, autotexts = ax1.pie(
    state_counts.values, labels=state_counts.index,
    colors=[STATE_COLORS[s] for s in state_counts.index],
    autopct=lambda p: f"{int(p*sum(state_counts)/100)}\n({p:.0f}%)",
    startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax1.set_title("(a) Samples by State", fontweight="bold")

# 1b: Temporal heatmap (State × Month)
ax2 = fig.add_subplot(gs[1])
temporal = pivot.groupby(["State","Month"]).size().unstack(fill_value=0)
temporal = temporal.reindex(index=["NSW","Tas","Vic","WA"], columns=month_order, fill_value=0)
temporal.columns = month_labels
sns.heatmap(temporal, annot=True, fmt="d", cmap="YlOrRd", linewidths=1.5,
            linecolor="white", ax=ax2, cbar_kws={"shrink": 0.7, "label": "Count"},
            annot_kws={"fontweight": "bold", "fontsize": 10})
ax2.set_title("(b) Sampling Calendar: State × Month", fontweight="bold")
ax2.set_xlabel("Sampling Month (2015)")
ax2.set_ylabel("")

# 1c: Species distribution (horizontal bar)
ax3 = fig.add_subplot(gs[2])
species_counts = pivot["Species"].value_counts().head(10)
species_palette = sns.color_palette("Set2", n_colors=10)
bars = ax3.barh(range(len(species_counts)), species_counts.values,
                color=species_palette, edgecolor="white", linewidth=1.0, height=0.7)
ax3.set_yticks(range(len(species_counts)))
ax3.set_yticklabels([s.replace("_", " ") for s in species_counts.index], fontsize=9)
for bar, v in zip(bars, species_counts.values):
    ax3.text(v + 1, bar.get_y() + bar.get_height()/2, str(v),
             ha="left", va="center", fontsize=9, fontweight="bold")
ax3.set_xlabel("Number of Samples")
ax3.set_title("(c) Top-10 Pasture Species", fontweight="bold")
ax3.invert_yaxis()
ax3.set_xlim(0, species_counts.max() * 1.18)

save(fig, "fig01_dataset_overview")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Target Variable Distributions with Log Transform
# Paper uses log(1+y) for evaluation — show both raw and transformed
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for i, tgt in enumerate(TARGETS):
    vals = pivot[tgt].dropna()
    log_vals = np.log1p(vals)
    
    # Row 1: Raw distributions
    axes[0,i].hist(vals, bins=30, color=TARGET_COLORS[tgt], alpha=0.75,
                   edgecolor="white", linewidth=0.5, density=True)
    sns.kdeplot(vals, ax=axes[0,i], color="black", linewidth=1.8)
    axes[0,i].axvline(vals.median(), color="#E64B35", linestyle="--", lw=1.2, alpha=0.9)
    axes[0,i].set_title(TARGET_LABELS[tgt], fontweight="bold", fontsize=11)
    skew = vals.skew()
    axes[0,i].text(0.95, 0.90, f"n={len(vals)}\nμ={vals.mean():.1f}\nσ={vals.std():.1f}\nskew={skew:.2f}",
                   transform=axes[0,i].transAxes, ha="right", va="top", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    if i == 0:
        axes[0,i].set_ylabel("Density (raw)")
    axes[0,i].set_xlabel("Biomass (g)")
    
    # Row 2: Log-transformed (as used in evaluation)
    axes[1,i].hist(log_vals, bins=30, color=TARGET_COLORS[tgt], alpha=0.75,
                   edgecolor="white", linewidth=0.5, density=True)
    sns.kdeplot(log_vals, ax=axes[1,i], color="black", linewidth=1.8)
    axes[1,i].axvline(log_vals.median(), color="#E64B35", linestyle="--", lw=1.2, alpha=0.9)
    log_skew = log_vals.skew()
    axes[1,i].text(0.95, 0.90, f"skew={log_skew:.2f}\nkurt={log_vals.kurtosis():.2f}",
                   transform=axes[1,i].transAxes, ha="right", va="top", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    if i == 0:
        axes[1,i].set_ylabel("Density (log-transformed)")
    axes[1,i].set_xlabel("log(1 + Biomass)")

fig.suptitle("Target Variable Distributions: Raw (top) vs. Log-Transformed (bottom)\n"
             "Paper evaluation uses log(1+y) transform — note skewness reduction",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig02_target_distributions_raw_vs_log")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Correlation Heatmap: Full Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════
corr_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "Greenness", "ExG", "Brightness"] + TARGETS
corr_labels = ["NDVI", "Height", "Img Green%", "ExG Index", "Brightness",
               "Dry Clover", "Dry Dead", "Dry Green", "Dry Total", "GDM"]
corr_mat = pivot[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8.5))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 15, s=90, l=45, as_cmap=True)
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=corr_labels, yticklabels=corr_labels,
            linewidths=1.5, linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            annot_kws={"size": 9.5, "fontweight": "bold"}, ax=ax)
ax.set_title("Feature Correlation Matrix\n(Metadata + Image-Derived + Biomass Targets)",
             fontweight="bold", fontsize=14, pad=15)
plt.xticks(rotation=40, ha="right")
fig.tight_layout()
save(fig, "fig03_correlation_heatmap_full")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — NDVI Saturation Analysis
# Paper notes NDVI saturates at high LAI (>3). Show this with biomass scatter.
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
targets_3 = [("GDM_g", "GDM (g)"), ("Dry_Total_g", "Dry Total (g)"), ("Dry_Green_g", "Dry Green (g)")]
for i, (tgt, lbl) in enumerate(targets_3):
    for state in ["NSW","Tas","Vic","WA"]:
        m = pivot["State"] == state
        axes[i].scatter(pivot.loc[m, "Pre_GSHH_NDVI"], pivot.loc[m, tgt],
                        c=STATE_COLORS[state], label=state, alpha=0.5, s=28,
                        edgecolors="white", linewidth=0.3)
    # Fit quadratic to show saturation
    x, y = pivot["Pre_GSHH_NDVI"].values, pivot[tgt].values
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_fit = np.linspace(x.min(), x.max(), 100)
    axes[i].plot(x_fit, p(x_fit), "k-", linewidth=2, alpha=0.6, label="Quadratic fit")
    # Pearson r
    r, pval = stats.pearsonr(x, y)
    # Spearman (better for non-linear)
    rho, _ = stats.spearmanr(x, y)
    axes[i].text(0.05, 0.95,
                 f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
                 transform=axes[i].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    axes[i].set_xlabel("GreenSeeker NDVI")
    axes[i].set_ylabel(lbl)
    axes[i].set_title(f"NDVI vs. {lbl}", fontweight="bold")
    if i == 0:
        axes[i].legend(title="State", frameon=True, markerscale=1.5, fontsize=8)
    # Shade saturation zone
    axes[i].axvspan(0.78, 0.92, alpha=0.06, color="red")
    axes[i].text(0.85, axes[i].get_ylim()[1]*0.15, "Saturation\nzone",
                 ha="center", fontsize=8, color="#E64B35", alpha=0.7, fontstyle="italic")

fig.suptitle("NDVI Saturation Analysis: GreenSeeker NDVI vs. Biomass Components\n"
             "NDVI saturates at high biomass (LAI > 3) — quadratic fit captures non-linearity",
             fontweight="bold", fontsize=13, y=1.04)
fig.tight_layout()
save(fig, "fig04_ndvi_saturation_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Biomass Composition Ternary-Style Analysis
# Show the three-component composition (Green, Dead, Clover) per state
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 5a: Stacked bar — mean composition by state
ax = axes[0]
components = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
comp_labels_short = ["Green", "Dead", "Clover"]
state_order = ["NSW", "Tas", "Vic", "WA"]
means = pivot.groupby("State")[components].mean().reindex(state_order)
stds  = pivot.groupby("State")[components].std().reindex(state_order)

bottom = np.zeros(4)
for comp, lbl, color in zip(components, comp_labels_short, [COMP_COLORS[c] for c in components]):
    vals = means[comp].values
    ax.bar(state_order, vals, bottom=bottom, label=lbl, color=color,
           edgecolor="white", linewidth=1.2, width=0.55)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 1.5:
            ax.text(j, b + v/2, f"{v:.1f}g", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottom += vals
# Add total on top
for j, state in enumerate(state_order):
    total = means.loc[state].sum()
    ax.text(j, total + 1, f"Σ={total:.1f}g", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Mean Biomass (g)")
ax.set_xlabel("Australian State")
ax.set_title("(a) Mean Biomass Composition by State", fontweight="bold")
ax.legend(frameon=True, loc="upper left")

# 5b: Green fraction vs Dead fraction scatter (composition space)
ax = axes[1]
valid = pivot.dropna(subset=["Green_Fraction", "Dead_Fraction", "Clover_Fraction"])
sc = ax.scatter(valid["Green_Fraction"], valid["Dead_Fraction"],
                c=valid["Dry_Total_g"], cmap="YlOrRd", s=35, alpha=0.7,
                edgecolors="white", linewidth=0.3)
cb = fig.colorbar(sc, ax=ax, shrink=0.85, label="Dry Total (g)")
ax.plot([0, 1], [1, 0], "k--", alpha=0.25, lw=1)
ax.set_xlabel("Green Fraction (Dry Green / Dry Total)")
ax.set_ylabel("Dead Fraction (Dry Dead / Dry Total)")
ax.set_title("(b) Composition Space (colored by total biomass)", fontweight="bold")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.text(0.75, 0.85, "Healthy", fontsize=10, color="#00A087", fontweight="bold",
        transform=ax.transAxes, ha="center")
ax.text(0.15, 0.85, "Senescent", fontsize=10, color="#B09C85", fontweight="bold",
        transform=ax.transAxes, ha="center")

fig.suptitle("Biomass Composition Analysis: Three-Component Breakdown",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig05_biomass_composition")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Height vs Biomass with NDVI Color
# Paper: Height from falling plate meter, NDVI from GreenSeeker
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 6a: Height vs Dry Total colored by NDVI
ax = axes[0]
sc = ax.scatter(pivot["Height_Ave_cm"], pivot["Dry_Total_g"],
                c=pivot["Pre_GSHH_NDVI"], cmap="RdYlGn", s=30, alpha=0.65,
                edgecolors="white", linewidth=0.3)
fig.colorbar(sc, ax=ax, shrink=0.85, label="NDVI")
r, _ = stats.pearsonr(pivot["Height_Ave_cm"], pivot["Dry_Total_g"])
ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
ax.set_xlabel("Falling Plate Height (cm)")
ax.set_ylabel("Dry Total (g)")
ax.set_title("(a) Height vs. Total Biomass\n(colored by NDVI)", fontweight="bold")

# 6b: Height vs GDM colored by NDVI  
ax = axes[1]
sc = ax.scatter(pivot["Height_Ave_cm"], pivot["GDM_g"],
                c=pivot["Pre_GSHH_NDVI"], cmap="RdYlGn", s=30, alpha=0.65,
                edgecolors="white", linewidth=0.3)
fig.colorbar(sc, ax=ax, shrink=0.85, label="NDVI")
r, _ = stats.pearsonr(pivot["Height_Ave_cm"], pivot["GDM_g"])
ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
ax.set_xlabel("Falling Plate Height (cm)")
ax.set_ylabel("GDM (g)")
ax.set_title("(b) Height vs. Green Dry Matter\n(colored by NDVI)", fontweight="bold")

fig.suptitle("Canopy Height–Biomass Relationship with Vegetation Vigor (NDVI)",
             fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig06_height_biomass_ndvi")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Target Distributions by State (Violin + Swarm)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
for i, tgt in enumerate(TARGETS):
    sns.violinplot(data=pivot, x="State", y=tgt, order=["NSW","Tas","Vic","WA"],
                   palette=STATE_COLORS, ax=axes[i], inner="quartile",
                   linewidth=1.2, cut=0, width=0.8)
    sns.stripplot(data=pivot, x="State", y=tgt, order=["NSW","Tas","Vic","WA"],
                  palette=STATE_COLORS, ax=axes[i], alpha=0.25, size=2.5, jitter=0.2)
    axes[i].set_title(TARGET_LABELS[tgt], fontweight="bold", fontsize=11)
    axes[i].set_xlabel("")
    if i > 0:
        axes[i].set_ylabel("")
    else:
        axes[i].set_ylabel("Biomass (g)")

fig.suptitle("Biomass Target Distributions by State (Violin + Individual Points)",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig07_violin_by_state")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Seasonal Dynamics:  Monthly trends with composition breakdown
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 8a: Total biomass trend
ax = axes[0]
for state in ["NSW","Tas","Vic","WA"]:
    sub = pivot[pivot["State"] == state]
    grp = sub.groupby("Month")["Dry_Total_g"].agg(["mean","std","count"])
    grp = grp.reindex(month_order)
    valid = grp["mean"].notna()
    ax.plot(grp.index[valid], grp["mean"][valid], "o-", color=STATE_COLORS[state],
            label=state, linewidth=2.2, markersize=6)
    ax.fill_between(grp.index[valid],
                    (grp["mean"] - grp["std"])[valid],
                    (grp["mean"] + grp["std"])[valid],
                    color=STATE_COLORS[state], alpha=0.1)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("Dry Total (g)")
ax.set_title("(a) Total Biomass Trend", fontweight="bold")
ax.legend(title="State", frameon=True, fontsize=9)

# 8b: Green/Dead ratio trend (overall)
ax = axes[1]
monthly = pivot.groupby("Month")[["Dry_Green_g","Dry_Dead_g","Dry_Clover_g"]].mean().reindex(month_order)
valid = monthly.notna().all(axis=1)
ax.stackplot(monthly.index[valid],
             monthly.loc[valid,"Dry_Green_g"],
             monthly.loc[valid,"Dry_Dead_g"],
             monthly.loc[valid,"Dry_Clover_g"],
             labels=["Green","Dead","Clover"],
             colors=[COMP_COLORS["Dry_Green_g"], COMP_COLORS["Dry_Dead_g"], COMP_COLORS["Dry_Clover_g"]],
             alpha=0.75)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("Mean Biomass (g)")
ax.set_title("(b) Seasonal Composition Trend", fontweight="bold")
ax.legend(frameon=True, loc="upper left", fontsize=9)

# 8c: NDVI seasonal trend
ax = axes[2]
for state in ["NSW","Tas","Vic","WA"]:
    sub = pivot[pivot["State"] == state]
    grp = sub.groupby("Month")["Pre_GSHH_NDVI"].agg(["mean","std"])
    grp = grp.reindex(month_order)
    valid = grp["mean"].notna()
    ax.plot(grp.index[valid], grp["mean"][valid], "s-", color=STATE_COLORS[state],
            label=state, linewidth=2.2, markersize=6)
    ax.fill_between(grp.index[valid],
                    (grp["mean"] - grp["std"])[valid],
                    (grp["mean"] + grp["std"])[valid],
                    color=STATE_COLORS[state], alpha=0.1)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("NDVI")
ax.set_title("(c) NDVI Seasonal Trend", fontweight="bold")
ax.legend(title="State", frameon=True, fontsize=9)

fig.suptitle("Seasonal Dynamics: Biomass, Composition, and Vegetation Index",
             fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig08_seasonal_dynamics")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Species × Biomass Violin with GDM weight context
# Paper: GDM_g weighted 0.2 in evaluation, Dry_Total weighted 0.5
# ══════════════════════════════════════════════════════════════════════════════
top8 = pivot["Species"].value_counts().head(8).index.tolist()
sub = pivot[pivot["Species"].isin(top8)].copy()
sub["Species_clean"] = sub["Species"].str.replace("_", " ")
species_order_clean = [s.replace("_", " ") for s in top8]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for i, (tgt, lbl) in enumerate([("Dry_Total_g","Dry Total (g) — eval weight 0.50"),
                                  ("GDM_g",      "GDM (g) — eval weight 0.20")]):
    palette8 = sns.color_palette("husl", 8)
    sns.violinplot(data=sub, x="Species_clean", y=tgt, order=species_order_clean,
                   palette=palette8, inner="quartile", linewidth=1.0, ax=axes[i], cut=0)
    axes[i].set_xlabel("Pasture Species")
    axes[i].set_ylabel(lbl)
    axes[i].set_title(lbl, fontweight="bold")
    plt.sca(axes[i])
    plt.xticks(rotation=25, ha="right")

fig.suptitle("Biomass Distribution by Species (Top 8) — Evaluation-Weighted Targets",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig09_species_violin")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Joint Distribution: NDVI × Height → Biomass (2D KDE + Hexbin)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 10a: Hexbin
ax = axes[0]
hb = ax.hexbin(pivot["Pre_GSHH_NDVI"], pivot["Height_Ave_cm"],
               C=pivot["Dry_Total_g"], gridsize=18, cmap="YlOrRd",
               mincnt=1, reduce_C_function=np.mean)
fig.colorbar(hb, ax=ax, label="Mean Dry Total (g)", shrink=0.85)
ax.set_xlabel("NDVI")
ax.set_ylabel("Height (cm)")
ax.set_title("(a) NDVI × Height → Biomass (hexbin)", fontweight="bold")

# 10b: 2D KDE contour for Dry_Total
ax = axes[1]
for state in ["NSW","Tas","Vic","WA"]:
    m = pivot["State"] == state
    if m.sum() > 10:
        sns.kdeplot(x=pivot.loc[m, "Pre_GSHH_NDVI"], y=pivot.loc[m, "Height_Ave_cm"],
                    ax=ax, color=STATE_COLORS[state], levels=3, linewidths=1.5, alpha=0.7)
    ax.scatter(pivot.loc[m, "Pre_GSHH_NDVI"], pivot.loc[m, "Height_Ave_cm"],
               c=STATE_COLORS[state], label=state, alpha=0.3, s=15, edgecolors="none")
ax.set_xlabel("NDVI")
ax.set_ylabel("Height (cm)")
ax.set_title("(b) Feature Space Density by State", fontweight="bold")
ax.legend(title="State", frameon=True, markerscale=2)

fig.suptitle("Joint Feature Distribution: NDVI × Height",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig10_joint_ndvi_height")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Image Color Features vs Biomass
# ExG = 2G - R - B (excess green index) — standard vegetation detection
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
color_features = [
    ("ExG", "Excess Green Index (ExG)", "GDM_g", "GDM (g)"),
    ("Greenness", "Image Greenness (G/(R+G+B))", "Dry_Green_g", "Dry Green (g)"),
    ("Brightness", "Mean Brightness", "Dry_Dead_g", "Dry Dead (g)"),
]
for i, (xf, xlbl, yf, ylbl) in enumerate(color_features):
    for state in ["NSW","Tas","Vic","WA"]:
        m = pivot["State"] == state
        axes[i].scatter(pivot.loc[m, xf], pivot.loc[m, yf],
                        c=STATE_COLORS[state], label=state, alpha=0.5, s=25,
                        edgecolors="white", linewidth=0.3)
    r, _ = stats.pearsonr(pivot[xf].dropna(), pivot[yf].dropna())
    rho, _ = stats.spearmanr(pivot[xf].dropna(), pivot[yf].dropna())
    axes[i].text(0.05, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}",
                 transform=axes[i].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    axes[i].set_xlabel(xlbl)
    axes[i].set_ylabel(ylbl)
    axes[i].set_title(f"{xlbl} vs. {ylbl}", fontweight="bold", fontsize=11)
    if i == 0:
        axes[i].legend(title="State", frameon=True, markerscale=1.5, fontsize=8)

fig.suptitle("Image-Derived Color Features vs. Biomass Components\n"
             "Visual features available at inference — metadata (NDVI, Height) is NOT",
             fontweight="bold", fontsize=13, y=1.04)
fig.tight_layout()
save(fig, "fig11_image_color_vs_biomass")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Evaluation Weight Analysis
# The paper weights: Dry_Total=0.5, GDM=0.2, others=0.1 each
# Show what fraction of the score each target contributes
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 12a: Weights visualization
weights = {"Dry_Total_g": 0.5, "GDM_g": 0.2, "Dry_Clover_g": 0.1, "Dry_Dead_g": 0.1, "Dry_Green_g": 0.1}
w_labels = [TARGET_LABELS[k] for k in weights.keys()]
w_values = list(weights.values())
w_colors = [TARGET_COLORS[k] for k in weights.keys()]

wedges, texts, autotexts = axes[0].pie(
    w_values, labels=w_labels, colors=w_colors,
    autopct=lambda p: f"{p:.0f}%", startangle=140, pctdistance=0.65,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2.5))
for t in autotexts:
    t.set_fontweight("bold")
    t.set_fontsize(11)
axes[0].set_title("(a) Evaluation Weight Distribution\n(Weighted R² Score)", fontweight="bold")

# 12b: Target difficulty — std/mean (CV) as proxy for difficulty
ax = axes[1]
cv_data = []
for tgt in TARGETS:
    vals = pivot[tgt]
    log_vals = np.log1p(vals)
    cv_raw = vals.std() / (vals.mean() + 1e-8)
    cv_log = log_vals.std() / (log_vals.mean() + 1e-8)
    zero_frac = (vals == 0).mean()
    cv_data.append({
        "Target": TARGET_LABELS[tgt],
        "CV (raw)": cv_raw,
        "CV (log)": cv_log,
        "Zero fraction": zero_frac,
        "Weight": weights[tgt],
    })
cv_df = pd.DataFrame(cv_data)

x = np.arange(len(TARGETS))
w = 0.3
b1 = ax.bar(x - w/2, cv_df["CV (raw)"], w, label="CV (raw)", color="#3C5488", alpha=0.8, edgecolor="white")
b2 = ax.bar(x + w/2, cv_df["CV (log)"], w, label="CV (log)", color="#00A087", alpha=0.8, edgecolor="white")
# Overlay zero fraction on twin axis
ax2 = ax.twinx()
ax2.plot(x, cv_df["Zero fraction"]*100, "D-", color="#E64B35", linewidth=2, markersize=8, label="% zeros")
ax2.set_ylabel("% Zero Values", color="#E64B35")
ax2.tick_params(axis="y", labelcolor="#E64B35")

ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Coefficient of Variation")
ax.set_title("(b) Target Difficulty: Variability & Zero-Inflation", fontweight="bold")
ax.legend(loc="upper left", frameon=True, fontsize=9)
ax2.legend(loc="upper right", frameon=True, fontsize=9)

fig.suptitle("Evaluation Framework Analysis: Weights, Difficulty, and Zero-Inflation",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig12_evaluation_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Empirical CDFs (all targets, state-stratified)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for i, (tgt, lbl) in enumerate([("Dry_Total_g","Dry Total"),("GDM_g","GDM"),("Dry_Green_g","Dry Green")]):
    for state in ["NSW","Tas","Vic","WA"]:
        vals = pivot.loc[pivot["State"]==state, tgt].sort_values()
        cdf = np.arange(1, len(vals)+1) / len(vals)
        axes[i].step(vals, cdf, where="post", color=STATE_COLORS[state], label=state, linewidth=2)
    axes[i].set_xlabel(f"{lbl} (g)")
    axes[i].set_ylabel("Cumulative Probability")
    axes[i].set_title(f"ECDF: {lbl}", fontweight="bold")
    if i == 0:
        axes[i].legend(title="State", frameon=True)
    axes[i].set_xlim(0, None)
fig.suptitle("Empirical Cumulative Distribution Functions by State",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig13_ecdf_by_state")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Species × State Heatmap (Presence & Mean Biomass)
# ══════════════════════════════════════════════════════════════════════════════
top10_sp = pivot["Species"].value_counts().head(10).index.tolist()
sub_heat = pivot[pivot["Species"].isin(top10_sp)]
heat_count = sub_heat.groupby(["Species","State"]).size().unstack(fill_value=0)
heat_mean  = sub_heat.groupby(["Species","State"])["Dry_Total_g"].mean().unstack(fill_value=np.nan)
heat_count = heat_count.reindex(index=top10_sp, columns=["NSW","Tas","Vic","WA"], fill_value=0)
heat_mean  = heat_mean.reindex(index=top10_sp,  columns=["NSW","Tas","Vic","WA"])
heat_count.index = [s.replace("_"," ") for s in heat_count.index]
heat_mean.index  = [s.replace("_"," ") for s in heat_mean.index]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(heat_count, annot=True, fmt="d", cmap="Blues", linewidths=1.5,
            linecolor="white", ax=axes[0], cbar_kws={"shrink": 0.7, "label": "Sample Count"})
axes[0].set_title("(a) Sample Count: Species × State", fontweight="bold")
axes[0].set_xlabel("State"); axes[0].set_ylabel("Pasture Species")

sns.heatmap(heat_mean, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=1.5,
            linecolor="white", ax=axes[1], cbar_kws={"shrink": 0.7, "label": "Mean Dry Total (g)"},
            mask=heat_mean.isna())
axes[1].set_title("(b) Mean Biomass: Species × State", fontweight="bold")
axes[1].set_xlabel("State"); axes[1].set_ylabel("")

fig.suptitle("Species–State Interaction: Coverage and Biomass",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig14_species_state_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — Pair Plot (key features, state-colored)
# ══════════════════════════════════════════════════════════════════════════════
pair_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "Dry_Total_g", "GDM_g", "ExG", "State"]
pair_df = pivot[pair_cols].copy()
pair_df.columns = ["NDVI", "Height (cm)", "Dry Total (g)", "GDM (g)", "ExG Index", "State"]

g = sns.pairplot(pair_df, hue="State", palette=STATE_COLORS,
                 diag_kind="kde", plot_kws={"alpha": 0.45, "s": 16, "edgecolor": "white", "linewidth": 0.2},
                 diag_kws={"linewidth": 1.5}, height=2.6, aspect=1)
g.figure.suptitle("Pair Plot: Key Features + Image ExG — Colored by State",
                   fontweight="bold", y=1.02, fontsize=14)
save(g.figure, "fig15_pairplot")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — Radar Chart: State Feature Profiles
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ["NDVI", "Height", "Dry Green", "Dry Dead", "Dry Clover", "ExG", "Brightness"]
state_raw = {}
for state in ["NSW","Tas","Vic","WA"]:
    sub = pivot[pivot["State"]==state]
    state_raw[state] = [sub["Pre_GSHH_NDVI"].mean(), sub["Height_Ave_cm"].mean(),
                        sub["Dry_Green_g"].mean(), sub["Dry_Dead_g"].mean(),
                        sub["Dry_Clover_g"].mean(), sub["ExG"].mean(), sub["Brightness"].mean()]

all_vals = np.array(list(state_raw.values()))
mins, maxs = all_vals.min(0), all_vals.max(0)
ranges = maxs - mins; ranges[ranges==0] = 1

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for state in ["NSW","Tas","Vic","WA"]:
    normed = (np.array(state_raw[state]) - mins) / ranges
    vals = normed.tolist() + normed[:1].tolist()
    ax.plot(angles, vals, "o-", linewidth=2.2, color=STATE_COLORS[state], label=state, markersize=5)
    ax.fill(angles, vals, color=STATE_COLORS[state], alpha=0.06)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title("State Feature Profiles (Normalized)\nMetadata + Image-Derived Features",
             fontweight="bold", fontsize=13, pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=True, title="State")
fig.tight_layout()
save(fig, "fig16_radar_state_profiles")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17 — Summary Statistics Table
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")

summary_data = []
eval_weights = {"Dry_Clover_g": 0.1, "Dry_Dead_g": 0.1, "Dry_Green_g": 0.1,
                "Dry_Total_g": 0.5, "GDM_g": 0.2}
for tgt in TARGETS:
    v = pivot[tgt]
    summary_data.append([
        TARGET_LABELS[tgt], f"{eval_weights[tgt]:.1f}",
        f"{v.count()}", f"{v.mean():.2f}", f"{v.std():.2f}",
        f"{v.min():.2f}", f"{v.quantile(.25):.2f}", f"{v.median():.2f}",
        f"{v.quantile(.75):.2f}", f"{v.max():.2f}",
        f"{v.skew():.2f}", f"{(v==0).mean()*100:.1f}%"
    ])

col_labels = ["Variable", "Eval\nWeight", "N", "Mean", "Std", "Min", "Q1",
              "Median", "Q3", "Max", "Skew", "% Zero"]
table = ax.table(cellText=summary_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#3C5488")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)
for i in range(1, len(summary_data)+1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor("#F0F4F8" if i%2==0 else "#FFFFFF")
ax.set_title("Summary Statistics of Biomass Target Variables (Training Set, n=357 images)",
             fontweight="bold", fontsize=14, pad=20)
fig.tight_layout()
save(fig, "fig17_summary_table")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 18-21 — Image + Metadata Side-by-Side Panels (4 diverse samples)
# Select images that represent diversity: different states, species, biomass
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating image + metadata panels …")

# Select 6 diverse, interesting samples
selection_criteria = [
    # (state, description, sort col, quantile_fraction, ascending)
    ("NSW", "High Green Biomass (NSW)", "Dry_Green_g", 0.95, False),
    ("Tas", "High Dead Biomass (Tas)", "Dry_Dead_g", 0.92, False),
    ("Vic", "Balanced Composition (Vic)", "Dry_Total_g", 0.50, True),
    ("WA",  "Low Biomass / Sparse (WA)", "Dry_Total_g", 0.20, True),
    ("Tas", "Maximum Biomass (Tas)", "Dry_Total_g", 0.99, False),
    ("Vic", "High Clover Content (Vic)", "Dry_Clover_g", 0.90, False),
]

for fig_idx, (state, desc, sort_col, q, asc) in enumerate(selection_criteria):
    sub = pivot[pivot["State"] == state].copy()
    sub = sub.sort_values(sort_col, ascending=asc)
    # Pick sample near given quantile
    idx = int(len(sub) * (1 - q if not asc else q))
    idx = max(0, min(idx, len(sub)-1))
    row = sub.iloc[idx]
    
    img_path = os.path.join(TRAIN_DIR, os.path.basename(row["image_path"]))
    img = Image.open(img_path)
    
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(2, 4, width_ratios=[2.5, 1, 1, 1], hspace=0.35, wspace=0.4)
    
    # Left: Full image (spans 2 rows)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.set_title(f"Sample: {os.path.basename(row['image_path'])}\n"
                     f"State: {row['State']} | Species: {row['Species'].replace('_',' ')}\n"
                     f"Date: {row['Sampling_Date'].strftime('%Y-%m-%d')}",
                     fontweight="bold", fontsize=11)
    ax_img.axis("off")
    # Add quadrat size annotation
    ax_img.text(0.5, -0.03, "70 cm × 30 cm quadrat  |  2000 × 1000 px",
                transform=ax_img.transAxes, ha="center", fontsize=9, style="italic", color="gray")
    
    # Top-right 1: Biomass composition pie
    ax_pie = fig.add_subplot(gs[0, 1])
    comp_vals = [row["Dry_Green_g"], row["Dry_Dead_g"], row["Dry_Clover_g"]]
    comp_names = ["Green", "Dead", "Clover"]
    comp_cols  = ["#00A087", "#B09C85", "#3C5488"]
    nonzero = [(v, n, c) for v, n, c in zip(comp_vals, comp_names, comp_cols) if v > 0]
    if nonzero:
        vals_nz, names_nz, cols_nz = zip(*nonzero)
        ax_pie.pie(vals_nz, labels=names_nz, colors=cols_nz,
                   autopct=lambda p: f"{p:.0f}%", startangle=90,
                   wedgeprops=dict(edgecolor="white", linewidth=1.5),
                   textprops={"fontsize": 9})
    ax_pie.set_title("Biomass Composition", fontweight="bold", fontsize=10)
    
    # Top-right 2: Target bar chart
    ax_bar = fig.add_subplot(gs[0, 2])
    tgt_vals = [row[t] for t in TARGETS]
    tgt_cols = [TARGET_COLORS[t] for t in TARGETS]
    tgt_short = ["Clover", "Dead", "Green", "Total", "GDM"]
    bars = ax_bar.barh(range(5), tgt_vals, color=tgt_cols, edgecolor="white", height=0.6)
    ax_bar.set_yticks(range(5))
    ax_bar.set_yticklabels(tgt_short, fontsize=9)
    for bar, v in zip(bars, tgt_vals):
        ax_bar.text(v + 0.5, bar.get_y() + bar.get_height()/2, f"{v:.1f}g",
                    va="center", fontsize=8, fontweight="bold")
    ax_bar.set_xlabel("Biomass (g)", fontsize=9)
    ax_bar.set_title("Target Values", fontweight="bold", fontsize=10)
    ax_bar.invert_yaxis()
    
    # Top-right 3: Metadata gauges
    ax_meta = fig.add_subplot(gs[0, 3])
    ax_meta.axis("off")
    meta_text = (
        f"NDVI:  {row['Pre_GSHH_NDVI']:.2f}\n"
        f"Height: {row['Height_Ave_cm']:.1f} cm\n"
        f"───────────────\n"
        f"Total:  {row['Dry_Total_g']:.1f} g\n"
        f"GDM:    {row['GDM_g']:.1f} g\n"
        f"───────────────\n"
        f"Green%: {row.get('Green_Fraction', 0)*100:.0f}%\n"
        f"Dead%:  {row.get('Dead_Fraction', 0)*100:.0f}%\n"
        f"Clover%:{row.get('Clover_Fraction', 0)*100:.0f}%"
    )
    ax_meta.text(0.1, 0.95, meta_text, transform=ax_meta.transAxes,
                 va="top", fontsize=10, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#F0F4F8", ec="gray", alpha=0.9))
    ax_meta.set_title("Metadata", fontweight="bold", fontsize=10)
    
    # Bottom-right 1: RGB histogram
    ax_rgb = fig.add_subplot(gs[1, 1])
    img_arr = np.array(img)
    for ch, color, name in [(0,"#E64B35","R"), (1,"#00A087","G"), (2,"#3C5488","B")]:
        ax_rgb.hist(img_arr[:,:,ch].ravel(), bins=64, alpha=0.45, color=color,
                    label=name, density=True)
    ax_rgb.set_xlabel("Pixel Value", fontsize=9)
    ax_rgb.set_ylabel("Density", fontsize=9)
    ax_rgb.set_title("RGB Histogram", fontweight="bold", fontsize=10)
    ax_rgb.legend(fontsize=8, frameon=True)
    ax_rgb.set_xlim(0, 255)
    
    # Bottom-right 2: Position in dataset (NDVI vs Height)
    ax_pos = fig.add_subplot(gs[1, 2])
    ax_pos.scatter(pivot["Pre_GSHH_NDVI"], pivot["Height_Ave_cm"],
                   c="lightgray", s=12, alpha=0.4, edgecolors="none")
    ax_pos.scatter(row["Pre_GSHH_NDVI"], row["Height_Ave_cm"],
                   c=STATE_COLORS[state], s=120, edgecolors="black", linewidth=1.5,
                   zorder=10, marker="*")
    ax_pos.set_xlabel("NDVI", fontsize=9)
    ax_pos.set_ylabel("Height (cm)", fontsize=9)
    ax_pos.set_title("Position in Feature Space", fontweight="bold", fontsize=10)
    
    # Bottom-right 3: Position in biomass space (Total vs GDM)
    ax_bm = fig.add_subplot(gs[1, 3])
    ax_bm.scatter(pivot["Dry_Total_g"], pivot["GDM_g"],
                  c="lightgray", s=12, alpha=0.4, edgecolors="none")
    ax_bm.scatter(row["Dry_Total_g"], row["GDM_g"],
                  c=STATE_COLORS[state], s=120, edgecolors="black", linewidth=1.5,
                  zorder=10, marker="*")
    ax_bm.set_xlabel("Dry Total (g)", fontsize=9)
    ax_bm.set_ylabel("GDM (g)", fontsize=9)
    ax_bm.set_title("Position in Biomass Space", fontweight="bold", fontsize=10)
    
    fig.suptitle(f"Sample Analysis: {desc}",
                 fontweight="bold", fontsize=14, y=1.01)
    
    save(fig, f"fig{18 + fig_idx:02d}_sample_panel_{state.lower()}_{fig_idx+1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 24 — RGB Channel Distributions by State (KDE overlay)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (col, ch_name, ch_color) in enumerate([("R_mean","Red","#E64B35"),
                                                ("G_mean","Green","#00A087"),
                                                ("B_mean","Blue","#3C5488")]):
    for state in ["NSW","Tas","Vic","WA"]:
        vals = pivot.loc[pivot["State"]==state, col]
        sns.kdeplot(vals, ax=axes[i], label=state, color=STATE_COLORS[state],
                    linewidth=2.5, fill=True, alpha=0.1)
    axes[i].set_xlabel(f"Mean {ch_name} Channel Value")
    axes[i].set_ylabel("Density")
    axes[i].set_title(f"{ch_name} Channel Distribution", fontweight="bold")
    if i == 0:
        axes[i].legend(title="State", frameon=True)
fig.suptitle("Image RGB Channel Distributions by State\n"
             "Camera diversity: iPhone, Canon, Nikon, Olympus, Sony, HTC",
             fontweight="bold", fontsize=13, y=1.04)
fig.tight_layout()
save(fig, "fig24_rgb_channel_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 25 — QC Context: Range Analysis
# Paper: 1162/3187 passed QC. Show range and outlier context.
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 25a: Box plots for all targets on same scale
ax = axes[0]
data_for_box = [pivot[tgt].values for tgt in TARGETS]
bp = ax.boxplot(data_for_box, labels=[TARGET_LABELS[t].replace(" (g)","") for t in TARGETS],
                patch_artist=True, notch=True, widths=0.5,
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
for patch, tgt in zip(bp["boxes"], TARGETS):
    patch.set_facecolor(TARGET_COLORS[tgt])
    patch.set_alpha(0.7)
ax.set_ylabel("Biomass (g)")
ax.set_title("(a) Target Range Comparison (notched box plots)", fontweight="bold")

# 25b: IQR and outlier fraction
ax = axes[1]
outlier_data = []
for tgt in TARGETS:
    v = pivot[tgt]
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    n_outliers = (v > upper_fence).sum()
    outlier_data.append({"Target": TARGET_LABELS[tgt].replace(" (g)",""),
                         "IQR": iqr, "Outlier %": n_outliers/len(v)*100})
outlier_df = pd.DataFrame(outlier_data)
x = np.arange(5)
b1 = ax.bar(x - 0.18, outlier_df["IQR"], 0.35, label="IQR", color="#3C5488", alpha=0.8, edgecolor="white")
ax2 = ax.twinx()
b2 = ax2.bar(x + 0.18, outlier_df["Outlier %"], 0.35, label="Outlier %", color="#E64B35", alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(outlier_df["Target"], fontsize=9)
ax.set_ylabel("IQR (g)", color="#3C5488")
ax2.set_ylabel("Outlier % (>Q3+1.5·IQR)", color="#E64B35")
ax.legend(loc="upper left", frameon=True)
ax2.legend(loc="upper right", frameon=True)
ax.set_title("(b) Spread & Outlier Analysis", fontweight="bold")

fig.suptitle("Data Quality & Range Analysis\n"
             "Paper: 1,162/3,187 samples passed rigorous QC (36.5% retention)",
             fontweight="bold", fontsize=13, y=1.04)
fig.tight_layout()
save(fig, "fig25_range_outlier_analysis")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total_figs = len([f for f in os.listdir(FIGDIR_PNG) if f.endswith(".png")])
print(f"Done! {total_figs} figures saved.")
print(f"  PNG: {FIGDIR_PNG}")
print(f"  SVG: {FIGDIR_SVG}")
print(f"{'='*60}")
