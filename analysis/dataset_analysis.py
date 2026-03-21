# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
CSIRO BioMass Dataset — Research-Grade Exploratory Data Analysis
================================================================
Produces publication-quality figures for the research paper.
All plots saved as PNG (300 dpi) and SVG in ./figures/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")

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

# Vibrant, accessible palette (CB-friendly)
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

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.svg"))
    plt.close(fig)
    print(f"  ✓ {name}")

# ── Load & prepare data ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
df["Month"] = df["Sampling_Date"].dt.month
df["MonthName"] = df["Sampling_Date"].dt.strftime("%b")

# Pivot: one row per image
TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
pivot = df.pivot_table(
    index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm", "Month", "MonthName"],
    columns="target_name", values="target"
).reset_index()

# Derived features
pivot["Green_Fraction"] = pivot["Dry_Green_g"] / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["Dead_Fraction"]  = pivot["Dry_Dead_g"]  / pivot["Dry_Total_g"].replace(0, np.nan)

# Ordered month names
month_order = sorted(pivot["Month"].unique())
month_labels = [pd.Timestamp(2015, m, 1).strftime("%b") for m in month_order]

print(f"Dataset: {len(pivot)} images, {len(df)} rows (multi-target)\n")
print("Generating figures …")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geographic & Species Distribution (State bar + Species bar)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a: State distribution
state_counts = pivot["State"].value_counts().reindex(["NSW", "Tas", "Vic", "WA"])
bars = axes[0].bar(state_counts.index, state_counts.values,
                   color=[STATE_COLORS[s] for s in state_counts.index],
                   edgecolor="white", linewidth=1.2, width=0.6)
for bar, v in zip(bars, state_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 2, str(v),
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
axes[0].set_xlabel("Australian State")
axes[0].set_ylabel("Number of Samples")
axes[0].set_title("(a) Sample Distribution by State", fontweight="bold")
axes[0].set_ylim(0, state_counts.max() * 1.15)

# 1b: Species distribution (top 10)
species_counts = pivot["Species"].value_counts().head(10)
species_palette = sns.color_palette("Set2", n_colors=len(species_counts))
bars2 = axes[1].barh(range(len(species_counts)), species_counts.values,
                      color=species_palette, edgecolor="white", linewidth=1.0)
axes[1].set_yticks(range(len(species_counts)))
axes[1].set_yticklabels([s.replace("_", " ") for s in species_counts.index], fontsize=9)
for bar, v in zip(bars2, species_counts.values):
    axes[1].text(v + 1, bar.get_y() + bar.get_height()/2, str(v),
                 ha="left", va="center", fontsize=9)
axes[1].set_xlabel("Number of Samples")
axes[1].set_title("(b) Top-10 Pasture Species", fontweight="bold")
axes[1].invert_yaxis()
axes[1].set_xlim(0, species_counts.max() * 1.15)

fig.tight_layout(w_pad=4)
save(fig, "fig1_geographic_species_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Temporal Sampling Distribution (grouped bar by State × Month)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
temporal = pivot.groupby(["Month", "State"]).size().unstack(fill_value=0)
temporal = temporal.reindex(month_order).reindex(columns=["NSW", "Tas", "Vic", "WA"], fill_value=0)

x = np.arange(len(month_order))
width = 0.18
states_list = ["NSW", "Tas", "Vic", "WA"]
for i, state in enumerate(states_list):
    vals = temporal[state].values
    bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                  label=state, color=STATE_COLORS[state],
                  edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_xlabel("Sampling Month (2015)")
ax.set_ylabel("Number of Samples")
ax.set_title("Temporal Sampling Distribution by State", fontweight="bold")
ax.legend(title="State", frameon=True, fancybox=True, shadow=False)
ax.set_ylim(0, temporal.values.max() * 1.2)
fig.tight_layout()
save(fig, "fig2_temporal_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Target Variable Distributions (histograms + KDE)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
for i, tgt in enumerate(TARGETS):
    vals = pivot[tgt].dropna()
    axes[i].hist(vals, bins=35, color=TARGET_COLORS[tgt], alpha=0.7,
                 edgecolor="white", linewidth=0.5, density=True)
    sns.kdeplot(vals, ax=axes[i], color="black", linewidth=1.5)
    axes[i].set_title(TARGET_LABELS[tgt], fontweight="bold", fontsize=11)
    axes[i].set_xlabel("Biomass (g)")
    if i == 0:
        axes[i].set_ylabel("Density")
    # Add stats annotation
    med = vals.median()
    axes[i].axvline(med, color="red", linestyle="--", linewidth=1, alpha=0.8)
    axes[i].text(0.95, 0.92, f"μ={vals.mean():.1f}\nσ={vals.std():.1f}\nMd={med:.1f}",
                 transform=axes[i].transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
fig.suptitle("Distribution of Biomass Target Variables", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig3_target_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Box Plots: Targets by State
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
for i, tgt in enumerate(TARGETS):
    sns.boxplot(data=pivot, x="State", y=tgt, order=["NSW","Tas","Vic","WA"],
                palette=STATE_COLORS, ax=axes[i], width=0.5, linewidth=1.2,
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
    axes[i].set_title(TARGET_LABELS[tgt], fontweight="bold", fontsize=11)
    axes[i].set_xlabel("")
    if i > 0:
        axes[i].set_ylabel("")
    else:
        axes[i].set_ylabel("Biomass (g)")
fig.suptitle("Biomass Distribution by Australian State", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig4_target_by_state_boxplots")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Correlation Heatmap (metadata + targets)
# ══════════════════════════════════════════════════════════════════════════════
corr_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm"] + TARGETS
corr_labels = ["NDVI", "Height (cm)", "Dry Clover", "Dry Dead", "Dry Green", "Dry Total", "GDM"]
corr_mat = pivot[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 15, s=90, l=45, as_cmap=True)
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=corr_labels, yticklabels=corr_labels,
            linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            ax=ax, annot_kws={"size": 10, "fontweight": "bold"})
ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14, pad=12)
plt.xticks(rotation=40, ha="right")
fig.tight_layout()
save(fig, "fig5_correlation_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — NDVI vs. Biomass Targets (scatter with regression)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_targets = ["Dry_Green_g", "Dry_Total_g", "GDM_g"]
scatter_labels  = ["Dry Green (g)", "Dry Total (g)", "GDM (g)"]
for i, (tgt, lbl) in enumerate(zip(scatter_targets, scatter_labels)):
    for state in ["NSW", "Tas", "Vic", "WA"]:
        mask = pivot["State"] == state
        axes[i].scatter(pivot.loc[mask, "Pre_GSHH_NDVI"], pivot.loc[mask, tgt],
                        c=STATE_COLORS[state], label=state, alpha=0.55, s=25, edgecolors="white", linewidth=0.3)
    # Overall regression line
    x = pivot["Pre_GSHH_NDVI"].values
    y = pivot[tgt].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[i].plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5, alpha=0.7)
    axes[i].text(0.05, 0.95, f"r = {r:.3f}\np < {max(p, 1e-10):.1e}",
                 transform=axes[i].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    axes[i].set_xlabel("Pre-Grazing NDVI")
    axes[i].set_ylabel(lbl)
    axes[i].set_title(f"NDVI vs. {lbl}", fontweight="bold")
    if i == 2:
        axes[i].legend(title="State", frameon=True, markerscale=1.5)
fig.suptitle("Relationship Between NDVI and Biomass Components", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig6_ndvi_vs_biomass")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Height vs. Biomass Targets (scatter with regression)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (tgt, lbl) in enumerate(zip(scatter_targets, scatter_labels)):
    for state in ["NSW", "Tas", "Vic", "WA"]:
        mask = pivot["State"] == state
        axes[i].scatter(pivot.loc[mask, "Height_Ave_cm"], pivot.loc[mask, tgt],
                        c=STATE_COLORS[state], label=state, alpha=0.55, s=25, edgecolors="white", linewidth=0.3)
    x = pivot["Height_Ave_cm"].values
    y = pivot[tgt].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[i].plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5, alpha=0.7)
    axes[i].text(0.05, 0.95, f"r = {r:.3f}\np < {max(p, 1e-10):.1e}",
                 transform=axes[i].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    axes[i].set_xlabel("Average Height (cm)")
    axes[i].set_ylabel(lbl)
    axes[i].set_title(f"Height vs. {lbl}", fontweight="bold")
    if i == 2:
        axes[i].legend(title="State", frameon=True, markerscale=1.5)
fig.suptitle("Relationship Between Canopy Height and Biomass Components", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig7_height_vs_biomass")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Violin: Dry Total by Species (top 8)
# ══════════════════════════════════════════════════════════════════════════════
top8 = pivot["Species"].value_counts().head(8).index.tolist()
sub = pivot[pivot["Species"].isin(top8)].copy()
sub["Species_clean"] = sub["Species"].str.replace("_", " ")
species_order_clean = [s.replace("_", " ") for s in top8]

fig, ax = plt.subplots(figsize=(14, 6))
palette8 = sns.color_palette("husl", n_colors=8)
sns.violinplot(data=sub, x="Species_clean", y="Dry_Total_g", order=species_order_clean,
               palette=palette8, inner="quartile", linewidth=1.2, ax=ax, cut=0)
ax.set_xlabel("Pasture Species")
ax.set_ylabel("Dry Total Biomass (g)")
ax.set_title("Distribution of Total Dry Biomass by Pasture Species (Top 8)", fontweight="bold")
plt.xticks(rotation=25, ha="right")
fig.tight_layout()
save(fig, "fig8_biomass_by_species_violin")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Seasonal Biomass Trend (line plot, mean ± std by month & state)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for tgt, lbl, ax in zip(["Dry_Total_g", "GDM_g"], ["Dry Total (g)", "GDM (g)"], axes):
    for state in ["NSW", "Tas", "Vic", "WA"]:
        sub_s = pivot[pivot["State"] == state]
        grp = sub_s.groupby("Month")[tgt].agg(["mean", "std"])
        grp = grp.reindex(month_order)
        valid = grp["mean"].notna()
        ax.plot(grp.index[valid], grp["mean"][valid], "o-", color=STATE_COLORS[state],
                label=state, linewidth=2, markersize=5)
        ax.fill_between(grp.index[valid],
                        (grp["mean"] - grp["std"])[valid],
                        (grp["mean"] + grp["std"])[valid],
                        color=STATE_COLORS[state], alpha=0.12)
    ax.set_xticks(month_order)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month (2015)")
    ax.set_ylabel(lbl)
    ax.set_title(f"Seasonal Trend of {lbl} by State", fontweight="bold")
    ax.legend(title="State", frameon=True)
fig.tight_layout()
save(fig, "fig9_seasonal_biomass_trend")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Biomass Composition Stacked Bar by State
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))
components = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
comp_labels = ["Dry Clover", "Dry Dead", "Dry Green"]
comp_colors = [TARGET_COLORS[c] for c in components]
state_order = ["NSW", "Tas", "Vic", "WA"]
means = pivot.groupby("State")[components].mean().reindex(state_order)

bottom = np.zeros(len(state_order))
for comp, lbl, color in zip(components, comp_labels, comp_colors):
    vals = means[comp].values
    ax.bar(state_order, vals, bottom=bottom, label=lbl, color=color,
           edgecolor="white", linewidth=1.0, width=0.55)
    # label inside each segment
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 2:
            ax.text(j, b + v/2, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom += vals

ax.set_ylabel("Mean Biomass (g)")
ax.set_xlabel("Australian State")
ax.set_title("Mean Biomass Composition by State", fontweight="bold")
ax.legend(frameon=True, loc="upper left")
fig.tight_layout()
save(fig, "fig10_biomass_composition_stacked")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Pair Plot: NDVI, Height, Dry_Total, GDM colored by State
# ══════════════════════════════════════════════════════════════════════════════
pair_df = pivot[["Pre_GSHH_NDVI", "Height_Ave_cm", "Dry_Total_g", "GDM_g", "State"]].copy()
pair_df.columns = ["NDVI", "Height (cm)", "Dry Total (g)", "GDM (g)", "State"]

g = sns.pairplot(pair_df, hue="State", palette=STATE_COLORS,
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18, "edgecolor": "white", "linewidth": 0.3},
                 diag_kws={"linewidth": 1.5}, height=2.8, aspect=1)
g.figure.suptitle("Pair Plot of Key Features Colored by State", fontweight="bold", y=1.02, fontsize=14)
save(g.figure, "fig11_pairplot")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — NDVI Distribution by State (Ridge-like KDE)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
for state in ["NSW", "Tas", "Vic", "WA"]:
    vals = pivot.loc[pivot["State"] == state, "Pre_GSHH_NDVI"]
    sns.kdeplot(vals, ax=ax, label=state, color=STATE_COLORS[state],
                linewidth=2.5, fill=True, alpha=0.15, clip=(0, 1))
ax.set_xlabel("Pre-Grazing NDVI")
ax.set_ylabel("Density")
ax.set_title("Distribution of NDVI by State", fontweight="bold")
ax.legend(title="State", frameon=True)
fig.tight_layout()
save(fig, "fig12_ndvi_by_state_kde")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Height distribution by State (Box + Swarm)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=pivot, x="State", y="Height_Ave_cm", order=["NSW","Tas","Vic","WA"],
            palette=STATE_COLORS, width=0.45, linewidth=1.2,
            fliersize=0, ax=ax)
sns.stripplot(data=pivot, x="State", y="Height_Ave_cm", order=["NSW","Tas","Vic","WA"],
              palette=STATE_COLORS, alpha=0.35, size=3, jitter=0.25, ax=ax)
ax.set_xlabel("Australian State")
ax.set_ylabel("Average Canopy Height (cm)")
ax.set_title("Canopy Height Distribution by State", fontweight="bold")
fig.tight_layout()
save(fig, "fig13_height_by_state")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Green Fraction vs Dead Fraction scatter (composition analysis)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
valid_fracs = pivot.dropna(subset=["Green_Fraction", "Dead_Fraction"])
for state in ["NSW", "Tas", "Vic", "WA"]:
    mask = valid_fracs["State"] == state
    ax.scatter(valid_fracs.loc[mask, "Green_Fraction"],
               valid_fracs.loc[mask, "Dead_Fraction"],
               c=STATE_COLORS[state], label=state, alpha=0.55, s=30,
               edgecolors="white", linewidth=0.3)
ax.set_xlabel("Green Fraction (Dry Green / Dry Total)")
ax.set_ylabel("Dead Fraction (Dry Dead / Dry Total)")
ax.set_title("Biomass Composition: Green vs. Dead Fraction", fontweight="bold")
ax.legend(title="State", frameon=True)
# Add diagonal reference
ax.plot([0, 1], [1, 0], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
save(fig, "fig14_green_vs_dead_fraction")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — Species Heatmap: Mean Dry Total by Species × State
# ══════════════════════════════════════════════════════════════════════════════
top10_species = pivot["Species"].value_counts().head(10).index.tolist()
sub_heat = pivot[pivot["Species"].isin(top10_species)]
heat_data = sub_heat.groupby(["Species", "State"])["Dry_Total_g"].mean().unstack(fill_value=np.nan)
heat_data = heat_data.reindex(index=top10_species, columns=["NSW", "Tas", "Vic", "WA"])
heat_data.index = [s.replace("_", " ") for s in heat_data.index]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=1, linecolor="white", ax=ax,
            cbar_kws={"label": "Mean Dry Total (g)", "shrink": 0.8},
            mask=heat_data.isna())
ax.set_title("Mean Dry Total Biomass: Species × State", fontweight="bold")
ax.set_xlabel("State")
ax.set_ylabel("Pasture Species")
fig.tight_layout()
save(fig, "fig15_species_state_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — Log-scale Target Distributions (to show skewness)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
log_targets = ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
log_labels  = ["Dry Total (g)", "GDM (g)", "Dry Green (g)"]
for i, (tgt, lbl) in enumerate(zip(log_targets, log_labels)):
    vals = pivot[tgt].replace(0, np.nan).dropna()
    log_vals = np.log1p(vals)
    axes[i].hist(log_vals, bins=35, color=TARGET_COLORS[tgt], alpha=0.7,
                 edgecolor="white", linewidth=0.5, density=True)
    sns.kdeplot(log_vals, ax=axes[i], color="black", linewidth=1.5)
    skew = vals.skew()
    kurt = vals.kurtosis()
    axes[i].text(0.95, 0.92, f"Skew={skew:.2f}\nKurt={kurt:.2f}",
                 transform=axes[i].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    axes[i].set_xlabel(f"log(1 + {lbl})")
    axes[i].set_title(f"Log-Transformed {lbl}", fontweight="bold")
    if i == 0:
        axes[i].set_ylabel("Density")
fig.suptitle("Log-Transformed Target Distributions (Skewness Analysis)", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig16_log_target_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17 — CDF of Dry_Total by State (Empirical CDF)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for state in ["NSW", "Tas", "Vic", "WA"]:
    vals = pivot.loc[pivot["State"] == state, "Dry_Total_g"].sort_values()
    cdf = np.arange(1, len(vals)+1) / len(vals)
    ax.step(vals, cdf, where="post", color=STATE_COLORS[state], label=state, linewidth=2)
ax.set_xlabel("Dry Total Biomass (g)")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Empirical CDF of Dry Total Biomass by State", fontweight="bold")
ax.legend(title="State", frameon=True)
ax.set_xlim(0, None)
fig.tight_layout()
save(fig, "fig17_ecdf_dry_total")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 18 — Image Mean Color Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("  Computing image color statistics (may take a moment) …")
color_stats = []
train_dir = os.path.join(os.path.dirname(__file__), "train")
for _, row in pivot.iterrows():
    img_name = os.path.basename(row["image_path"])
    img_path = os.path.join(train_dir, img_name)
    if os.path.exists(img_path):
        img = np.array(Image.open(img_path))
        r_mean, g_mean, b_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        brightness = img.mean()
        greenness = g_mean / (r_mean + g_mean + b_mean + 1e-8)
        color_stats.append({
            "image_path": row["image_path"],
            "R_mean": r_mean, "G_mean": g_mean, "B_mean": b_mean,
            "Brightness": brightness, "Greenness": greenness,
            "State": row["State"],
            "Dry_Total_g": row["Dry_Total_g"],
            "GDM_g": row["GDM_g"],
            "Pre_GSHH_NDVI": row["Pre_GSHH_NDVI"],
        })

color_df = pd.DataFrame(color_stats)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 18a: Greenness vs GDM
for state in ["NSW", "Tas", "Vic", "WA"]:
    mask = color_df["State"] == state
    axes[0].scatter(color_df.loc[mask, "Greenness"], color_df.loc[mask, "GDM_g"],
                    c=STATE_COLORS[state], label=state, alpha=0.55, s=25,
                    edgecolors="white", linewidth=0.3)
r_val = color_df["Greenness"].corr(color_df["GDM_g"])
axes[0].text(0.05, 0.95, f"r = {r_val:.3f}", transform=axes[0].transAxes, va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
axes[0].set_xlabel("Image Greenness Index")
axes[0].set_ylabel("GDM (g)")
axes[0].set_title("(a) Image Greenness vs. GDM", fontweight="bold")
axes[0].legend(title="State", frameon=True, markerscale=1.5)

# 18b: Brightness vs Dry Total
for state in ["NSW", "Tas", "Vic", "WA"]:
    mask = color_df["State"] == state
    axes[1].scatter(color_df.loc[mask, "Brightness"], color_df.loc[mask, "Dry_Total_g"],
                    c=STATE_COLORS[state], alpha=0.55, s=25,
                    edgecolors="white", linewidth=0.3)
r_val = color_df["Brightness"].corr(color_df["Dry_Total_g"])
axes[1].text(0.05, 0.95, f"r = {r_val:.3f}", transform=axes[1].transAxes, va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
axes[1].set_xlabel("Mean Image Brightness")
axes[1].set_ylabel("Dry Total (g)")
axes[1].set_title("(b) Image Brightness vs. Dry Total", fontweight="bold")

# 18c: Greenness vs NDVI
for state in ["NSW", "Tas", "Vic", "WA"]:
    mask = color_df["State"] == state
    axes[2].scatter(color_df.loc[mask, "Greenness"], color_df.loc[mask, "Pre_GSHH_NDVI"],
                    c=STATE_COLORS[state], alpha=0.55, s=25,
                    edgecolors="white", linewidth=0.3)
r_val = color_df["Greenness"].corr(color_df["Pre_GSHH_NDVI"])
axes[2].text(0.05, 0.95, f"r = {r_val:.3f}", transform=axes[2].transAxes, va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
axes[2].set_xlabel("Image Greenness Index")
axes[2].set_ylabel("Pre-Grazing NDVI")
axes[2].set_title("(c) Image Greenness vs. NDVI", fontweight="bold")

fig.suptitle("Image-Derived Color Features and Their Relationship to Biomass", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig18_image_color_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 19 — Sample Images Grid (4 states × diverse biomass)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for col, state in enumerate(["NSW", "Tas", "Vic", "WA"]):
    sub_s = pivot[pivot["State"] == state].sort_values("Dry_Total_g")
    # Low biomass
    low_row = sub_s.iloc[len(sub_s)//5]
    # High biomass
    high_row = sub_s.iloc[-len(sub_s)//5]
    for row_idx, (sample_row, bm_label) in enumerate([(low_row, "Low"), (high_row, "High")]):
        img_name = os.path.basename(sample_row["image_path"])
        img_path = os.path.join(train_dir, img_name)
        img = Image.open(img_path)
        axes[row_idx, col].imshow(img)
        axes[row_idx, col].set_title(
            f"{state} — {bm_label} Biomass\n"
            f"Total={sample_row['Dry_Total_g']:.1f}g, NDVI={sample_row['Pre_GSHH_NDVI']:.2f}",
            fontsize=9, fontweight="bold")
        axes[row_idx, col].axis("off")
axes[0, 0].set_ylabel("Low Biomass", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("High Biomass", fontsize=12, fontweight="bold")
fig.suptitle("Sample Images: Low vs. High Biomass Across States", fontweight="bold", fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "fig19_sample_images_grid")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 20 — Dataset Summary Statistics Table (as figure)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

summary_data = []
for tgt in TARGETS:
    vals = pivot[tgt]
    summary_data.append([
        TARGET_LABELS[tgt],
        f"{vals.count()}",
        f"{vals.mean():.2f}",
        f"{vals.std():.2f}",
        f"{vals.min():.2f}",
        f"{vals.quantile(0.25):.2f}",
        f"{vals.median():.2f}",
        f"{vals.quantile(0.75):.2f}",
        f"{vals.max():.2f}",
        f"{vals.skew():.2f}",
    ])

col_labels = ["Variable", "N", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skew"]
table = ax.table(cellText=summary_data, colLabels=col_labels,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#3C5488")
    table[0, j].set_text_props(color="white", fontweight="bold")    
# Alternate row colors
for i in range(1, len(summary_data)+1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor("#F0F4F8")
        else:
            table[i, j].set_facecolor("#FFFFFF")

ax.set_title("Summary Statistics of Biomass Target Variables", fontweight="bold", fontsize=14, pad=20)
fig.tight_layout()
save(fig, "fig20_summary_statistics_table")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 21 — Joint Distribution: NDVI × Height → Dry Total (2D hex)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(pivot["Pre_GSHH_NDVI"], pivot["Height_Ave_cm"], C=pivot["Dry_Total_g"],
               gridsize=20, cmap="YlOrRd", mincnt=1, reduce_C_function=np.mean)
cb = fig.colorbar(hb, ax=ax, label="Mean Dry Total (g)")
ax.set_xlabel("Pre-Grazing NDVI")
ax.set_ylabel("Average Height (cm)")
ax.set_title("Joint Distribution: NDVI × Height → Mean Biomass", fontweight="bold")
fig.tight_layout()
save(fig, "fig21_ndvi_height_hexbin")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 22 — RGB Channel Distributions by State
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
channels = [("R_mean", "Red", "#E64B35"), ("G_mean", "Green", "#00A087"), ("B_mean", "Blue", "#3C5488")]
for i, (col, ch_name, ch_color) in enumerate(channels):
    for state in ["NSW", "Tas", "Vic", "WA"]:
        vals = color_df.loc[color_df["State"] == state, col]
        sns.kdeplot(vals, ax=axes[i], label=state, color=STATE_COLORS[state],
                    linewidth=2, fill=False)
    axes[i].set_xlabel(f"Mean {ch_name} Channel Value")
    axes[i].set_ylabel("Density")
    axes[i].set_title(f"{ch_name} Channel by State", fontweight="bold")
    if i == 0:
        axes[i].legend(title="State", frameon=True)
fig.suptitle("RGB Channel Distribution by State", fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig22_rgb_channels_by_state")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 23 — Multi-Target Radar/Spider Chart by State
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
categories = ["NDVI", "Height", "Dry Clover", "Dry Dead", "Dry Green"]
state_means = {}
for state in ["NSW", "Tas", "Vic", "WA"]:
    sub_s = pivot[pivot["State"] == state]
    raw = [sub_s["Pre_GSHH_NDVI"].mean(), sub_s["Height_Ave_cm"].mean(),
           sub_s["Dry_Clover_g"].mean(), sub_s["Dry_Dead_g"].mean(), sub_s["Dry_Green_g"].mean()]
    state_means[state] = raw

# Normalize to [0,1] for radar
all_vals = np.array(list(state_means.values()))
mins = all_vals.min(axis=0)
maxs = all_vals.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for state in ["NSW", "Tas", "Vic", "WA"]:
    normed = (np.array(state_means[state]) - mins) / ranges
    values = normed.tolist() + normed[:1].tolist()
    ax.plot(angles, values, "o-", linewidth=2, color=STATE_COLORS[state], label=state, markersize=5)
    ax.fill(angles, values, color=STATE_COLORS[state], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title("Normalized Feature Profile by State", fontweight="bold", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, title="State")
fig.tight_layout()
save(fig, "fig23_radar_chart_by_state")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Done! {23} figures saved to {FIGDIR}")
print(f"Each figure saved as PNG (300 dpi) and SVG.")
print(f"{'='*60}")
