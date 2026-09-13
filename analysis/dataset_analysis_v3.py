# Author: Mridankan Mandal
# Paper: arXiv:2603.07819

"""
CSIRO BioMass Dataset: Consolidated Research Analysis (v3)
==========================================================
Informed by Liao et al. (arXiv:2510.22916).
10 figures total (merged panels), saved as PNG and SVG.
"""

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = str(REPO_ROOT / "csiro-biomass")
PROJECT = str(REPO_ROOT)
FIGDIR_PNG = os.path.join(PROJECT, "output", "analysis", "png")
FIGDIR_SVG = os.path.join(PROJECT, "output", "analysis", "svg")
TRAIN_DIR  = os.path.join(BASE, "train")
os.makedirs(FIGDIR_PNG, exist_ok=True)
os.makedirs(FIGDIR_SVG, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

STATE_COLORS  = {"NSW": "#E64B35", "Tas": "#4DBBD5", "Vic": "#00A087", "WA": "#F39B7F"}
TARGET_COLORS = {"Dry_Clover_g":"#3C5488","Dry_Dead_g":"#B09C85","Dry_Green_g":"#00A087",
                 "Dry_Total_g":"#E64B35","GDM_g":"#F39B7F"}
TARGET_LABELS = {"Dry_Clover_g":"Dry Clover (g)","Dry_Dead_g":"Dry Dead (g)",
                 "Dry_Green_g":"Dry Green (g)","Dry_Total_g":"Dry Total (g)","GDM_g":"GDM (g)"}
COMP_COLORS   = {"Dry_Green_g":"#00A087","Dry_Dead_g":"#B09C85","Dry_Clover_g":"#3C5488"}

# ── Species abbreviation ────────────────────────────────────────────────────
_SPECIES_MAP = {
    "Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed": "Mixed Pasture A\u2020",
    "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass": "Mixed Pasture B\u2020",
}
_SPECIES_LEGEND = ("\u2020 A: Phalaris, BarleyGrass, SilverGrass, SpearGrass, Clover, Capeweed\n"
                   "\u2020 B: Phalaris, Clover, Ryegrass, Barleygrass, Bromegrass")

def short_species(name):
    """Abbreviate long species names for readability."""
    if name in _SPECIES_MAP:
        return _SPECIES_MAP[name]
    return name.replace("_", " ")

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR_PNG, f"{name}.png"))
    fig.savefig(os.path.join(FIGDIR_SVG, f"{name}.svg"))
    plt.close(fig)
    print(f"  [OK] {name}")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "train.csv"))
df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
df["Month"] = df["Sampling_Date"].dt.month
TARGETS = ["Dry_Clover_g","Dry_Dead_g","Dry_Green_g","Dry_Total_g","GDM_g"]

pivot = df.pivot_table(
    index=["image_path","Sampling_Date","State","Species","Pre_GSHH_NDVI","Height_Ave_cm","Month"],
    columns="target_name", values="target"
).reset_index()

pivot["Green_Fraction"]  = pivot["Dry_Green_g"] / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["Dead_Fraction"]   = pivot["Dry_Dead_g"]  / pivot["Dry_Total_g"].replace(0, np.nan)
pivot["Clover_Fraction"] = pivot["Dry_Clover_g"]/ pivot["Dry_Total_g"].replace(0, np.nan)

print("Computing image color features ...")
color_records = []
for _, row in pivot.iterrows():
    img_path = os.path.join(TRAIN_DIR, os.path.basename(row["image_path"]))
    if os.path.exists(img_path):
        img = np.array(Image.open(img_path))
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        R, G, B = r.mean()/255, g.mean()/255, b.mean()/255
        color_records.append({
            "image_path": row["image_path"],
            "Brightness": img.mean(),
            "Greenness": g.mean()/(r.mean()+g.mean()+b.mean()+1e-8),
            "ExG": 2*G - R - B,
        })
color_df = pd.DataFrame(color_records)
pivot = pivot.merge(color_df, on="image_path", how="left")

month_order = sorted(pivot["Month"].unique())
month_labels = [pd.Timestamp(2015, m, 1).strftime("%b") for m in month_order]

comps = ["Dry_Green_g","Dry_Dead_g","Dry_Clover_g"]
c_labels = ["Green","Dead","Clover"]
c_colors = [COMP_COLORS[c] for c in comps]
so = ["NSW","Tas","Vic","WA"]

print(f"Dataset: {len(pivot)} images\nGenerating consolidated figures ...\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Dataset Overview (3-panel: Donut, Heatmap, Species Bar)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.4], wspace=0.35)

ax1 = fig.add_subplot(gs[0])
sc = pivot["State"].value_counts().reindex(["Tas","Vic","NSW","WA"])
ax1.pie(sc.values, labels=sc.index, colors=[STATE_COLORS[s] for s in sc.index],
        autopct=lambda p: f"{int(p*sum(sc)/100)}\n({p:.0f}%)", startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
ax1.set_title("(a) Samples by State", fontweight="bold")

ax2 = fig.add_subplot(gs[1])
temp = pivot.groupby(["State","Month"]).size().unstack(fill_value=0)
temp = temp.reindex(index=["NSW","Tas","Vic","WA"], columns=month_order, fill_value=0)
temp.columns = month_labels
sns.heatmap(temp, annot=True, fmt="d", cmap="YlOrRd", linewidths=1.5, linecolor="white",
            ax=ax2, cbar_kws={"shrink":0.7,"label":"Count"},
            annot_kws={"fontweight":"bold","fontsize":10})
ax2.set_title("(b) Sampling Calendar", fontweight="bold")
ax2.set_xlabel("Month (2015)"); ax2.set_ylabel("")

ax3 = fig.add_subplot(gs[2])
sp = pivot["Species"].value_counts().head(10)
sp_labels = [short_species(s) for s in sp.index]
bars = ax3.barh(range(len(sp)), sp.values, color=sns.color_palette("Set2", 10),
                edgecolor="white", height=0.7)
ax3.set_yticks(range(len(sp)))
ax3.set_yticklabels(sp_labels, fontsize=9)
for bar, v in zip(bars, sp.values):
    ax3.text(v+1, bar.get_y()+bar.get_height()/2, str(v),
             ha="left", va="center", fontsize=9, fontweight="bold")
ax3.set_xlabel("Samples"); ax3.set_title("(c) Top-10 Pasture Species", fontweight="bold")
ax3.invert_yaxis(); ax3.set_xlim(0, sp.max()*1.18)
# Add abbreviation footnote
if any(s in sp.index for s in _SPECIES_MAP):
    ax3.text(0.02, -0.12, _SPECIES_LEGEND, transform=ax3.transAxes,
             fontsize=7, style="italic", color="gray", va="top")
save(fig, "fig01_dataset_overview")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Target Distributions: Raw vs. Log-Transformed
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for i, tgt in enumerate(TARGETS):
    vals = pivot[tgt].dropna()
    log_vals = np.log1p(vals)
    for row_idx, (data, xlabel, ylabel_prefix) in enumerate([
        (vals, "Biomass (g)", "Density (raw)"),
        (log_vals, "log(1 + Biomass)", "Density (log)")
    ]):
        ax = axes[row_idx, i]
        ax.hist(data, bins=30, color=TARGET_COLORS[tgt], alpha=0.75,
                edgecolor="white", density=True)
        sns.kdeplot(data, ax=ax, color="black", linewidth=1.8)
        ax.axvline(data.median(), color="#E64B35", ls="--", lw=1.2)
        skew = data.skew()
        ax.text(0.95, 0.90, f"skew={skew:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
        if row_idx == 0:
            ax.set_title(TARGET_LABELS[tgt], fontweight="bold", fontsize=11)
        ax.set_xlabel(xlabel)
        if i == 0: ax.set_ylabel(ylabel_prefix)
fig.suptitle("Target Distributions: Raw (top) vs. Log-Transformed (bottom)\n"
             "Evaluation uses log(1+y) transform",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig02_target_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Correlation Heatmap (Sensor Metadata, Image-Derived, and Targets)
# ══════════════════════════════════════════════════════════════════════════════
corr_cols = ["Pre_GSHH_NDVI","Height_Ave_cm","Greenness","ExG","Brightness"] + TARGETS
corr_labels = ["NDVI","Height","Img Green%","ExG","Brightness",
               "Dry Clover","Dry Dead","Dry Green","Dry Total","GDM"]
corr_mat = pivot[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8.5))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(250,15,s=90,l=45,as_cmap=True),
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=corr_labels, yticklabels=corr_labels,
            linewidths=1.5, linecolor="white",
            cbar_kws={"shrink":0.75,"label":"Pearson r"},
            annot_kws={"size":9.5,"fontweight":"bold"}, ax=ax)
ax.set_title("Feature Correlation Matrix\n(Sensor Metadata, Image-Derived, and Targets)",
             fontweight="bold", fontsize=14, pad=15)
plt.xticks(rotation=40, ha="right")
fig.tight_layout()
save(fig, "fig03_correlation_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: NDVI and Height vs. Biomass (4-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for i, (tgt, lbl) in enumerate([("GDM_g","GDM (g)"), ("Dry_Total_g","Dry Total (g)")]):
    ax = axes[0, i]
    for state in so:
        m = pivot["State"]==state
        ax.scatter(pivot.loc[m,"Pre_GSHH_NDVI"], pivot.loc[m,tgt], c=STATE_COLORS[state],
                   label=state, alpha=0.5, s=28, edgecolors="white", linewidth=0.3)
    x, y = pivot["Pre_GSHH_NDVI"].values, pivot[tgt].values
    z = np.polyfit(x, y, 2); p = np.poly1d(z)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, p(x_fit), "k-", lw=2, alpha=0.6, label="Quadratic fit")
    r, _ = stats.pearsonr(x, y); rho, _ = stats.spearmanr(x, y)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nSpearman \u03c1 = {rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    ax.axvspan(0.78, 0.92, alpha=0.06, color="red")
    ax.set_xlabel("GreenSeeker NDVI"); ax.set_ylabel(lbl)
    ax.set_title(f"({chr(97+i)}) NDVI vs. {lbl}", fontweight="bold")
    if i == 0: ax.legend(title="State", frameon=True, fontsize=8, markerscale=1.5, loc="center left")

for i, (tgt, lbl) in enumerate([("Dry_Total_g","Dry Total (g)"), ("GDM_g","GDM (g)")]):
    ax = axes[1, i]
    sc_ax = ax.scatter(pivot["Height_Ave_cm"], pivot[tgt], c=pivot["Pre_GSHH_NDVI"],
                       cmap="RdYlGn", s=30, alpha=0.65, edgecolors="white", linewidth=0.3)
    fig.colorbar(sc_ax, ax=ax, shrink=0.8, label="NDVI")
    r, _ = stats.pearsonr(pivot["Height_Ave_cm"], pivot[tgt])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    ax.set_xlabel("Falling Plate Height (cm)"); ax.set_ylabel(lbl)
    ax.set_title(f"({chr(99+i)}) Height vs. {lbl} (colored by NDVI)", fontweight="bold")

fig.suptitle("Metadata and Biomass Relationships\n"
             "NDVI saturates at high biomass; Height shows strong linear correlation",
             fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig04_ndvi_height_biomass")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Biomass by State: Composition and Distribution (merged)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# (a) Stacked bar
ax = fig.add_subplot(gs[0, 0])
means = pivot.groupby("State")[comps].mean().reindex(so)
bottom = np.zeros(4)
for comp, lbl, color in zip(comps, c_labels, c_colors):
    vals = means[comp].values
    ax.bar(so, vals, bottom=bottom, label=lbl, color=color, edgecolor="white", width=0.55)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 1.5:
            ax.text(j, b+v/2, f"{v:.1f}g", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottom += vals
for j, state in enumerate(so):
    ax.text(j, bottom[j]+1, f"\u03a3={bottom[j]:.1f}g", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Mean Biomass (g)"); ax.set_xlabel("State")
ax.set_title("(a) Mean Composition by State", fontweight="bold")
ax.legend(frameon=True, loc="upper right")

# (b) Composition scatter
ax = fig.add_subplot(gs[0, 1:])
valid = pivot.dropna(subset=["Green_Fraction","Dead_Fraction"])
sc_ax = ax.scatter(valid["Green_Fraction"], valid["Dead_Fraction"], c=valid["Dry_Total_g"],
                   cmap="YlOrRd", s=35, alpha=0.7, edgecolors="white", linewidth=0.3)
fig.colorbar(sc_ax, ax=ax, shrink=0.85, label="Dry Total (g)")
ax.plot([0,1],[1,0], "k--", alpha=0.25, lw=1)
ax.set_xlabel("Green Fraction"); ax.set_ylabel("Dead Fraction")
ax.set_title("(b) Composition Space (colored by total biomass)", fontweight="bold")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

# (c-e) Violin plots for 3 key targets
key_targets = [("Dry_Total_g","Dry Total (g): weight 0.50"),
               ("GDM_g","GDM (g): weight 0.20"),
               ("Dry_Green_g","Dry Green (g): weight 0.10")]
for i, (tgt, lbl) in enumerate(key_targets):
    ax = fig.add_subplot(gs[1, i])
    sns.violinplot(data=pivot, x="State", y=tgt, order=so,
                   palette=STATE_COLORS, ax=ax, inner="quartile", linewidth=1.2, cut=0)
    sns.stripplot(data=pivot, x="State", y=tgt, order=so,
                  palette=STATE_COLORS, ax=ax, alpha=0.25, size=2.5, jitter=0.2)
    ax.set_title(f"({chr(99+i)}) {lbl}", fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    if i > 0: ax.set_ylabel("")
    else: ax.set_ylabel("Biomass (g)")

fig.suptitle("Biomass by State: Composition and Distribution",
             fontweight="bold", fontsize=14, y=1.01)
save(fig, "fig05_biomass_by_state")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Seasonal Dynamics (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
for state in so:
    g = pivot[pivot["State"]==state].groupby("Month")["Dry_Total_g"].agg(["mean","std"]).reindex(month_order)
    v = g["mean"].notna()
    ax.plot(g.index[v], g["mean"][v], "o-", color=STATE_COLORS[state], label=state, lw=2.2, ms=6)
    ax.fill_between(g.index[v], (g["mean"]-g["std"])[v], (g["mean"]+g["std"])[v],
                    color=STATE_COLORS[state], alpha=0.1)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("Dry Total (g)")
ax.set_title("(a) Total Biomass Trend", fontweight="bold")
ax.legend(title="State", frameon=True, fontsize=9)

ax = axes[1]
monthly = pivot.groupby("Month")[comps].mean().reindex(month_order)
v = monthly.notna().all(axis=1)
ax.stackplot(monthly.index[v], monthly.loc[v,"Dry_Green_g"], monthly.loc[v,"Dry_Dead_g"],
             monthly.loc[v,"Dry_Clover_g"], labels=c_labels, colors=c_colors, alpha=0.75)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("Mean Biomass (g)")
ax.set_title("(b) Seasonal Composition Trend", fontweight="bold")
ax.legend(frameon=True, fontsize=9)

ax = axes[2]
for state in so:
    g = pivot[pivot["State"]==state].groupby("Month")["Pre_GSHH_NDVI"].agg(["mean","std"]).reindex(month_order)
    v = g["mean"].notna()
    ax.plot(g.index[v], g["mean"][v], "s-", color=STATE_COLORS[state], label=state, lw=2.2, ms=6)
    ax.fill_between(g.index[v], (g["mean"]-g["std"])[v], (g["mean"]+g["std"])[v],
                    color=STATE_COLORS[state], alpha=0.1)
ax.set_xticks(month_order); ax.set_xticklabels(month_labels)
ax.set_xlabel("Month"); ax.set_ylabel("NDVI")
ax.set_title("(c) NDVI Seasonal Trend", fontweight="bold")
ax.legend(title="State", frameon=True, fontsize=9, loc="lower right")

fig.suptitle("Seasonal Dynamics: Biomass, Composition, and Vegetation Index",
             fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
save(fig, "fig06_seasonal_dynamics")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Species Analysis (violin and heatmap)
# ══════════════════════════════════════════════════════════════════════════════
top8 = pivot["Species"].value_counts().head(8).index.tolist()
sub = pivot[pivot["Species"].isin(top8)].copy()
sub["Species_short"] = sub["Species"].apply(short_species)
so_short = [short_species(s) for s in top8]

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

palette8 = sns.color_palette("husl", 8)
sns.violinplot(data=sub, x="Species_short", y="Dry_Total_g", order=so_short,
               palette=palette8, inner="quartile", linewidth=1.0, ax=axes[0], cut=0)
axes[0].set_xlabel("Pasture Species"); axes[0].set_ylabel("Dry Total (g)")
axes[0].set_title("(a) Biomass by Species: Top 8", fontweight="bold")
plt.sca(axes[0]); plt.xticks(rotation=25, ha="right")
# Add footnote if abbreviations present
if any(s in top8 for s in _SPECIES_MAP):
    axes[0].text(0.02, -0.18, _SPECIES_LEGEND, transform=axes[0].transAxes,
                 fontsize=7, style="italic", color="gray", va="top")

top10_sp = pivot["Species"].value_counts().head(10).index.tolist()
sh = pivot[pivot["Species"].isin(top10_sp)]
hm = sh.groupby(["Species","State"])["Dry_Total_g"].mean().unstack(fill_value=np.nan)
hm = hm.reindex(index=top10_sp, columns=so)
hm.index = [short_species(s) for s in hm.index]
sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=1.5, linecolor="white",
            ax=axes[1], cbar_kws={"shrink":0.7,"label":"Mean Dry Total (g)"}, mask=hm.isna())
axes[1].set_title("(b) Mean Biomass: Species x State", fontweight="bold")
axes[1].set_xlabel("State"); axes[1].set_ylabel("")

fig.suptitle("Species-Level Biomass Analysis", fontweight="bold", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "fig07_species_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Feature Space Analysis (Image Color and Joint NDVI x Height)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Row 0: Image color features vs biomass
color_feats = [
    ("ExG", "Excess Green Index (ExG)", "GDM_g", "GDM (g)"),
    ("Greenness", "Image Greenness (G/(R+G+B))", "Dry_Green_g", "Dry Green (g)"),
    ("Brightness", "Mean Brightness", "Dry_Dead_g", "Dry Dead (g)"),
]
for i, (xf, xlbl, yf, ylbl) in enumerate(color_feats):
    ax = fig.add_subplot(gs[0, i])
    for state in so:
        m = pivot["State"]==state
        ax.scatter(pivot.loc[m,xf], pivot.loc[m,yf], c=STATE_COLORS[state],
                   label=state, alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
    r, _ = stats.pearsonr(pivot[xf].dropna(), pivot[yf].dropna())
    rho, _ = stats.spearmanr(pivot[xf].dropna(), pivot[yf].dropna())
    ax.text(0.05, 0.95, f"r = {r:.3f}\n\u03c1 = {rho:.3f}", transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="gray"))
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.set_title(f"({chr(97+i)}) {xlbl} vs. {ylbl}", fontweight="bold", fontsize=10)
    if i == 0: ax.legend(title="State", frameon=True, markerscale=1.5, fontsize=8)

# Row 1: Joint NDVI x Height feature space
ax = fig.add_subplot(gs[1, 0:2])
hb = ax.hexbin(pivot["Pre_GSHH_NDVI"], pivot["Height_Ave_cm"], C=pivot["Dry_Total_g"],
               gridsize=18, cmap="YlOrRd", mincnt=1, reduce_C_function=np.mean)
fig.colorbar(hb, ax=ax, label="Mean Dry Total (g)", shrink=0.85)
ax.set_xlabel("NDVI"); ax.set_ylabel("Height (cm)")
ax.set_title("(d) NDVI x Height: Mean Biomass (hexbin)", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
for state in so:
    m = pivot["State"]==state
    if m.sum() > 10:
        sns.kdeplot(x=pivot.loc[m,"Pre_GSHH_NDVI"], y=pivot.loc[m,"Height_Ave_cm"],
                    ax=ax, color=STATE_COLORS[state], levels=3, linewidths=1.5, alpha=0.7)
    ax.scatter(pivot.loc[m,"Pre_GSHH_NDVI"], pivot.loc[m,"Height_Ave_cm"],
               c=STATE_COLORS[state], label=state, alpha=0.3, s=15, edgecolors="none")
ax.set_xlabel("NDVI"); ax.set_ylabel("Height (cm)")
ax.set_title("(e) Feature Space Density by State", fontweight="bold")
ax.legend(title="State", frameon=True, markerscale=2, loc="upper left")

fig.suptitle("Feature Space Analysis: Image-Derived Colors and Sensor Metadata\n"
             "Color features are available at test time; NDVI and Height are not",
             fontweight="bold", fontsize=13, y=1.02)
save(fig, "fig08_feature_space_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Evaluation and Data Quality (6-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
weights = {"Dry_Total_g":0.5, "GDM_g":0.2, "Dry_Clover_g":0.1, "Dry_Dead_g":0.1, "Dry_Green_g":0.1}
w_labels = [TARGET_LABELS[k] for k in weights]; w_values = list(weights.values())
w_colors = [TARGET_COLORS[k] for k in weights]
wedges, texts, autotexts = ax.pie(w_values, labels=w_labels, colors=w_colors,
    autopct=lambda p: f"{p:.0f}%", startangle=140, pctdistance=0.65,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2.5))
for t in autotexts: t.set_fontweight("bold"); t.set_fontsize(11)
ax.set_title("(a) Evaluation Weight Distribution", fontweight="bold")

ax = fig.add_subplot(gs[0, 1])
cv_data = []
for tgt in TARGETS:
    v = pivot[tgt]; lv = np.log1p(v)
    cv_data.append({"Target": TARGET_LABELS[tgt].replace(" (g)",""),
                    "CV_raw": v.std()/(v.mean()+1e-8), "CV_log": lv.std()/(lv.mean()+1e-8),
                    "Zero%": (v==0).mean()*100})
cv_df = pd.DataFrame(cv_data)
x = np.arange(5); w = 0.3
ax.bar(x-w/2, cv_df["CV_raw"], w, label="CV (raw)", color="#3C5488", alpha=0.8, edgecolor="white")
ax.bar(x+w/2, cv_df["CV_log"], w, label="CV (log)", color="#00A087", alpha=0.8, edgecolor="white")
ax2 = ax.twinx()
ax2.plot(x, cv_df["Zero%"], "D-", color="#E64B35", lw=2, ms=8, label="% zeros")
ax2.set_ylabel("% Zero Values", color="#E64B35")
ax.set_xticks(x); ax.set_xticklabels(cv_df["Target"], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Coefficient of Variation"); ax.set_title("(b) Target Difficulty", fontweight="bold")
ax.legend(loc="right", frameon=True, fontsize=8, bbox_to_anchor=(0.95, 0.75))
ax2.legend(loc="upper right", frameon=True, fontsize=8)

ax = fig.add_subplot(gs[0, 2])
bp = ax.boxplot([pivot[t].values for t in TARGETS],
                labels=[TARGET_LABELS[t].replace(" (g)","") for t in TARGETS],
                patch_artist=True, notch=True, widths=0.5,
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
for patch, tgt in zip(bp["boxes"], TARGETS):
    patch.set_facecolor(TARGET_COLORS[tgt]); patch.set_alpha(0.7)
ax.set_ylabel("Biomass (g)"); ax.set_title("(c) Target Range Comparison", fontweight="bold")

for i, (tgt, lbl) in enumerate([("Dry_Total_g","Dry Total"),("GDM_g","GDM"),("Dry_Green_g","Dry Green")]):
    ax = fig.add_subplot(gs[1, i])
    for state in so:
        vals = pivot.loc[pivot["State"]==state, tgt].sort_values()
        cdf = np.arange(1, len(vals)+1)/len(vals)
        ax.step(vals, cdf, where="post", color=STATE_COLORS[state], label=state, lw=2)
    ax.set_xlabel(f"{lbl} (g)"); ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"({chr(100+i)}) ECDF: {lbl}", fontweight="bold")
    if i == 0: ax.legend(title="State", frameon=True)
    ax.set_xlim(0, None)

fig.suptitle("Evaluation and Data Quality Analysis\n"
             "1,162 of 3,187 samples passed QC (36.5%): Weighted R\u00b2 on log(1+y)",
             fontweight="bold", fontsize=14, y=1.02)
save(fig, "fig09_evaluation_quality")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Sample Image Analysis (4 samples merged, 1 per state)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating merged sample panel ...")
selections = [
    ("NSW", "High Green Biomass", "Dry_Green_g", 0.90, False),
    ("Tas", "High Dead Matter",   "Dry_Dead_g",  0.90, False),
    ("Vic", "Balanced Composition","Dry_Total_g", 0.50, True),
    ("WA",  "Low/Sparse Cover",   "Dry_Total_g", 0.20, True),
]

sample_rows = []
for state, desc, sort_col, q, asc in selections:
    sub_s = pivot[pivot["State"]==state].sort_values(sort_col, ascending=asc)
    idx = max(0, min(int(len(sub_s)*(1-q if not asc else q)), len(sub_s)-1))
    sample_rows.append((sub_s.iloc[idx], state, desc))

fig = plt.figure(figsize=(20, 18))
outer_gs = gridspec.GridSpec(4, 1, hspace=0.35)

for row_idx, (row, state, desc) in enumerate(sample_rows):
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, outer_gs[row_idx],
                                                width_ratios=[3, 1.2, 1.2, 0.8], wspace=0.25)
    img_path = os.path.join(TRAIN_DIR, os.path.basename(row["image_path"]))
    img = Image.open(img_path)

    # Image
    ax = fig.add_subplot(inner_gs[0])
    ax.imshow(img); ax.axis("off")
    sp_name = short_species(row["Species"])
    ax.set_title(f"({chr(97+row_idx)}) {desc}: {state}\n{sp_name} | "
                 f"Date: {row['Sampling_Date'].strftime('%Y-%m-%d')}",
                 fontweight="bold", fontsize=11)

    # Composition pie
    ax = fig.add_subplot(inner_gs[1])
    cv = [row["Dry_Green_g"], row["Dry_Dead_g"], row["Dry_Clover_g"]]
    cn = ["Green","Dead","Clover"]; cc = ["#00A087","#B09C85","#3C5488"]
    nz = [(v,n,c) for v,n,c in zip(cv,cn,cc) if v > 0]
    if nz:
        vv,nn,ccc = zip(*nz)
        ax.pie(vv, labels=nn, colors=ccc, autopct=lambda p: f"{p:.0f}%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops={"fontsize":9})
    ax.set_title("Composition", fontweight="bold", fontsize=10)

    # Target bars
    ax = fig.add_subplot(inner_gs[2])
    tvals = [row[t] for t in TARGETS]
    tcols = [TARGET_COLORS[t] for t in TARGETS]
    tshort = ["Clover","Dead","Green","Total","GDM"]
    bars_h = ax.barh(range(5), tvals, color=tcols, edgecolor="white", height=0.55)
    ax.set_yticks(range(5)); ax.set_yticklabels(tshort, fontsize=9)
    for bar, v in zip(bars_h, tvals):
        if v > 0.1:
            ax.text(v+0.3, bar.get_y()+bar.get_height()/2, f"{v:.1f}g",
                    va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Biomass (g)", fontsize=9)
    ax.set_title("Targets", fontweight="bold", fontsize=10)
    ax.invert_yaxis()

    # Metadata card
    ax = fig.add_subplot(inner_gs[3]); ax.axis("off")
    gf = row.get("Green_Fraction", 0)
    df_val = row.get("Dead_Fraction", 0)
    cf = row.get("Clover_Fraction", 0)
    gf = gf if not np.isnan(gf) else 0
    df_val = df_val if not np.isnan(df_val) else 0
    cf = cf if not np.isnan(cf) else 0
    meta = (f"NDVI:    {row['Pre_GSHH_NDVI']:.2f}\n"
            f"Height:  {row['Height_Ave_cm']:.1f} cm\n"
            f"\u2500"*16 + "\n"
            f"Total:   {row['Dry_Total_g']:.1f} g\n"
            f"GDM:     {row['GDM_g']:.1f} g\n"
            f"\u2500"*16 + "\n"
            f"Green%:  {gf*100:.0f}%\n"
            f"Dead%:   {df_val*100:.0f}%\n"
            f"Clover%: {cf*100:.0f}%")
    ax.text(0.1, 0.95, meta, transform=ax.transAxes, va="top", fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F0F4F8", ec="gray", alpha=0.9))
    ax.set_title("Metadata", fontweight="bold", fontsize=10)

fig.suptitle("Sample Image Analysis: One Representative per State",
             fontweight="bold", fontsize=15, y=1.0)
save(fig, "fig10_sample_analysis")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = len([f for f in os.listdir(FIGDIR_PNG) if f.endswith(".png")])
print(f"Done! {total} figures saved.")
print(f"  PNG: {FIGDIR_PNG}")
print(f"  SVG: {FIGDIR_SVG}")
print(f"{'='*60}")
