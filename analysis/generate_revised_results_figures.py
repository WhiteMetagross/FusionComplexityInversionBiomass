"""Generate corrected paper figures from verified raw scale evaluation files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OUT_ROOT = WORKSPACE_ROOT / "newPaper" / "img"
PNG_DIR = OUT_ROOT / "png"
SVG_DIR = OUT_ROOT / "svg"

DEEP_AGG_PATH = REPO_ROOT / "output" / "checkpoint_integrity_2026_09_14" / "aggregate_metrics.csv"
DEEP_FOLD_PATH = REPO_ROOT / "output" / "checkpoint_integrity_2026_09_14" / "fold_metrics.csv"
CLASSICAL_AGG_PATH = REPO_ROOT / "output" / "classical_baselines_2026_09_15" / "aggregate_metrics.csv"
CLASSICAL_FOLD_PATH = REPO_ROOT / "output" / "classical_baselines_2026_09_15" / "fold_metrics.csv"


C_LOCAL = "#43A047"
C_LOCAL_LIGHT = "#66BB6A"
C_ATTN = "#8E24AA"
C_ATTN_LIGHT = "#BA68C8"
C_SSM_BIDIR = "#E64A19"
C_SSM_FULL = "#C62828"
C_NONE = "#78909C"
C_META = "#F9A825"
C_VMAMBA = "#6D4C41"
C_EFFNET = "#546E7A"
C_DINOV2 = "#00838F"
C_DINOV3 = "#1565C0"
C_CLASSICAL = "#607D8B"
C_HANDCRAFTED = "#00838F"
C_CLASSICAL_MULTIMODAL = "#E64A19"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.55,
    }
)


@dataclass(frozen=True)
class Result:
    model_id: str
    label: str
    mean: float
    std: float
    pooled: float
    folds: tuple[float, ...]
    targets: tuple[float, ...]
    color: str


def load_results() -> dict[str, Result]:
    deep_agg = pd.read_csv(DEEP_AGG_PATH)
    deep_fold = pd.read_csv(DEEP_FOLD_PATH)
    classical_agg = pd.read_csv(CLASSICAL_AGG_PATH)
    classical_fold = pd.read_csv(CLASSICAL_FOLD_PATH)

    labels = {
        "B4": "B4. DINOv2-L + GDWC",
        "B5": "B5. DINOv3-L + GDWC",
        "B7": "B7. CVGA + metadata",
        "B8": "B8. BidirMamba + metadata",
        "B9": "B9. VMamba-Base",
        "E1": "E1. BidirMamba",
        "E2": "E2. CVGA",
        "E3": "E3. GDWC + metadata",
        "E4": "E4. Identity",
        "E5": "E5. FullMamba",
        "E6": "E6. GDWC 1×",
        "E7": "E7. GDWC 4×",
        "E8": "E8. Identity + metadata",
        "C1": "C1. Metadata Ridge",
        "C2": "C2. Metadata XGBoost",
        "C3": "C3. Handcrafted Ridge",
        "C4": "C4. Handcrafted XGBoost",
        "C5": "C5. Handcrafted + metadata XGBoost",
        "C6": "C6. Frozen DINOv2-B + Ridge",
        "C7": "C7. Frozen DINOv3-L + Ridge",
        "C8": "C8. Frozen DINOv3-L + PCA/XGBoost",
    }
    colors = {
        "B4": C_DINOV2,
        "B5": C_LOCAL,
        "B7": C_ATTN_LIGHT,
        "B8": C_SSM_BIDIR,
        "B9": C_VMAMBA,
        "E1": C_SSM_BIDIR,
        "E2": C_ATTN,
        "E3": C_LOCAL_LIGHT,
        "E4": C_NONE,
        "E5": C_SSM_FULL,
        "E6": "#2E7D32",
        "E7": "#81C784",
        "E8": "#90A4AE",
        "C1": C_META,
        "C2": C_META,
        "C3": C_HANDCRAFTED,
        "C4": C_HANDCRAFTED,
        "C5": C_CLASSICAL_MULTIMODAL,
        "C6": C_DINOV2,
        "C7": C_DINOV3,
        "C8": C_DINOV3,
    }

    results: dict[str, Result] = {}
    for model_id in [key for key in labels if not key.startswith("C")]:
        agg_rows = deep_agg.loc[deep_agg["experiment_id"].eq(model_id)]
        fold_rows = deep_fold.loc[deep_fold["experiment_id"].eq(model_id)].sort_values("fold")
        if len(agg_rows) != 1 or len(fold_rows) != 5:
            raise RuntimeError(f"Incomplete verified result for {model_id}")
        agg = agg_rows.iloc[0]
        folds = tuple(float(value) for value in fold_rows["raw_weighted_r2"])
        mean = float(agg["mean_fold_raw_r2"])
        std = float(agg["std_fold_raw_r2_ddof0"])
        if not np.isclose(mean, np.mean(folds), atol=1e-10):
            raise RuntimeError(f"Mean mismatch for {model_id}")
        if not np.isclose(std, np.std(folds, ddof=0), atol=1e-10):
            raise RuntimeError(f"Standard deviation mismatch for {model_id}")
        target_values = ast.literal_eval(str(agg["pooled_target_raw_r2"]))
        results[model_id] = Result(
            model_id=model_id,
            label=labels[model_id],
            mean=mean,
            std=std,
            pooled=float(agg["pooled_oof_raw_r2"]),
            folds=folds,
            targets=tuple(float(value) for value in target_values),
            color=colors[model_id],
        )

    for model_id in tuple(f"C{index}" for index in range(1, 9)):
        agg_rows = classical_agg.loc[classical_agg["model"].eq(model_id)]
        fold_rows = classical_fold.loc[classical_fold["model"].eq(model_id)].sort_values("fold")
        if len(agg_rows) != 1 or len(fold_rows) != 5:
            raise RuntimeError(f"Incomplete verified classical result for {model_id}")
        agg = agg_rows.iloc[0]
        folds = tuple(float(value) for value in fold_rows["weighted_r2"])
        mean = float(agg["mean_fold_raw_weighted_r2"])
        std = float(agg["std_fold_raw_weighted_r2_ddof0"])
        if not np.isclose(mean, np.mean(folds), atol=1e-10):
            raise RuntimeError(f"Mean mismatch for {model_id}")
        if not np.isclose(std, np.std(folds, ddof=0), atol=1e-10):
            raise RuntimeError(f"Standard deviation mismatch for {model_id}")
        results[model_id] = Result(
            model_id=model_id,
            label=labels[model_id],
            mean=mean,
            std=std,
            pooled=float(agg["pooled_oof_raw_weighted_r2"]),
            folds=folds,
            targets=tuple(
                float(agg[column])
                for column in (
                    "pooled_green_r2",
                    "pooled_dead_r2",
                    "pooled_clover_r2",
                    "pooled_gdm_r2",
                    "pooled_total_r2",
                )
            ),
            color=colors[model_id],
        )
    return results


def save_figure(fig: plt.Figure, name: str) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / f"{name}.png")
    fig.savefig(SVG_DIR / f"{name}.svg")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.svg")


def make_benchmark_overview(results: dict[str, Result]) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.7),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )

    selected = [results[key] for key in ("C2", "C7", "B9", "B4", "B5", "B7", "B8", "E2")]
    selected.sort(key=lambda item: item.mean)
    y = np.arange(len(selected))
    bars = ax1.barh(
        y,
        [item.mean for item in selected],
        color=[item.color for item in selected],
        edgecolor="white",
        linewidth=0.8,
        height=0.64,
        zorder=3,
    )
    ax1.errorbar(
        [item.mean for item in selected],
        y,
        xerr=[item.std for item in selected],
        fmt="none",
        ecolor="#37474F",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=4,
    )
    for bar, item in zip(bars, selected):
        ax1.text(
            min(item.mean + item.std + 0.008, 0.895),
            bar.get_y() + bar.get_height() / 2,
            f"{item.mean:.3f}",
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold" if item.model_id == "E2" else "normal",
            color="#263238",
        )
    ax1.set_yticks(y, [item.label for item in selected])
    ax1.set_xlim(0.0, 0.92)
    ax1.set_xlabel("Mean weighted raw scale R²", fontweight="bold")
    ax1.set_title("(a) Verified Single Model Comparison", fontweight="bold", pad=10)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.grid(axis="x", zorder=0)

    classical_ids = tuple(f"C{index}" for index in range(1, 9))
    classical_y = {
        "C1": 10.0,
        "C2": 9.0,
        "C3": 7.4,
        "C4": 6.4,
        "C5": 4.8,
        "C6": 3.2,
        "C7": 2.2,
        "C8": 1.2,
    }
    family = {
        "C1": ("Metadata only", C_META),
        "C2": ("Metadata only", C_META),
        "C3": ("Handcrafted image", C_HANDCRAFTED),
        "C4": ("Handcrafted image", C_HANDCRAFTED),
        "C5": ("Handcrafted + metadata", C_CLASSICAL_MULTIMODAL),
        "C6": ("Frozen foundation features", C_DINOV3),
        "C7": ("Frozen foundation features", C_DINOV3),
        "C8": ("Frozen foundation features", C_DINOV3),
    }
    learner = {
        "C1": "Ridge",
        "C2": "XGBoost",
        "C3": "Ridge",
        "C4": "XGBoost",
        "C5": "XGBoost",
        "C6": "Ridge",
        "C7": "Ridge",
        "C8": "XGBoost",
    }
    marker = {"Ridge": "o", "XGBoost": "s"}
    short_label = {
        "C1": "C1. Metadata, Ridge",
        "C2": "C2. Metadata, XGBoost",
        "C3": "C3. Handcrafted, Ridge",
        "C4": "C4. Handcrafted, XGBoost",
        "C5": "C5. Handcrafted + metadata, XGBoost",
        "C6": "C6. Frozen DINOv2-B, Ridge",
        "C7": "C7. Frozen DINOv3-L, Ridge",
        "C8": "C8. Frozen DINOv3-L, PCA + XGBoost",
    }

    group_spans = (
        (8.55, 10.45, C_META, "METADATA ONLY"),
        (5.95, 7.85, C_HANDCRAFTED, "HANDCRAFTED IMAGE"),
        (4.35, 5.25, C_CLASSICAL_MULTIMODAL, "HANDCRAFTED + METADATA"),
        (0.75, 3.65, C_DINOV3, "FROZEN FOUNDATION FEATURES"),
    )
    for low, high, color, group_label in group_spans:
        ax2.axhspan(low, high, color=color, alpha=0.045, zorder=0)
        ax2.text(
            0.252,
            low + 0.10,
            group_label,
            ha="left",
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color=color,
            alpha=0.82,
            zorder=1,
        )

    for model_id in classical_ids:
        item = results[model_id]
        ypos = classical_y[model_id]
        color = family[model_id][1]
        ax2.errorbar(
            item.mean,
            ypos,
            xerr=item.std,
            fmt=marker[learner[model_id]],
            markersize=6.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        ax2.text(
            item.mean + item.std + 0.008,
            ypos,
            f"{item.mean:.3f}",
            va="center",
            fontsize=7.1,
            color="#263238",
        )

    delta_pairs = (
        ("C1", "C2", "+0.037"),
        ("C3", "C4", "+0.074"),
        ("C4", "C5", "+0.191"),
        ("C6", "C7", "+0.063"),
        ("C7", "C8", "−0.085"),
    )
    for first_id, second_id, delta_text in delta_pairs:
        first = results[first_id]
        second = results[second_id]
        y_first = classical_y[first_id]
        y_second = classical_y[second_id]
        ax2.plot(
            [first.mean, second.mean],
            [y_first, y_second],
            color="#90A4AE",
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            zorder=2,
        )
        ax2.text(
            (first.mean + second.mean) / 2,
            (y_first + y_second) / 2,
            delta_text,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="#455A64",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            zorder=5,
        )

    ax2.axvline(results["E2"].mean, color=C_ATTN, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax2.text(
        results["E2"].mean - 0.004,
        0.72,
        f"E2 reference {results['E2'].mean:.3f}",
        ha="right",
        va="top",
        fontsize=7.1,
        fontweight="bold",
        color=C_ATTN,
    )
    ax2.set_yticks(
        [classical_y[model_id] for model_id in classical_ids],
        [short_label[model_id] for model_id in classical_ids],
    )
    ax2.set_xlim(0.24, 0.88)
    ax2.set_ylim(0.65, 10.90)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean weighted raw scale R²", fontweight="bold")
    ax2.set_title("(b) Classical and Frozen Feature Baselines", fontweight="bold", pad=10)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.grid(axis="x", zorder=0)

    learner_handles = [
        plt.Line2D([], [], color="#455A64", marker="o", linestyle="none", markersize=5.5, label="Ridge"),
        plt.Line2D([], [], color="#455A64", marker="s", linestyle="none", markersize=5.5, label="XGBoost"),
    ]
    ax2.legend(
        handles=learner_handles,
        title="Learner",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.005),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(w_pad=2.8)
    save_figure(fig, "fig03_benchmark_overview")


def make_fusion_metadata(results: dict[str, Result]) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(14.6, 4.9),
        gridspec_kw={"width_ratios": [1.25, 0.90, 1.0]},
    )

    fusion_ids = ("E2", "B5", "E6", "E1", "E4", "E7", "E5")
    fusion_results = sorted((results[key] for key in fusion_ids), key=lambda item: item.mean, reverse=True)
    y = np.arange(len(fusion_results))
    for index, item in enumerate(fusion_results):
        if index % 2 == 0:
            ax1.axhspan(index - 0.42, index + 0.42, color="#ECEFF1", alpha=0.52, zorder=0)
        ax1.errorbar(
            item.mean,
            index,
            xerr=item.std,
            fmt="o",
            markersize=7,
            color=item.color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=item.color,
            elinewidth=1.2,
            capsize=3,
            zorder=4,
        )
        ax1.text(
            item.mean + item.std + 0.004,
            index,
            f"{item.mean:.3f}",
            va="center",
            fontsize=7.4,
            fontweight="bold" if item.model_id == "E2" else "normal",
            color="#263238",
        )
    ax1.set_yticks(y, [item.label for item in fusion_results])
    ax1.invert_yaxis()
    ax1.set_xlim(0.72, 0.90)
    ax1.set_xlabel("Mean weighted raw scale R²", fontweight="bold")
    ax1.set_title("(a) Fusion Family\nDINOv3-L, no metadata", fontweight="bold", pad=8)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.grid(axis="x", zorder=0)
    ax1.text(
        0.0,
        -0.16,
        "Points show mean; whiskers show SD across five outer folds.",
        transform=ax1.transAxes,
        fontsize=6.8,
        color="#607D8B",
        style="italic",
    )

    depth_ids = {0: "E4", 1: "E6", 2: "B5", 4: "E7"}
    depths = list(depth_ids)
    means = [results[depth_ids[depth]].mean for depth in depths]
    stds = [results[depth_ids[depth]].std for depth in depths]
    ax2.errorbar(
        depths,
        means,
        yerr=stds,
        fmt="o",
        color=C_LOCAL,
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        ecolor=C_LOCAL,
        elinewidth=1.25,
        capsize=4,
        capthick=1.1,
        zorder=4,
    )
    best_index = int(np.argmax(means))
    ax2.scatter(
        depths[best_index],
        means[best_index],
        s=104,
        facecolor=C_LOCAL,
        edgecolor="#1B5E20",
        linewidth=1.3,
        zorder=5,
    )
    identity_mean = means[0]
    label_offsets = {
        0: (10, "left"),
        1: (10, "left"),
        2: (10, "left"),
        4: (-10, "right"),
    }
    for depth, mean in zip(depths, means):
        delta = mean - identity_mean
        label = f"{mean:.3f}" if depth == 0 else f"{mean:.3f}\n({delta:+.3f})"
        offset, alignment = label_offsets[depth]
        ax2.annotate(
            label,
            (depth, mean),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=7.2,
            fontweight="bold" if depth == 2 else "normal",
            color="#263238",
        )
    ax2.set_xticks(depths, ["0\nIdentity", "1", "2", "4"])
    ax2.set_xlim(-0.35, 4.35)
    ax2.set_ylim(0.75, 0.89)
    ax2.set_xlabel("Number of GDWC blocks", fontweight="bold")
    ax2.set_ylabel("Mean weighted raw scale R²", fontweight="bold")
    ax2.set_title("(b) Fusion Depth\nMean ± fold SD", fontweight="bold", pad=8)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax2.grid(axis="y", zorder=0)

    metadata_pairs = [
        ("GDWC", "B5", "E3", C_LOCAL),
        ("CVGA", "E2", "B7", C_ATTN),
        ("BidirMamba", "E1", "B8", C_SSM_BIDIR),
        ("Identity", "E4", "E8", C_NONE),
    ]
    pair_rows = []
    for name, no_meta_id, meta_id, color in metadata_pairs:
        no_meta = results[no_meta_id].mean
        with_meta = results[meta_id].mean
        pair_rows.append((name, no_meta, with_meta, with_meta - no_meta, color))
    pair_rows.sort(key=lambda row: row[3], reverse=True)
    pair_y = np.arange(len(pair_rows))
    for index, (name, no_meta, with_meta, delta, color) in enumerate(pair_rows):
        ax3.barh(
            index,
            delta,
            left=0,
            height=0.52,
            color=color,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        text_x = delta + (0.00035 if delta >= 0 else -0.00035)
        ax3.text(
            text_x,
            index,
            f"{delta:+.3f}",
            ha="left" if delta >= 0 else "right",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#263238",
        )
        ax3.text(
            0.0132,
            index + 0.33,
            f"{no_meta:.3f} → {with_meta:.3f}",
            ha="right",
            va="center",
            fontsize=6.5,
            color="#607D8B",
        )
    ax3.axvline(0, color="#263238", linewidth=1.0, zorder=2)
    ax3.set_yticks(pair_y, [row[0] for row in pair_rows])
    ax3.invert_yaxis()
    ax3.set_xlim(-0.0085, 0.0140)
    ax3.set_xlabel("Change from adding metadata, ΔR²", fontweight="bold")
    ax3.set_title("(c) Metadata Effect by Fusion", fontweight="bold", pad=8)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%+.3f"))
    ax3.grid(axis="x", zorder=0)
    ax3.text(
        0.99,
        -0.16,
        "Labels at right: no metadata → 23 features.",
        transform=ax3.transAxes,
        ha="right",
        fontsize=6.8,
        color="#607D8B",
        style="italic",
    )

    fig.suptitle("Fusion and Metadata Ablations", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=2.3)
    save_figure(fig, "fig04_fusion_metadata")


def make_fold_stability(results: dict[str, Result]) -> None:
    selected_ids = ("E2", "B8", "B7", "B5", "B4")
    selected = [results[key] for key in selected_ids]
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.0),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
    )

    fold_frame = pd.DataFrame(
        [item.folds for item in selected],
        index=[item.label for item in selected],
        columns=[f"Fold {index}" for index in range(5)],
    )
    sns.heatmap(
        fold_frame,
        annot=True,
        fmt=".3f",
        cmap=sns.color_palette("YlGn", as_cmap=True),
        vmin=0.74,
        vmax=0.91,
        linewidths=1.1,
        linecolor="white",
        cbar_kws={"label": "Weighted raw scale R²", "shrink": 0.82},
        annot_kws={"fontsize": 7.7},
        ax=ax1,
    )
    ax1.add_patch(
        plt.Rectangle(
            (4, 0),
            1,
            len(selected),
            fill=False,
            edgecolor="#C62828",
            linewidth=1.8,
            zorder=5,
        )
    )
    ax1.set_xlabel("Outer fold", fontweight="bold")
    ax1.set_ylabel("")
    ax1.set_title("(a) Fold by Model Scores", fontweight="bold", pad=30)
    ax1.tick_params(axis="x", rotation=0)
    ax1.tick_params(axis="y", rotation=0)
    ax1.get_xticklabels()[4].set_color("#C62828")
    ax1.get_xticklabels()[4].set_fontweight("bold")
    ax1.text(
        0.5,
        1.015,
        "Fold 4 is lowest for all models",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#C62828",
        fontweight="bold",
    )

    ordered = selected
    y = np.arange(len(ordered))
    for index, item in enumerate(ordered):
        values = np.asarray(item.folds)
        if index % 2 == 0:
            ax2.axhspan(index - 0.40, index + 0.40, color="#ECEFF1", alpha=0.52, zorder=0)
        ax2.hlines(
            index,
            values.min(),
            values.max(),
            color=item.color,
            linewidth=2.2,
            alpha=0.30,
            zorder=2,
        )
        ax2.vlines(
            [values.min(), values.max()],
            index - 0.08,
            index + 0.08,
            color=item.color,
            linewidth=1.0,
            alpha=0.65,
        )
        ax2.errorbar(
            item.mean,
            index,
            xerr=item.std,
            fmt="D",
            markersize=7.2,
            color=item.color,
            markeredgecolor="#263238",
            markeredgewidth=0.9,
            ecolor=item.color,
            elinewidth=1.5,
            capsize=3,
            capthick=1.1,
            zorder=4,
        )
        ax2.text(
            0.927,
            index,
            f"{item.mean:.3f} ± {item.std:.3f}",
            ha="right",
            va="center",
            fontsize=7.1,
            color="#455A64",
        )
    ax2.set_yticks(y, [item.model_id for item in ordered])
    ax2.tick_params(axis="y", pad=1.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0.70, 0.935)
    ax2.set_xlabel("Weighted raw scale R²", fontweight="bold")
    ax2.set_title("(b) Mean, Fold SD, and Range", fontweight="bold", pad=10)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.04))
    ax2.grid(axis="x", zorder=0)
    fig.suptitle("Cross Validation Stability", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=2.8)
    save_figure(fig, "fig05_fold_stability")


def make_target_performance(results: dict[str, Result]) -> None:
    selected_ids = ("E2", "B5", "B8", "B7", "B4", "C7", "C2")
    target_names = ["Green", "Dead", "Clover", "GDM", "Total"]
    frame = pd.DataFrame(
        [list(results[key].targets) + [results[key].pooled] for key in selected_ids],
        index=[results[key].label for key in selected_ids],
        columns=target_names + ["Overall"],
    )
    fig, ax = plt.subplots(figsize=(10.7, 4.7))
    sns.heatmap(
        frame,
        annot=True,
        fmt=".3f",
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        vmin=0.25,
        vmax=0.90,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "Pooled out of fold R²", "shrink": 0.82},
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.axvline(5, color="#263238", linewidth=1.6)
    ax.set_xlabel("Biomass target", fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(
        "Target Level Performance\nDead Biomass Remains the Main Error Source",
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, "fig06_target_performance")


def main() -> None:
    results = load_results()
    make_benchmark_overview(results)
    make_fusion_metadata(results)
    make_fold_stability(results)
    make_target_performance(results)


if __name__ == "__main__":
    main()
