"""Generate target specific spatial attribution maps from trained checkpoints.

The maps use positive gradient times activation relevance on final backbone
patch tokens. They are sensitivity visualizations, not causal explanations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from models import BiomassModelTimm  # noqa: E402


TARGET_COLUMNS = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_name: str
    image_size: int
    checkpoint: Path
    cv_mean: float
    cv_std: float


@dataclass(frozen=True)
class SampleSpec:
    image_id: str
    target_name: str
    target_index: int
    panel: str


MODEL_SPECS = (
    ModelSpec(
        key="B4",
        label="DINOv2-L + 2× GDWC",
        model_name="vit_large_patch14_dinov2.lvd142m",
        image_size=518,
        checkpoint=REPO_ROOT / "checkpoints" / "B4_DINOv2_L_GatedDWConv_NoMeta" / "fold0_best.pth",
        cv_mean=0.7902171075344087,
        cv_std=0.03576383846383332,
    ),
    ModelSpec(
        key="B5",
        label="DINOv3-L + 2× GDWC",
        model_name="vit_large_patch16_dinov3.lvd1689m",
        image_size=512,
        checkpoint=REPO_ROOT / "checkpoints" / "B5_DINOv3_L_GatedDWConv" / "fold0_best.pth",
        cv_mean=0.8273967456817628,
        cv_std=0.044323736830277384,
    ),
)


SAMPLE_SPECS = (
    SampleSpec("ID157479394", "Dry_Green_g", 0, "(a) High green biomass"),
    SampleSpec("ID1139866256", "Dry_Dead_g", 1, "(b) High dead biomass"),
    SampleSpec("ID1471216911", "Dry_Clover_g", 2, "(c) High clover biomass"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wide_data(data_root: Path, fold_file: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_root / "train.csv")
    raw["image_id"] = raw["sample_id"].str.split("__", n=1).str[0]
    metadata_columns = [
        "image_id",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    wide = raw.pivot_table(
        index=metadata_columns,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    folds = pd.read_csv(fold_file)
    wide = wide.merge(folds, on="image_id", validate="one_to_one")
    if len(wide) != 357:
        raise RuntimeError(f"Expected 357 paired images, found {len(wide)}")
    return wide


def load_original_views(data_root: Path, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    image_path = data_root / "train" / f"{image_id}.jpg"
    image = np.asarray(Image.open(image_path).convert("RGB"))
    width = image.shape[1]
    if width % 2:
        raise RuntimeError(f"Expected even paired image width for {image_id}, found {width}")
    return image[:, : width // 2].copy(), image[:, width // 2 :].copy()


def preprocess(view: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(view, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor


def unwrap_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(raw_state).__name__}")
    if "model_state_dict" in raw_state:
        state = raw_state["model_state_dict"]
    elif "state_dict" in raw_state:
        state = raw_state["state_dict"]
    else:
        state = raw_state
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state dictionary is malformed")
    if state and all(str(key).startswith("_orig_mod.") for key in state):
        state = {str(key).removeprefix("_orig_mod."): value for key, value in state.items()}
    return state


def load_model(spec: ModelSpec, device: torch.device) -> BiomassModelTimm:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(spec.checkpoint)
    model = BiomassModelTimm(
        model_name=spec.model_name,
        dropout=0.2,
        use_mamba_ssm=False,
        pretrained=False,
        use_metadata=False,
        img_size=spec.image_size,
    )
    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(False)
    state = unwrap_state_dict(torch.load(spec.checkpoint, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    return model.to(device)


def token_relevance(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    prefix_tokens: int,
) -> np.ndarray:
    activation = activation.detach().float()[0, prefix_tokens:, :]
    gradient = gradient.detach().float()[0, prefix_tokens:, :]
    if activation.shape != gradient.shape:
        raise RuntimeError(f"Activation and gradient mismatch: {activation.shape} vs {gradient.shape}")

    relevance = torch.relu((activation * gradient).sum(dim=-1))
    side = math.isqrt(relevance.numel())
    if side * side != relevance.numel():
        raise RuntimeError(
            f"Patch token count {relevance.numel()} is not a square after removing {prefix_tokens} prefix tokens"
        )
    upper = torch.quantile(relevance, 0.99)
    if not torch.isfinite(upper) or upper <= 1e-12:
        raise RuntimeError("Positive token relevance is empty or nonfinite")
    relevance = torch.clamp(relevance / upper, 0.0, 1.0)
    return relevance.reshape(side, side).cpu().numpy()


def attribute_sample(
    model: BiomassModelTimm,
    spec: ModelSpec,
    left_view: np.ndarray,
    right_view: np.ndarray,
    target_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: list[torch.Tensor] = []

    def capture_backbone_output(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor backbone output, got {type(output).__name__}")
        output.retain_grad()
        captured.append(output)

    hook = model.backbone.register_forward_hook(capture_backbone_output)
    left = preprocess(left_view, spec.image_size, device)
    right = preprocess(right_view, spec.image_size, device)
    try:
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            prediction = model(left, right)
            score = prediction[0, target_index]
        if len(captured) != 2:
            raise RuntimeError(f"Expected two backbone outputs, captured {len(captured)}")
        score.backward()
        if captured[0].grad is None or captured[1].grad is None:
            raise RuntimeError("Backbone token gradients were not retained")
        prefix_tokens = int(getattr(model.backbone, "num_prefix_tokens", 1))
        left_map = token_relevance(captured[0], captured[0].grad, prefix_tokens)
        right_map = token_relevance(captured[1], captured[1].grad, prefix_tokens)
        prediction_np = prediction.detach().float().cpu().numpy()[0]
    finally:
        hook.remove()
        del left, right
    return prediction_np, left_map, right_map


def overlay_relevance(image: np.ndarray, relevance: np.ndarray) -> np.ndarray:
    resized = cv2.resize(relevance, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    focused = np.clip((resized - 0.20) / 0.80, 0.0, 1.0)
    color = plt.get_cmap("turbo")(focused)[..., :3] * 255.0
    alpha = (0.64 * np.power(focused, 0.80))[..., None]
    overlay = image.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def join_views(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    separator = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def build_figure(
    rows: list[dict[str, object]],
    destination_png: Path,
    destination_svg: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 7.6))
    column_titles = [
        "Original paired view",
        f"B4. {MODEL_SPECS[0].label}\nCV R² {MODEL_SPECS[0].cv_mean:.3f} ± {MODEL_SPECS[0].cv_std:.3f}",
        f"B5. {MODEL_SPECS[1].label}\nCV R² {MODEL_SPECS[1].cv_mean:.3f} ± {MODEL_SPECS[1].cv_std:.3f}",
    ]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontweight="bold", color="#253238", pad=7)

    for row_index, row in enumerate(rows):
        sample = row["sample"]
        truth = row["truth"]
        original = row["original"]
        axes[row_index, 0].imshow(original)
        axes[row_index, 0].set_ylabel(
            f"{sample.panel}\n{sample.image_id}\nTrue {sample.target_name.removeprefix('Dry_').removesuffix('_g')}: {truth:.1f} g",
            fontweight="bold",
            color="#253238",
            labelpad=9,
        )
        axes[row_index, 0].text(
            0.02,
            0.04,
            "Left view  |  Right view",
            transform=axes[row_index, 0].transAxes,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#102027", "alpha": 0.70, "edgecolor": "none"},
        )

        for column, model_spec in enumerate(MODEL_SPECS, start=1):
            result = row[model_spec.key]
            axes[row_index, column].imshow(result["overlay"])
            axes[row_index, column].text(
                0.02,
                0.04,
                f"Predicted: {result['prediction']:.1f} g",
                transform=axes[row_index, column].transAxes,
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#102027", "alpha": 0.70, "edgecolor": "none"},
            )

        for column in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_color("#CFD8DC")
                spine.set_linewidth(0.8)

    fig.suptitle(
        "Target Specific Spatial Attribution from Trained Biomass Models",
        fontsize=14,
        fontweight="bold",
        color="#1F2933",
        y=0.985,
    )
    colorbar_axis = fig.add_axes([0.4325, 0.046, 0.24, 0.013])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="turbo"),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 1], labels=["Lower", "Higher"])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(labelsize=7, length=2)
    fig.text(
        0.5525,
        0.071,
        "Within view positive contribution",
        ha="center",
        va="center",
        fontsize=8,
        color="#455A64",
    )
    fig.text(
        0.5525,
        0.006,
        "Positive gradient × final patch token activation. Each view is normalized independently.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#000000",
    )
    fig.subplots_adjust(left=0.11, right=0.995, top=0.925, bottom=0.082, wspace=0.025, hspace=0.13)

    destination_png.parent.mkdir(parents=True, exist_ok=True)
    destination_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination_png, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(destination_svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE_ROOT / "csiro-biomass")
    parser.add_argument(
        "--fold-file",
        type=Path,
        default=REPO_ROOT / "output" / "reruns_2026_09_13" / "folds_seed17.csv",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=WORKSPACE_ROOT / "newPaper" / "img" / "png" / "fig02_model_attributions.png",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=WORKSPACE_ROOT / "newPaper" / "img" / "svg" / "fig02_model_attributions.svg",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "output" / "interpretability" / "fig02_model_attributions.json",
    )
    parser.add_argument("--smoke", action="store_true", help="Run only B5 on the first sample without writing a figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical DINO Large attribution")

    wide = load_wide_data(args.data_root.resolve(), args.fold_file.resolve())
    sample_specs = SAMPLE_SPECS[:1] if args.smoke else SAMPLE_SPECS
    selected_models = MODEL_SPECS[1:] if args.smoke else MODEL_SPECS
    rows: list[dict[str, object]] = []
    for sample in sample_specs:
        matches = wide.loc[wide["image_id"].eq(sample.image_id)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one row for {sample.image_id}, found {len(matches)}")
        record = matches.iloc[0]
        if int(record["fold"]) != 0:
            raise RuntimeError(f"Sample {sample.image_id} is not in fold 0")
        left_view, right_view = load_original_views(args.data_root.resolve(), sample.image_id)
        rows.append(
            {
                "sample": sample,
                "truth": float(record[sample.target_name]),
                "original": join_views(left_view, right_view),
                "left_view": left_view,
                "right_view": right_view,
            }
        )

    checkpoint_metadata: dict[str, object] = {}
    for model_spec in selected_models:
        print(f"Loading {model_spec.key}: {model_spec.checkpoint}")
        model = load_model(model_spec, device)
        checkpoint_metadata[model_spec.key] = {
            "checkpoint": str(model_spec.checkpoint.resolve()),
            "checkpoint_sha256": sha256(model_spec.checkpoint),
            "model_name": model_spec.model_name,
            "image_size": model_spec.image_size,
            "fold": 0,
            "cv_mean_raw_r2": model_spec.cv_mean,
            "cv_std_raw_r2_ddof0": model_spec.cv_std,
        }
        for row in rows:
            sample = row["sample"]
            prediction, left_map, right_map = attribute_sample(
                model,
                model_spec,
                row["left_view"],
                row["right_view"],
                sample.target_index,
                device,
            )
            left_overlay = overlay_relevance(row["left_view"], left_map)
            right_overlay = overlay_relevance(row["right_view"], right_map)
            row[model_spec.key] = {
                "prediction": float(prediction[sample.target_index]),
                "all_predictions": {name: float(value) for name, value in zip(TARGET_COLUMNS, prediction)},
                "left_grid": list(left_map.shape),
                "right_grid": list(right_map.shape),
                "overlay": join_views(left_overlay, right_overlay),
            }
            print(
                f"  {sample.image_id} {sample.target_name}: "
                f"truth={row['truth']:.4f}, pred={prediction[sample.target_index]:.4f}, "
                f"grid={left_map.shape}"
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.smoke:
        print("Smoke attribution passed")
        return

    build_figure(rows, args.png.resolve(), args.svg.resolve())
    metadata = {
        "method": "positive gradient times final backbone patch token activation",
        "normalization": "independent per view, clipped at the 99th percentile",
        "interpretation_limit": "sensitivity visualization, not causal evidence",
        "data_fold": 0,
        "models": checkpoint_metadata,
        "samples": [],
    }
    for row in rows:
        sample = row["sample"]
        entry = {
            "image_id": sample.image_id,
            "target_name": sample.target_name,
            "truth_g": row["truth"],
            "fold": 0,
            "models": {},
        }
        for model_spec in MODEL_SPECS:
            result = row[model_spec.key]
            entry["models"][model_spec.key] = {
                "prediction_g": result["prediction"],
                "all_predictions_g": result["all_predictions"],
                "token_grid": result["left_grid"],
            }
        metadata["samples"].append(entry)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved figure: {args.png.resolve()}")
    print(f"Saved figure: {args.svg.resolve()}")
    print(f"Saved metadata: {args.metadata.resolve()}")


if __name__ == "__main__":
    main()
