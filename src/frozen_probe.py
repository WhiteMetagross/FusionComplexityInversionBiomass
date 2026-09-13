"""Supervised frozen dual view probes for DINOv2 biomass baselines."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import AutoModel

from models import GatedDepthwiseConvBlock


TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)


class SlowModelError(RuntimeError):
    """Raised when measured extraction time exceeds the configured limit."""


@dataclass(frozen=True)
class ProbeConfig:
    run_name: str
    model_id: str
    data_dir: str
    output_dir: str
    fold_file: str
    seed: int = 17
    folds: int = 5
    image_size: int = 448
    extraction_batch_size: int = 4
    workers: int = 2
    projection_dim: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    head_batch_size: int = 64
    max_epochs: int = 400
    validation_interval: int = 5
    patience_checks: int = 20
    max_extraction_hours: float = 2.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    scores = []
    for index in range(y_true.shape[1]):
        residual = np.square(y_true[:, index] - y_pred[:, index]).sum()
        centered = np.square(y_true[:, index] - y_true[:, index].mean()).sum()
        scores.append(1.0 - residual / centered if centered > 0 else 0.0)
    per_target = np.asarray(scores, dtype=np.float64)
    return float(np.sum(per_target * TARGET_WEIGHTS)), per_target


def load_fold_data(config: ProbeConfig) -> pd.DataFrame:
    csv_path = Path(config.data_dir) / "train.csv"
    long_df = pd.read_csv(csv_path)
    long_df["image_id"] = long_df["sample_id"].str.split("__").str[0]
    wide_df = long_df.pivot_table(
        index=["image_id", "image_path"],
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    missing_targets = [column for column in TARGET_COLS if column not in wide_df]
    if missing_targets:
        raise ValueError(f"Missing targets: {missing_targets}")

    fold_path = Path(config.fold_file)
    if fold_path.exists():
        saved = pd.read_csv(fold_path)
        wide_df = wide_df.merge(saved[["image_id", "fold"]], on="image_id", validate="one_to_one")
        if len(wide_df) != saved["image_id"].nunique():
            raise ValueError("Fold file does not exactly match the training images")
    else:
        wide_df["total_bin"] = pd.qcut(
            wide_df["Dry_Total_g"], q=5, labels=False, duplicates="drop"
        )
        splitter = StratifiedGroupKFold(
            n_splits=config.folds, shuffle=True, random_state=config.seed
        )
        wide_df["fold"] = -1
        for fold, (_, val_indices) in enumerate(
            splitter.split(wide_df, wide_df["total_bin"], groups=wide_df["image_id"])
        ):
            wide_df.loc[val_indices, "fold"] = fold
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        wide_df[["image_id", "fold"]].sort_values("image_id").to_csv(fold_path, index=False)

    wide_df = wide_df.sort_values("image_id").reset_index(drop=True)
    if sorted(wide_df["fold"].unique().tolist()) != list(range(config.folds)):
        raise ValueError("Fold labels are incomplete")
    return wide_df


class SplitImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_dir: Path, image_size: int):
        self.frame = frame
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        relative_path = Path(self.frame.iloc[index]["image_path"])
        image_path = relative_path if relative_path.is_absolute() else self.data_dir / relative_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            midpoint = image.width // 2
            left = image.crop((0, 0, midpoint, image.height))
            right = image.crop((midpoint, 0, image.width, image.height))
            return self.transform(left), self.transform(right), index


def extract_features(config: ProbeConfig, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "features.npz"
    metadata_path = output_dir / "feature_metadata.json"
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("model_id") == config.model_id
            and metadata.get("image_size") == config.image_size
            and metadata.get("image_ids") == frame["image_id"].tolist()
        ):
            cached = np.load(cache_path)
            return cached["left"], cached["right"], metadata

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for foundation model feature extraction")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        config.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    hidden_size = int(model.config.hidden_size)

    dataset = SplitImageDataset(frame, Path(config.data_dir), config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )
    left_features = np.empty((len(frame), hidden_size), dtype=np.float32)
    right_features = np.empty((len(frame), hidden_size), dtype=np.float32)

    extraction_started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for batch_index, (left, right, indices) in enumerate(tqdm(loader, desc=config.run_name)):
            pixels = torch.cat([left, right], dim=0).to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = model(pixel_values=pixels).last_hidden_state
                pooled = tokens[:, 0].float().cpu().numpy()
            batch_size = left.shape[0]
            numpy_indices = indices.numpy()
            left_features[numpy_indices] = pooled[:batch_size]
            right_features[numpy_indices] = pooled[batch_size:]
            completed += batch_size

            if batch_index == 4:
                elapsed = time.perf_counter() - extraction_started
                projected_hours = elapsed * len(frame) / completed / 3600.0
                if projected_hours > config.max_extraction_hours:
                    raise SlowModelError(
                        f"Projected extraction is {projected_hours:.2f} hours, "
                        f"above limit {config.max_extraction_hours:.2f} hours"
                    )

    extraction_seconds = time.perf_counter() - extraction_started
    metadata = {
        "model_id": config.model_id,
        "parameter_count": parameter_count,
        "hidden_size": hidden_size,
        "image_size": config.image_size,
        "image_ids": frame["image_id"].tolist(),
        "load_seconds": load_seconds,
        "extraction_seconds": extraction_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "dtype": "float16",
        "attention": "PyTorch SDPA",
    }
    np.savez_compressed(cache_path, left=left_features, right=right_features)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return left_features, right_features, metadata


def _component_head(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim // 2, 1),
        nn.Softplus(),
    )


class DualViewFrozenProbe(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.view_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            GatedDepthwiseConvBlock(projection_dim, kernel_size=3, dropout=dropout),
            GatedDepthwiseConvBlock(projection_dim, kernel_size=3, dropout=dropout),
        )
        self.head_green = _component_head(projection_dim, dropout)
        self.head_dead = _component_head(projection_dim, dropout)
        self.head_clover = _component_head(projection_dim, dropout)

    def set_component_biases(self, means: np.ndarray) -> None:
        for head, mean in zip(
            (self.head_green, self.head_dead, self.head_clover), means[:3], strict=True
        ):
            final = head[-2]
            nn.init.normal_(final.weight, std=1e-3)
            value = max(float(mean), 1e-4)
            inverse_softplus = value + math.log(-math.expm1(-value))
            nn.init.constant_(final.bias, inverse_softplus)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        sequence = torch.stack(
            [self.view_projection(left), self.view_projection(right)], dim=1
        )
        pooled = self.fusion(sequence).mean(dim=1)
        green = self.head_green(pooled)
        dead = self.head_dead(pooled)
        clover = self.head_clover(pooled)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)


@torch.no_grad()
def _predict(model: nn.Module, features: tuple[torch.Tensor, torch.Tensor], indices: np.ndarray) -> np.ndarray:
    model.eval()
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features[0].device)
    predictions = model(features[0][index_tensor], features[1][index_tensor])
    return predictions.cpu().numpy()


def _new_model(config: ProbeConfig, features, labels, train_indices, seed):
    _seed_everything(seed)
    model = DualViewFrozenProbe(
        features[0].shape[1], config.projection_dim, config.dropout
    ).to(features[0].device)
    model.set_component_biases(labels[train_indices].mean(dim=0).cpu().numpy())
    return model


def _train_epoch(model, loader, features, labels, optimizer, loss_fn):
    model.train()
    for (batch_indices,) in loader:
        batch_indices = batch_indices.to(features[0].device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features[0][batch_indices], features[1][batch_indices])
        loss = loss_fn(predictions, labels[batch_indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _make_loader(indices, config, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.as_tensor(indices, dtype=torch.long)),
        batch_size=config.head_batch_size,
        shuffle=True,
        generator=generator,
    )


def _fit_for_epochs(config, features, labels, train_indices, epochs, seed):
    model = _new_model(config, features, labels, train_indices, seed)
    loader = _make_loader(train_indices, config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True
    )
    loss_fn = nn.SmoothL1Loss(beta=5.0)
    for _ in range(epochs):
        _train_epoch(model, loader, features, labels, optimizer, loss_fn)
    return model


def _select_epoch(config, features, labels, outer_train, seed):
    target_values = labels[outer_train, 4].cpu().numpy()
    bins = pd.qcut(target_values, q=5, labels=False, duplicates="drop")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_train_relative, inner_val_relative = next(splitter.split(outer_train, bins))
    inner_train = outer_train[inner_train_relative]
    inner_val = outer_train[inner_val_relative]
    model = _new_model(config, features, labels, inner_train, seed)
    loader = _make_loader(inner_train, config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True
    )
    loss_fn = nn.SmoothL1Loss(beta=5.0)
    best_epoch = config.validation_interval
    best_score = -float("inf")
    stale_checks = 0

    for epoch in range(1, config.max_epochs + 1):
        _train_epoch(model, loader, features, labels, optimizer, loss_fn)
        if epoch % config.validation_interval:
            continue
        predictions = _predict(model, features, inner_val)
        score, _ = weighted_r2(labels[inner_val].cpu().numpy(), predictions)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_checks = 0
        else:
            stale_checks += 1
            if stale_checks >= config.patience_checks:
                break
    return best_epoch, best_score


def run_probe(config: ProbeConfig) -> dict:
    started = time.perf_counter()
    _seed_everything(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    frame = load_fold_data(config)
    left_features, right_features, feature_metadata = extract_features(config, frame)
    device = torch.device("cuda")
    feature_tensors = (
        torch.from_numpy(left_features).to(device),
        torch.from_numpy(right_features).to(device),
    )
    labels = torch.from_numpy(frame[TARGET_COLS].to_numpy(np.float32)).to(device)
    oof_predictions = np.full((len(frame), len(TARGET_COLS)), np.nan, dtype=np.float32)
    fold_results = []
    output_dir = Path(config.output_dir)

    for fold in range(config.folds):
        outer_train = np.flatnonzero(frame["fold"].to_numpy() != fold)
        outer_val = np.flatnonzero(frame["fold"].to_numpy() == fold)
        fold_seed = config.seed + fold
        selected_epoch, inner_score = _select_epoch(
            config, feature_tensors, labels, outer_train, fold_seed
        )
        model = _fit_for_epochs(
            config, feature_tensors, labels, outer_train, selected_epoch, fold_seed
        )
        predictions = _predict(model, feature_tensors, outer_val)
        score, per_target = weighted_r2(labels[outer_val].cpu().numpy(), predictions)
        oof_predictions[outer_val] = predictions
        torch.save(model.state_dict(), output_dir / f"probe_fold{fold}.pth")
        result = {
            "fold": fold,
            "selected_epoch": selected_epoch,
            "inner_validation_weighted_r2": inner_score,
            "outer_weighted_r2": score,
            "outer_per_target_r2": dict(zip(TARGET_COLS, per_target.tolist(), strict=True)),
            "train_samples": int(len(outer_train)),
            "validation_samples": int(len(outer_val)),
        }
        fold_results.append(result)
        (output_dir / f"fold{fold}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        del model
        torch.cuda.empty_cache()

    if np.isnan(oof_predictions).any():
        raise RuntimeError("OOF prediction matrix is incomplete")
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        image_id=frame["image_id"].to_numpy(),
        fold=frame["fold"].to_numpy(),
        targets=frame[TARGET_COLS].to_numpy(np.float32),
        predictions=oof_predictions,
    )
    oof_score, oof_per_target = weighted_r2(
        frame[TARGET_COLS].to_numpy(np.float32), oof_predictions
    )
    fold_scores = [item["outer_weighted_r2"] for item in fold_results]
    root = Path(__file__).resolve().parents[1]
    summary = {
        "status": "complete",
        "run_name": config.run_name,
        "protocol": "supervised frozen dual view nonlinear probe",
        "backbone_frozen": True,
        "outer_fold_selection_reuse": False,
        "seed": config.seed,
        "seed_uncertainty_measured": False,
        "config": asdict(config),
        "model": feature_metadata,
        "fold_results": fold_results,
        "mean_fold_weighted_r2": float(np.mean(fold_scores)),
        "fold_dispersion_std": float(np.std(fold_scores)),
        "pooled_oof_weighted_r2": oof_score,
        "pooled_oof_per_target_r2": dict(zip(TARGET_COLS, oof_per_target.tolist(), strict=True)),
        "dataset_train_csv_sha256": _sha256(Path(config.data_dir) / "train.csv"),
        "fold_file_sha256": _sha256(Path(config.fold_file)),
        "git_revision": _git_revision(root),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_seconds": time.perf_counter() - started,
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
