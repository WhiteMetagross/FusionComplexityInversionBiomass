"""Deterministic feature extraction and preprocessing utilities for C series baselines.

Implements metadata preprocessing, handcrafted image feature extraction (163 per view,
652 cross-view), foundation model feature cache loading (DINOv2 Base and DINOv3 Large),
and universal target derivation and metric routines.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

EXPECTED_TRAIN_SHA256 = (
    "1a6b4c019c3a2386c7626b4002f1b2923d402341fb2a89132f66c57815c5f0e0"
)
EXPECTED_FOLDS_SHA256 = (
    "721f05a3dd0c00e4e2bd30bd2474da45d4ec1c7e3ca2abf88b75821bf7f7fe9f"
)

PRIMITIVE_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
ALL_TARGETS = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]
TARGET_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
EXPECTED_FOLD_COUNTS = {0: 72, 1: 72, 2: 71, 3: 71, 4: 71}


def sha256_file(filepath: Path | str) -> str:
    """Compute SHA256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_and_folds(
    data_dir: Path | str,
    fold_file: Path | str,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Load train.csv and locked outer folds, assert hashes and dimensions, pivot to wide format."""
    data_dir = Path(data_dir)
    train_csv_path = data_dir / "train.csv"
    fold_file = Path(fold_file)

    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found at {fold_file}")

    if verify_hashes:
        train_sha = sha256_file(train_csv_path)
        if train_sha != EXPECTED_TRAIN_SHA256:
            raise ValueError(
                f"train.csv SHA256 mismatch: got {train_sha}, expected {EXPECTED_TRAIN_SHA256}"
            )
        fold_sha = sha256_file(fold_file)
        if fold_sha != EXPECTED_FOLDS_SHA256:
            raise ValueError(
                f"Fold file SHA256 mismatch: got {fold_sha}, expected {EXPECTED_FOLDS_SHA256}"
            )

    long_df = pd.read_csv(train_csv_path)
    long_df["image_id"] = long_df["sample_id"].str.split("__").str[0]

    wide_df = long_df.pivot_table(
        index=[
            "image_id",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm",
            "Sampling_Date",
            "image_path",
        ],
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    for col in ALL_TARGETS:
        if col not in wide_df.columns:
            raise ValueError(f"Missing target column: {col}")

    folds_df = pd.read_csv(fold_file)
    merged_df = wide_df.merge(
        folds_df[["image_id", "fold"]], on="image_id", validate="one_to_one"
    )

    if len(merged_df) != 357:
        raise ValueError(
            f"Expected 357 rows after merging folds, got {len(merged_df)}"
        )

    merged_df = merged_df.sort_values("image_id").reset_index(drop=True)

    fold_counts = merged_df["fold"].value_counts().sort_index().to_dict()
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(
            f"Fold counts mismatch: got {fold_counts}, expected {EXPECTED_FOLD_COUNTS}"
        )

    # Verify all image files exist on disk
    for idx, row in merged_df.iterrows():
        rel_path = Path(row["image_path"])
        full_path = (
            rel_path if rel_path.is_absolute() else data_dir / rel_path
        )
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found at {full_path}")

    return merged_df


# -----------------------------------------------------------------------------
# Metadata Preprocessing
# -----------------------------------------------------------------------------


class MetadataPreprocessor:
    """Preprocesses State, Species, NDVI, Height, and month sine/cosine.

    Fits OneHotEncoder and optional continuous StandardScaler on training split
    only.
    """

    def __init__(self, scale_continuous: bool = True):
        self.scale_continuous = scale_continuous
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []

    def _extract_raw(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        cats = df[["State", "Species"]].copy()

        dates = pd.to_datetime(df["Sampling_Date"])
        months = dates.dt.month.to_numpy(dtype=np.float64)
        sin_month = np.sin(2.0 * np.pi * months / 12.0)
        cos_month = np.cos(2.0 * np.pi * months / 12.0)

        conts = np.column_stack(
            [
                df["Pre_GSHH_NDVI"].to_numpy(dtype=np.float64),
                df["Height_Ave_cm"].to_numpy(dtype=np.float64),
                sin_month,
                cos_month,
            ]
        )
        return cats, conts

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        cats, conts = self._extract_raw(df)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(cats)

        cat_names = [f"cat_{c}" for c in self.ohe.get_feature_names_out()]
        cont_names = [
            "cont_Pre_GSHH_NDVI",
            "cont_Height_Ave_cm",
            "cont_month_sin",
            "cont_month_cos",
        ]

        if self.scale_continuous:
            self.scaler = StandardScaler()
            self.scaler.fit(conts)

        self.feature_names_ = cat_names + cont_names
        return self

    def transform(
        self,
        df: pd.DataFrame,
        mode: str = "present",
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        if self.ohe is None:
            raise RuntimeError("Preprocessor has not been fitted")

        cats, conts = self._extract_raw(df)
        cat_arr = self.ohe.transform(cats)

        if self.scale_continuous and self.scaler is not None:
            cont_arr = self.scaler.transform(conts)
        else:
            cont_arr = conts

        features = np.hstack([cat_arr, cont_arr]).astype(np.float32)

        if mode == "present":
            return features
        elif mode == "zero":
            return np.zeros_like(features)
        elif mode == "shuffled":
            if random_state is None:
                raise ValueError(
                    "random_state must be provided for shuffled mode"
                )
            rng = np.random.RandomState(random_state)
            perm = rng.permutation(len(features))
            return features[perm]
        else:
            raise ValueError(f"Unknown metadata mode: {mode}")


# -----------------------------------------------------------------------------
# Handcrafted Image Features (163 per view, 652 cross-view)
# -----------------------------------------------------------------------------


def _safe_denom(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sign = np.where(d >= 0.0, 1.0, -1.0)
    return np.where(np.abs(d) < eps, sign * eps, d)


def get_handcrafted_feature_names() -> Tuple[List[str], List[str]]:
    """Generate exact 163 per-view feature names and 652 cross-view feature names."""
    names: List[str] = []

    # 1. Color moments (30)
    for ch in ["R", "G", "B", "H", "S", "V"]:
        for stat in ["mean", "std", "p10", "median", "p90"]:
            names.append(f"moment_{ch}_{stat}")

    # 2. Histograms (80)
    for ch in ["R", "G", "B", "H"]:
        for b in range(16):
            names.append(f"hist_{ch}_bin{b}")
    for ch in ["S", "V"]:
        for b in range(8):
            names.append(f"hist_{ch}_bin{b}")

    # 3. Visible vegetation indices (24)
    for idx in ["ExG", "GLI", "NGRDI", "VARI"]:
        for stat in ["mean", "std", "p10", "median", "p90"]:
            names.append(f"vvi_{idx}_{stat}")
        names.append(f"vvi_{idx}_pos_frac")

    # 4. Texture and edges (21)
    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]:
        names.append(f"glcm_{prop}_mean")
        names.append(f"glcm_{prop}_std")
    for b in range(10):
        names.append(f"lbp_bin{b}")
    names.append("canny_edge_density")

    # 5. Spatial heterogeneity (8)
    for q in ["tl", "tr", "bl", "br"]:
        names.append(f"spatial_exg_{q}_mean")
        names.append(f"spatial_exg_{q}_pos_frac")

    if len(names) != 163:
        raise ValueError(
            f"Expected 163 feature names per view, got {len(names)}"
        )

    cross_names: List[str] = []
    for n in names:
        cross_names.append(f"L_{n}")
    for n in names:
        cross_names.append(f"R_{n}")
    for n in names:
        cross_names.append(f"diff_{n}")
    for n in names:
        cross_names.append(f"prod_{n}")

    if len(cross_names) != 652:
        raise ValueError(
            f"Expected 652 cross-view feature names, got {len(cross_names)}"
        )

    return names, cross_names


def extract_view_features(half_rgb_float: np.ndarray) -> np.ndarray:
    """Extract deterministic 163-dimensional handcrafted descriptor from a 256x256 float32 RGB image in [0, 1]."""
    feats = []

    # 1. Color moments (30)
    R = half_rgb_float[:, :, 0]
    G = half_rgb_float[:, :, 1]
    B = half_rgb_float[:, :, 2]

    hsv = cv2.cvtColor(half_rgb_float, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]  # [0, 360)
    S = hsv[:, :, 1]  # [0, 1]
    V = hsv[:, :, 2]  # [0, 1]

    for ch in [R, G, B, H, S, V]:
        feats.append(float(np.mean(ch)))
        feats.append(float(np.std(ch, ddof=0)))
        feats.append(float(np.percentile(ch, 10)))
        feats.append(float(np.median(ch)))
        feats.append(float(np.percentile(ch, 90)))

    # 2. Histograms (80)
    for ch in [R, G, B]:
        h, _ = np.histogram(ch, bins=16, range=(0.0, 1.0))
        s = h.sum()
        feats.extend(
            (h.astype(np.float64) / s if s > 0 else np.zeros(16)).tolist()
        )

    h, _ = np.histogram(H, bins=16, range=(0.0, 360.0))
    s = h.sum()
    feats.extend((h.astype(np.float64) / s if s > 0 else np.zeros(16)).tolist())

    for ch in [S, V]:
        h, _ = np.histogram(ch, bins=8, range=(0.0, 1.0))
        s = h.sum()
        feats.extend(
            (h.astype(np.float64) / s if s > 0 else np.zeros(8)).tolist()
        )

    # 3. Visible vegetation indices (24)
    eps = 1e-6
    exg = 2.0 * G - R - B
    gli = (2.0 * G - R - B) / (2.0 * G + R + B + eps)
    ngrdi = (G - R) / (G + R + eps)
    vari = (G - R) / _safe_denom(G + R - B, eps=eps)

    for idx_arr in [exg, gli, ngrdi, vari]:
        idx_clipped = np.clip(idx_arr, -5.0, 5.0)
        feats.append(float(np.mean(idx_clipped)))
        feats.append(float(np.std(idx_clipped, ddof=0)))
        feats.append(float(np.percentile(idx_clipped, 10)))
        feats.append(float(np.median(idx_clipped)))
        feats.append(float(np.percentile(idx_clipped, 90)))
        feats.append(float((idx_clipped > 0.0).mean()))

    # 4. Texture and edges (21)
    gray_float = cv2.cvtColor(half_rgb_float, cv2.COLOR_RGB2GRAY)
    gray_uint8 = np.clip(gray_float * 255.0 + 0.5, 0, 255).astype(np.uint8)

    quantized = (gray_uint8 // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]:
        p_vals = graycoprops(glcm, prop)
        feats.append(float(np.mean(p_vals)))
        feats.append(float(np.std(p_vals, ddof=0)))

    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    s_lbp = lbp_hist.sum()
    feats.extend(
        (
            lbp_hist.astype(np.float64) / s_lbp
            if s_lbp > 0
            else np.zeros(10, dtype=np.float64)
        ).tolist()
    )

    edges = canny(gray_float, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    feats.append(float(edges.mean()))

    # 5. Spatial heterogeneity (8)
    q_tl = exg[:128, :128]
    q_tr = exg[:128, 128:]
    q_bl = exg[128:, :128]
    q_br = exg[128:, 128:]

    for q in [q_tl, q_tr, q_bl, q_br]:
        feats.append(float(np.mean(q)))
        feats.append(float((q > 0.0).mean()))

    arr = np.array(feats, dtype=np.float32)
    if arr.shape[0] != 163 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"View feature extraction failed: shape {arr.shape}, finite {np.all(np.isfinite(arr))}"
        )
    return arr


def extract_handcrafted_pair(
    image_path: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open image, split at width // 2, resize each half to 256x256 INTER_AREA, extract L, R, and 652 cross-view vector."""
    with Image.open(image_path) as img:
        img_rgb = np.array(img.convert("RGB"))

    width = img_rgb.shape[1]
    left = img_rgb[:, : width // 2]
    right = img_rgb[:, width // 2 :]

    left_resized = cv2.resize(left, (256, 256), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, (256, 256), interpolation=cv2.INTER_AREA)

    left_float = left_resized.astype(np.float32) / 255.0
    right_float = right_resized.astype(np.float32) / 255.0

    L = extract_view_features(left_float)
    R = extract_view_features(right_float)

    cross_view = np.hstack([L, R, np.abs(L - R), L * R]).astype(np.float32)
    if cross_view.shape[0] != 652 or not np.all(np.isfinite(cross_view)):
        raise ValueError("Invalid cross-view feature vector")
    return L, R, cross_view


def build_or_load_handcrafted_cache(
    df: pd.DataFrame,
    data_dir: Path | str,
    cache_dir: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build or load cached handcrafted image features for all 357 images."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "handcrafted_features.npz"
    meta_path = cache_dir / "handcrafted_features_metadata.json"

    view_names, cross_names = get_handcrafted_feature_names()
    image_ids = df["image_id"].tolist()

    if npz_path.exists() and meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        if (
            metadata.get("image_ids") == image_ids
            and metadata.get("feature_dim_per_view") == 163
            and metadata.get("cross_view_dim") == 652
        ):
            cached = np.load(npz_path)
            return cached["left"], cached["right"], cached["cross_view"], metadata

    print(
        f"Building handcrafted image feature cache for {len(df)} images on CPU..."
    )
    left_list, right_list, cross_list = [], [], []

    for idx, row in df.iterrows():
        rel_path = Path(row["image_path"])
        full_path = (
            rel_path if rel_path.is_absolute() else Path(data_dir) / rel_path
        )
        L, R, cross = extract_handcrafted_pair(full_path)
        left_list.append(L)
        right_list.append(R)
        cross_list.append(cross)

    left_arr = np.vstack(left_list)
    right_arr = np.vstack(right_list)
    cross_arr = np.vstack(cross_list)

    np.savez_compressed(
        npz_path, left=left_arr, right=right_arr, cross_view=cross_arr
    )

    npz_sha = sha256_file(npz_path)
    metadata = {
        "feature_type": "handcrafted",
        "implementation_version": "1.0.0",
        "resize": "256x256 INTER_AREA",
        "normalization": "RGB in [0, 1]",
        "num_images": len(df),
        "feature_dim_per_view": 163,
        "cross_view_dim": 652,
        "image_ids": image_ids,
        "source_train_sha256": EXPECTED_TRAIN_SHA256,
        "cache_sha256": npz_sha,
        "view_feature_names": view_names,
        "cross_feature_names": cross_names,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return left_arr, right_arr, cross_arr, metadata


# -----------------------------------------------------------------------------
# C6: DINOv2 Base Feature Loading
# -----------------------------------------------------------------------------


def load_dinov2_b3_cache(
    df: pd.DataFrame,
    b3_cache_dir: Path | str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load B3 DINOv2 Base features and construct 3072-dimensional cross-view vector."""
    b3_cache_dir = Path(b3_cache_dir)
    npz_path = b3_cache_dir / "features.npz"
    meta_path = b3_cache_dir / "feature_metadata.json"

    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"B3 DINOv2 cache files missing at {b3_cache_dir}"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if meta.get("model_id") != "facebook/dinov2-base":
        raise ValueError(f"Unexpected B3 model_id: {meta.get('model_id')}")
    if meta.get("image_ids") != df["image_id"].tolist():
        raise ValueError(
            "B3 cache image_ids do not match locked dataframe order"
        )

    cached = np.load(npz_path)
    left = cached["left"].astype(np.float32)
    right = cached["right"].astype(np.float32)

    if (
        left.shape != (357, 768)
        or right.shape != (357, 768)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError(f"Corrupt B3 cache arrays: {left.shape}, {right.shape}")

    cross_view = np.hstack([left, right, np.abs(left - right), left * right])
    if cross_view.shape != (357, 3072):
        raise ValueError(
            f"Expected cross-view shape (357, 3072), got {cross_view.shape}"
        )

    return cross_view, meta


# -----------------------------------------------------------------------------
# C7 and C8: DINOv3 Large Feature Extraction and Caching
# -----------------------------------------------------------------------------


class DINOv3SplitDataset(Dataset):
    """Dataset applying deterministic B5 validation transform to left and right halves."""

    def __init__(self, df: pd.DataFrame, data_dir: Path | str):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = T.Compose(
            [
                T.Resize(
                    (512, 512),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        rel_path = Path(row["image_path"])
        full_path = (
            rel_path if rel_path.is_absolute() else self.data_dir / rel_path
        )
        with Image.open(full_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            left = rgb.crop((0, 0, w // 2, h))
            right = rgb.crop((w // 2, 0, w, h))
            return self.transform(left), self.transform(right), idx


def build_or_load_dinov3_cache(
    df: pd.DataFrame,
    data_dir: Path | str,
    cache_dir: Path | str,
    batch_size: int = 8,
    git_revision: str = "unknown",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract or load shared frozen DINOv3 Large pooled features (4096-dim cross-view)."""
    import timm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "dinov3_vitl16_512.npz"
    meta_path = cache_dir / "dinov3_vitl16_512_metadata.json"

    image_ids = df["image_id"].tolist()

    if npz_path.exists() and meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        if (
            metadata.get("model_name") == "vit_large_patch16_dinov3.lvd1689m"
            and metadata.get("image_ids") == image_ids
            and metadata.get("cross_view_dim") == 4096
        ):
            cached = np.load(npz_path)
            left = cached["left"].astype(np.float32)
            right = cached["right"].astype(np.float32)
            cross_view = np.hstack(
                [left, right, np.abs(left - right), left * right]
            )
            return cross_view, metadata

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for DINOv3 Large feature extraction"
        )

    print(
        f"Extracting DINOv3 Large features on GPU for {len(df)} images (batch_size={batch_size})..."
    )
    device = torch.device("cuda")

    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    ).to(device)
    model.eval()

    dataset = DINOv3SplitDataset(df, data_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    left_feats = []
    right_feats = []

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            for left_batch, right_batch, _ in loader:
                left_batch = left_batch.to(device)
                right_batch = right_batch.to(device)

                out_l = model(left_batch)
                out_r = model(right_batch)

                left_feats.append(out_l.cpu().float().numpy())
                right_feats.append(out_r.cpu().float().numpy())

    left_arr = np.vstack(left_feats).astype(np.float32)
    right_arr = np.vstack(right_feats).astype(np.float32)

    if left_arr.shape != (357, 1024) or right_arr.shape != (357, 1024):
        raise ValueError(
            f"DINOv3 feature shape mismatch: {left_arr.shape}, {right_arr.shape}"
        )

    cross_view = np.hstack(
        [left_arr, right_arr, np.abs(left_arr - right_arr), left_arr * right_arr]
    ).astype(np.float32)

    np.savez_compressed(npz_path, left=left_arr, right=right_arr)
    cache_sha = sha256_file(npz_path)

    metadata = {
        "model_name": "vit_large_patch16_dinov3.lvd1689m",
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda
        if torch.cuda.is_available()
        else None,
        "input_transform": (
            "Resize(512, 512, BILINEAR), ImageNet Normalize, no aug"
        ),
        "pooling_method": "global_pool=avg",
        "embedding_dim": 1024,
        "cross_view_dim": 4096,
        "num_images": len(df),
        "image_ids": image_ids,
        "source_train_sha256": EXPECTED_TRAIN_SHA256,
        "cache_sha256": cache_sha,
        "model_source": "timm official weights",
        "code_revision": git_revision,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return cross_view, metadata


# -----------------------------------------------------------------------------
# Target Derivation and Weighted R²
# -----------------------------------------------------------------------------


def derive_targets(preds_primitive: np.ndarray) -> np.ndarray:
    """Derive 5 targets from primitive predictions: Green, Dead, Clover, GDM, Total.

    Clips Green, Dead, Clover at lower bound 0.0.
    GDM = Green + Clover
    Total = GDM + Dead
    """
    clipped = np.clip(preds_primitive, 0.0, None)
    green = clipped[:, 0]
    dead = clipped[:, 1]
    clover = clipped[:, 2]

    gdm = green + clover
    total = gdm + dead

    derived = np.column_stack([green, dead, clover, gdm, total])
    return derived


def compute_weighted_r2(
    y_true_5: np.ndarray, y_pred_5: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute raw weighted R² across 5 targets matching src/engine.py implementation."""
    r2_scores = []
    for i in range(5):
        yt = y_true_5[:, i]
        yp = y_pred_5[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
        r2_scores.append(r2)

    r2_arr = np.array(r2_scores, dtype=np.float64)
    weighted = float(np.sum(r2_arr * TARGET_WEIGHTS) / np.sum(TARGET_WEIGHTS))
    return weighted, r2_arr
