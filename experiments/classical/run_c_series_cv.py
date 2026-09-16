"""Runner for C series classical baselines (C1 through C8) with inner 3-fold CV selection

and locked outer 5-fold cross-validation.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import torch

# Ensure src/ is importable
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from classical_features import (
    ALL_TARGETS,
    EXPECTED_FOLD_COUNTS,
    EXPECTED_FOLDS_SHA256,
    EXPECTED_TRAIN_SHA256,
    PRIMITIVE_TARGETS,
    TARGET_WEIGHTS,
    MetadataPreprocessor,
    build_or_load_dinov3_cache,
    build_or_load_handcrafted_cache,
    compute_weighted_r2,
    derive_targets,
    load_dataset_and_folds,
    load_dinov2_b3_cache,
    sha256_file,
)

MASTER_SEED = 17

XGB_CONFIGS: Dict[str, Dict[str, Any]] = {
    "X1": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "complexity": 1,
    },
    "X2": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_child_weight": 5,
        "subsample": 1.00,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.00,
        "reg_lambda": 5.0,
        "complexity": 2,
    },
    "X3": {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "complexity": 3,
    },
    "X4": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.50,
        "reg_alpha": 1.00,
        "reg_lambda": 10.0,
        "complexity": 4,
    },
    "X5": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 1.00,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.00,
        "reg_lambda": 5.0,
        "complexity": 5,
    },
    "X6": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.50,
        "reg_alpha": 1.00,
        "reg_lambda": 10.0,
        "complexity": 6,
    },
    "X7": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.10,
        "reg_lambda": 10.0,
        "complexity": 7,
    },
    "X8": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 15,
        "subsample": 1.00,
        "colsample_bytree": 0.50,
        "reg_alpha": 1.00,
        "reg_lambda": 20.0,
        "complexity": 8,
    },
}

RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def get_git_revision() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return rev
    except Exception:
        return "unknown"


class MultiTargetXGBoost:
    """Trains three independent XGBoost regressors, one per primitive target."""

    def __init__(self, params: Dict[str, Any]):
        self.params = {k: v for k, v in params.items() if k != "complexity"}
        self.models: List[xgb.XGBRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiTargetXGBoost":
        self.models = []
        for i in range(3):
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                n_jobs=1,
                random_state=MASTER_SEED,
                verbosity=0,
                **self.params,
            )
            model.fit(X, y[:, i])
            self.models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [model.predict(X) for model in self.models]
        return np.column_stack(preds)


class ClassicalPipeline:
    """Wrapper encapsulating feature scaler/PCA and multioutput regressor."""

    def __init__(
        self,
        model_type: str,
        pca_components: Optional[int] = None,
        use_scaler: bool = True,
        ridge_alpha: Optional[float] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.pca_components = pca_components
        self.use_scaler = use_scaler
        self.ridge_alpha = ridge_alpha
        self.xgb_params = xgb_params

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.regressor: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalPipeline":
        curr_X = X
        if self.use_scaler:
            self.scaler = StandardScaler()
            curr_X = self.scaler.fit_transform(curr_X)

        if self.pca_components is not None:
            self.pca = PCA(
                n_components=self.pca_components, random_state=MASTER_SEED
            )
            curr_X = self.pca.fit_transform(curr_X)

        if self.model_type == "ridge":
            self.regressor = Ridge(
                alpha=self.ridge_alpha, fit_intercept=True, random_state=MASTER_SEED
            )
            self.regressor.fit(curr_X, y)
        elif self.model_type == "xgboost":
            self.regressor = MultiTargetXGBoost(self.xgb_params)
            self.regressor.fit(curr_X, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        curr_X = X
        if self.use_scaler and self.scaler is not None:
            curr_X = self.scaler.transform(curr_X)
        if self.pca_components is not None and self.pca is not None:
            curr_X = self.pca.transform(curr_X)
        return self.regressor.predict(curr_X)


def define_candidate_grid(
    model_name: str,
) -> List[Tuple[str, Dict[str, Any], int]]:
    """Define candidate configs as (config_id, params_dict, complexity_rank).

    Lower complexity_rank indicates simpler model (tie-break rule).
    """
    candidates = []

    if model_name in ["C1", "C3"]:
        # Ridge over alpha grid
        for i, alpha in enumerate(RIDGE_ALPHAS):
            cid = f"Ridge_alpha_{alpha:g}"
            # Higher alpha = more regularization = lower complexity
            complexity = len(RIDGE_ALPHAS) - i
            params = {
                "model_type": "ridge",
                "use_scaler": (model_name == "C3"),
                "pca_components": None,
                "ridge_alpha": alpha,
                "xgb_params": None,
            }
            candidates.append((cid, params, complexity))

    elif model_name in ["C2", "C4", "C5"]:
        # XGBoost configs X1 through X8
        for cid in ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]:
            cfg = XGB_CONFIGS[cid]
            params = {
                "model_type": "xgboost",
                "use_scaler": False,
                "pca_components": None,
                "ridge_alpha": None,
                "xgb_params": cfg,
            }
            candidates.append((cid, params, cfg["complexity"]))

    elif model_name in ["C6", "C7"]:
        # Ridge with PCA candidates [None, 64, 128] x 9 alphas
        pca_opts = [None, 64, 128]
        for p_idx, p_comp in enumerate(pca_opts):
            p_str = f"pca_{p_comp}" if p_comp is not None else "pca_none"
            for a_idx, alpha in enumerate(RIDGE_ALPHAS):
                cid = f"{p_str}__alpha_{alpha:g}"
                # Complexity: 64 < 128 < none; higher alpha = lower complexity
                p_rank = 0 if p_comp == 64 else (1 if p_comp == 128 else 2)
                a_rank = len(RIDGE_ALPHAS) - a_idx
                complexity = p_rank * 100 + a_rank
                params = {
                    "model_type": "ridge",
                    "use_scaler": True,
                    "pca_components": p_comp,
                    "ridge_alpha": alpha,
                    "xgb_params": None,
                }
                candidates.append((cid, params, complexity))

    elif model_name == "C8":
        # DINOv3 + PCA (32, 64, 128) + XGBoost (X1..X4)
        pca_opts = [32, 64, 128]
        xgb_opts = ["X1", "X2", "X3", "X4"]
        for p_comp in pca_opts:
            for x_id in xgb_opts:
                cid = f"pca_{p_comp}__{x_id}"
                cfg = XGB_CONFIGS[x_id]
                complexity = p_comp * 10 + cfg["complexity"]
                params = {
                    "model_type": "xgboost",
                    "use_scaler": True,
                    "pca_components": p_comp,
                    "ridge_alpha": None,
                    "xgb_params": cfg,
                }
                candidates.append((cid, params, complexity))
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return candidates


def run_inner_selection(
    X_train_raw: np.ndarray,
    y_train_primitive: np.ndarray,
    y_train_5: np.ndarray,
    df_train_sub: pd.DataFrame,
    outer_fold: int,
    candidates: List[Tuple[str, Dict[str, Any], int]],
    is_multimodal: bool = False,
    image_feats_train: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Perform 3-fold stratified inner cross-validation on outer train rows.

    Strata derived from 3 quantile bins of Dry_Total_g.
    Evaluates each candidate on inner validation folds, derives 5 targets,
    scores via mean inner weighted R2.
    """
    total_target = df_train_sub["Dry_Total_g"].to_numpy()
    try:
        strata = pd.qcut(total_target, q=3, labels=False, duplicates="drop")
        unique_strata = np.unique(strata)
        if len(unique_strata) < 3:
            raise RuntimeError(
                f"Failed to create 3 non-empty strata for outer fold {outer_fold}"
            )
    except Exception as e:
        raise RuntimeError(
            f"Inner quantile binning failed on outer fold {outer_fold}: {e}"
        )

    inner_splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=MASTER_SEED + outer_fold
    )
    inner_splits = list(inner_splitter.split(df_train_sub, strata))

    candidate_results = []
    best_candidate_id = None
    best_candidate_params = None
    best_candidate_complexity = 999999
    best_candidate_score = -999999.0

    for cid, params, complexity in candidates:
        inner_scores = []

        for in_train_idx, in_val_idx in inner_splits:
            if is_multimodal:
                # For C5: fit metadata preprocessor strictly on inner train
                df_in_tr = df_train_sub.iloc[in_train_idx]
                df_in_val = df_train_sub.iloc[in_val_idx]

                mp = MetadataPreprocessor(scale_continuous=False)
                mp.fit(df_in_tr)
                meta_tr = mp.transform(df_in_tr, mode="present")
                meta_val = mp.transform(df_in_val, mode="present")

                img_tr = image_feats_train[in_train_idx]
                img_val = image_feats_train[in_val_idx]

                X_in_tr = np.hstack([img_tr, meta_tr])
                X_in_val = np.hstack([img_val, meta_val])
            elif "is_c1_metadata" in params or "is_c2_metadata" in params:
                df_in_tr = df_train_sub.iloc[in_train_idx]
                df_in_val = df_train_sub.iloc[in_val_idx]

                scale_cont = params.get("is_c1_metadata", False)
                mp = MetadataPreprocessor(scale_continuous=scale_cont)
                mp.fit(df_in_tr)
                X_in_tr = mp.transform(df_in_tr, mode="present")
                X_in_val = mp.transform(df_in_val, mode="present")
            else:
                X_in_tr = X_train_raw[in_train_idx]
                X_in_val = X_train_raw[in_val_idx]

            y_in_tr = y_train_primitive[in_train_idx]
            y_in_val_5 = y_train_5[in_val_idx]

            pipe = ClassicalPipeline(
                model_type=params["model_type"],
                pca_components=params["pca_components"],
                use_scaler=params["use_scaler"],
                ridge_alpha=params["ridge_alpha"],
                xgb_params=params["xgb_params"],
            )
            pipe.fit(X_in_tr, y_in_tr)
            raw_preds = pipe.predict(X_in_val)
            derived_preds = derive_targets(raw_preds)

            score, _ = compute_weighted_r2(y_in_val_5, derived_preds)
            inner_scores.append(score)

        mean_inner = float(np.mean(inner_scores))
        candidate_results.append(
            {
                "candidate_id": cid,
                "inner_fold_0": inner_scores[0],
                "inner_fold_1": inner_scores[1],
                "inner_fold_2": inner_scores[2],
                "mean_score": mean_inner,
                "complexity": complexity,
                "params": params,
            }
        )

        # Selection logic: highest mean score; exact ties within 1e-12 use lower complexity, then lexical ID
        is_better = False
        if mean_inner > best_candidate_score + 1e-12:
            is_better = True
        elif abs(mean_inner - best_candidate_score) <= 1e-12:
            if complexity < best_candidate_complexity:
                is_better = True
            elif complexity == best_candidate_complexity:
                if best_candidate_id is None or cid < best_candidate_id:
                    is_better = True

        if is_better:
            best_candidate_score = mean_inner
            best_candidate_id = cid
            best_candidate_params = params
            best_candidate_complexity = complexity

    return best_candidate_id, best_candidate_params, candidate_results


def run_model_5fold_cv(
    model_name: str,
    df: pd.DataFrame,
    features_dict: Dict[str, Any],
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute complete 5-fold cross validation for a C series model."""
    print(f"\n=======================================================")
    print(f"Starting 5-Fold Cross-Validation for {model_name}...")
    print(f"=======================================================")

    candidates = define_candidate_grid(model_name)
    fold_models_dir = output_dir / "fold_models" / model_name
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    oof_records = []
    inner_candidate_rows = []
    selected_hyperparam_rows = []
    fold_metric_rows = []

    y_primitive_all = df[PRIMITIVE_TARGETS].to_numpy(dtype=np.float64)
    y_5_all = df[ALL_TARGETS].to_numpy(dtype=np.float64)

    for outer_fold in range(5):
        val_mask = df["fold"] == outer_fold
        train_mask = ~val_mask

        val_indices = np.where(val_mask)[0]
        train_indices = np.where(train_mask)[0]

        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_val = df.iloc[val_indices].reset_index(drop=True)

        y_train_prim = y_primitive_all[train_indices]
        y_val_prim = y_primitive_all[val_indices]
        y_train_5 = y_5_all[train_indices]
        y_val_5 = y_5_all[val_indices]

        # Prepare feature representations
        meta_preprocessor: Optional[MetadataPreprocessor] = None
        X_train_fold: Optional[np.ndarray] = None
        X_val_fold: Optional[np.ndarray] = None

        if model_name == "C1":
            # Tag candidate params for inner split preprocessing
            for _, p, _ in candidates:
                p["is_c1_metadata"] = True
            meta_preprocessor = MetadataPreprocessor(scale_continuous=True)
            meta_preprocessor.fit(df_train)
            X_train_fold = meta_preprocessor.transform(df_train, mode="present")
            X_val_fold = meta_preprocessor.transform(df_val, mode="present")

        elif model_name == "C2":
            for _, p, _ in candidates:
                p["is_c2_metadata"] = True
            meta_preprocessor = MetadataPreprocessor(scale_continuous=False)
            meta_preprocessor.fit(df_train)
            X_train_fold = meta_preprocessor.transform(df_train, mode="present")
            X_val_fold = meta_preprocessor.transform(df_val, mode="present")

        elif model_name in ["C3", "C4"]:
            img_feats = features_dict["handcrafted_cross"]
            X_train_fold = img_feats[train_indices]
            X_val_fold = img_feats[val_indices]

        elif model_name == "C5":
            meta_preprocessor = MetadataPreprocessor(scale_continuous=False)
            meta_preprocessor.fit(df_train)
            meta_train = meta_preprocessor.transform(df_train, mode="present")
            meta_val = meta_preprocessor.transform(df_val, mode="present")

            img_feats = features_dict["handcrafted_cross"]
            img_train = img_feats[train_indices]
            img_val = img_feats[val_indices]

            X_train_fold = np.hstack([img_train, meta_train])
            X_val_fold = np.hstack([img_val, meta_val])

        elif model_name == "C6":
            dinov2_cross = features_dict["dinov2_cross"]
            X_train_fold = dinov2_cross[train_indices]
            X_val_fold = dinov2_cross[val_indices]

        elif model_name in ["C7", "C8"]:
            dinov3_cross = features_dict["dinov3_cross"]
            X_train_fold = dinov3_cross[train_indices]
            X_val_fold = dinov3_cross[val_indices]

        # Inner selection
        best_cid, best_params, cand_results = run_inner_selection(
            X_train_raw=X_train_fold,
            y_train_primitive=y_train_prim,
            y_train_5=y_train_5,
            df_train_sub=df_train,
            outer_fold=outer_fold,
            candidates=candidates,
            is_multimodal=(model_name == "C5"),
            image_feats_train=features_dict.get("handcrafted_cross")[
                train_indices
            ]
            if model_name == "C5"
            else None,
        )

        for cr in cand_results:
            inner_candidate_rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "candidate_id": cr["candidate_id"],
                    "inner_fold_0_score": cr["inner_fold_0"],
                    "inner_fold_1_score": cr["inner_fold_1"],
                    "inner_fold_2_score": cr["inner_fold_2"],
                    "mean_inner_score": cr["mean_score"],
                    "is_selected": (cr["candidate_id"] == best_cid),
                }
            )

        selected_hyperparam_rows.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_candidate_id": best_cid,
                "feature_dim": X_train_fold.shape[1],
                "params_json": json.dumps(
                    {
                        k: v
                        for k, v in best_params.items()
                        if not k.startswith("is_c")
                    }
                ),
            }
        )

        # Refit selected pipeline on all outer training rows
        pipe = ClassicalPipeline(
            model_type=best_params["model_type"],
            pca_components=best_params["pca_components"],
            use_scaler=best_params["use_scaler"],
            ridge_alpha=best_params["ridge_alpha"],
            xgb_params=best_params["xgb_params"],
        )
        pipe.fit(X_train_fold, y_train_prim)

        # Save fold artifacts
        fold_save_dict = {
            "pipeline": pipe,
            "metadata_preprocessor": meta_preprocessor,
            "best_candidate_id": best_cid,
            "best_params": best_params,
        }
        joblib.dump(
            fold_save_dict, fold_models_dir / f"fold_{outer_fold}_model.joblib"
        )

        # Predict outer validation split exactly once
        raw_val_preds = pipe.predict(X_val_fold)
        derived_val_preds = derive_targets(raw_val_preds)

        fold_weighted_r2, fold_per_target = compute_weighted_r2(
            y_val_5, derived_val_preds
        )

        fold_metric_rows.append(
            {
                "model": model_name,
                "fold": outer_fold,
                "weighted_r2": fold_weighted_r2,
                "r2_green": fold_per_target[0],
                "r2_dead": fold_per_target[1],
                "r2_clover": fold_per_target[2],
                "r2_gdm": fold_per_target[3],
                "r2_total": fold_per_target[4],
            }
        )

        print(
            f"Fold {outer_fold}: Selected {best_cid:25s} | Outer Weighted R² = {fold_weighted_r2:.4f}"
        )

        for i, val_row_idx in enumerate(val_indices):
            row_data = df.iloc[val_row_idx]
            oof_records.append(
                {
                    "model": model_name,
                    "fold": outer_fold,
                    "image_id": row_data["image_id"],
                    "true_Dry_Green_g": y_val_5[i, 0],
                    "true_Dry_Dead_g": y_val_5[i, 1],
                    "true_Dry_Clover_g": y_val_5[i, 2],
                    "true_GDM_g": y_val_5[i, 3],
                    "true_Dry_Total_g": y_val_5[i, 4],
                    "pred_Dry_Green_g": derived_val_preds[i, 0],
                    "pred_Dry_Dead_g": derived_val_preds[i, 1],
                    "pred_Dry_Clover_g": derived_val_preds[i, 2],
                    "pred_GDM_g": derived_val_preds[i, 3],
                    "pred_Dry_Total_g": derived_val_preds[i, 4],
                    "selected_config_id": best_cid,
                }
            )

    oof_df = (
        pd.DataFrame(oof_records)
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    return (
        oof_df,
        fold_metric_rows,
        inner_candidate_rows,
        selected_hyperparam_rows,
    )


def run_metadata_modes_evaluation(
    model_name: str,
    df: pd.DataFrame,
    features_dict: Dict[str, Any],
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Evaluate saved fold models under present, zero, and shuffled metadata counterfactuals."""
    print(
        f"\nEvaluating metadata modes (present, zero, shuffled) for {model_name}..."
    )
    fold_models_dir = output_dir / "fold_models" / model_name

    meta_oof_records = []
    meta_metric_rows = []

    y_5_all = df[ALL_TARGETS].to_numpy(dtype=np.float64)

    for outer_fold in range(5):
        val_mask = df["fold"] == outer_fold
        val_indices = np.where(val_mask)[0]
        df_val = df.iloc[val_indices].reset_index(drop=True)
        y_val_5 = y_5_all[val_indices]

        model_file = fold_models_dir / f"fold_{outer_fold}_model.joblib"
        fold_data = joblib.load(model_file)
        pipe: ClassicalPipeline = fold_data["pipeline"]
        mp: MetadataPreprocessor = fold_data["metadata_preprocessor"]

        for mode in ["present", "zero", "shuffled"]:
            random_state = MASTER_SEED + outer_fold if mode == "shuffled" else None
            meta_val = mp.transform(
                df_val, mode=mode, random_state=random_state
            )

            if model_name in ["C1", "C2"]:
                X_val = meta_val
            elif model_name == "C5":
                img_feats = features_dict["handcrafted_cross"][val_indices]
                X_val = np.hstack([img_feats, meta_val])
            else:
                raise ValueError(
                    f"Model {model_name} does not support metadata modes"
                )

            raw_preds = pipe.predict(X_val)
            derived_preds = derive_targets(raw_preds)

            weighted_r2, per_target = compute_weighted_r2(
                y_val_5, derived_preds
            )

            meta_metric_rows.append(
                {
                    "model": model_name,
                    "mode": mode,
                    "fold": outer_fold,
                    "weighted_r2": weighted_r2,
                    "r2_green": per_target[0],
                    "r2_dead": per_target[1],
                    "r2_clover": per_target[2],
                    "r2_gdm": per_target[3],
                    "r2_total": per_target[4],
                }
            )

            for i, val_row_idx in enumerate(val_indices):
                row_data = df.iloc[val_row_idx]
                meta_oof_records.append(
                    {
                        "model": model_name,
                        "mode": mode,
                        "fold": outer_fold,
                        "image_id": row_data["image_id"],
                        "true_Dry_Green_g": y_val_5[i, 0],
                        "true_Dry_Dead_g": y_val_5[i, 1],
                        "true_Dry_Clover_g": y_val_5[i, 2],
                        "true_GDM_g": y_val_5[i, 3],
                        "true_Dry_Total_g": y_val_5[i, 4],
                        "pred_Dry_Green_g": derived_preds[i, 0],
                        "pred_Dry_Dead_g": derived_preds[i, 1],
                        "pred_Dry_Clover_g": derived_preds[i, 2],
                        "pred_GDM_g": derived_preds[i, 3],
                        "pred_Dry_Total_g": derived_preds[i, 4],
                    }
                )

    meta_oof_df = pd.DataFrame(meta_oof_records)
    return meta_oof_df, meta_metric_rows


def save_configs(output_dir: Path) -> None:
    """Save C1 through C8 configuration JSON files."""
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "C1": {
            "name": "Metadata Ridge",
            "family": "Classical Metadata",
            "inputs": [
                "State",
                "Species",
                "Pre_GSHH_NDVI",
                "Height_Ave_cm",
                "Sampling_Date (month sin/cos)",
            ],
            "preprocessing": (
                "OneHotEncoder(handle_unknown='ignore') for categoricals, "
                "StandardScaler for continuous"
            ),
            "model": "Multioutput Ridge(fit_intercept=True)",
            "alpha_grid": RIDGE_ALPHAS,
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C2": {
            "name": "Metadata XGBoost",
            "family": "Classical Metadata",
            "inputs": [
                "State",
                "Species",
                "Pre_GSHH_NDVI",
                "Height_Ave_cm",
                "Sampling_Date (month sin/cos)",
            ],
            "preprocessing": (
                "OneHotEncoder for categoricals, raw continuous"
            ),
            "model": "3x XGBRegressor (reg:squarederror, hist, cpu)",
            "candidate_configs": list(XGB_CONFIGS.keys()),
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C3": {
            "name": "Handcrafted Image Ridge",
            "family": "Classical Vision",
            "inputs": "652-dim handcrafted cross-view descriptor [L, R, |L-R|, L*R]",
            "preprocessing": (
                "StandardScaler fit on inner/outer training folds"
            ),
            "model": "Multioutput Ridge(fit_intercept=True)",
            "alpha_grid": RIDGE_ALPHAS,
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C4": {
            "name": "Handcrafted Image XGBoost",
            "family": "Classical Vision",
            "inputs": "652-dim handcrafted cross-view descriptor [L, R, |L-R|, L*R]",
            "preprocessing": "Raw handcrafted features",
            "model": "3x XGBRegressor (reg:squarederror, hist, cpu)",
            "candidate_configs": list(XGB_CONFIGS.keys()),
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C5": {
            "name": "Handcrafted Image + Metadata XGBoost",
            "family": "Classical Multimodal",
            "inputs": "652-dim handcrafted image descriptor + fold-fitted C2 metadata block",
            "preprocessing": (
                "Raw image descriptor, OneHotEncoder for metadata categoricals"
            ),
            "model": "3x XGBRegressor (reg:squarederror, hist, cpu)",
            "candidate_configs": list(XGB_CONFIGS.keys()),
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C6": {
            "name": "Frozen DINOv2 Base + Ridge",
            "family": "Foundation Frozen Feature Linear Probe",
            "inputs": "3072-dim cross-view vector from B3 DINOv2-Base cache",
            "preprocessing": "StandardScaler + optional PCA(none, 64, 128)",
            "model": "Multioutput Ridge(fit_intercept=True)",
            "alpha_grid": RIDGE_ALPHAS,
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C7": {
            "name": "Frozen DINOv3 Large + Ridge",
            "family": "Foundation Frozen Feature Linear Probe",
            "inputs": (
                "4096-dim cross-view vector from timm vit_large_patch16_dinov3.lvd1689m"
                " 512x512"
            ),
            "preprocessing": "StandardScaler + optional PCA(none, 64, 128)",
            "model": "Multioutput Ridge(fit_intercept=True)",
            "alpha_grid": RIDGE_ALPHAS,
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
        "C8": {
            "name": "Frozen DINOv3 Large + PCA + XGBoost",
            "family": "Foundation Frozen Feature Nonlinear Regressor",
            "inputs": (
                "4096-dim cross-view vector from timm vit_large_patch16_dinov3.lvd1689m"
                " 512x512"
            ),
            "preprocessing": "StandardScaler + PCA(32, 64, 128)",
            "model": "3x XGBRegressor (reg:squarederror, hist, cpu)",
            "candidate_configs": ["X1", "X2", "X3", "X4"],
            "inner_cv": (
                "3-fold StratifiedKFold on 3 quantile bins of Dry_Total_g"
            ),
            "master_seed": MASTER_SEED,
        },
    }

    for cid, cfg in configs.items():
        with open(configs_dir / f"{cid}.json", "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run C series classical baseline study."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../csiro-biomass",
        help="Path to csiro-biomass dataset directory",
    )
    parser.add_argument(
        "--fold-file",
        type=str,
        default="output/reruns_2026_09_13/folds_seed17.csv",
        help="Path to locked outer fold split CSV",
    )
    parser.add_argument(
        "--b3-cache-dir",
        type=str,
        default="checkpoints/B3_DINOv2_Base_probe",
        help="Path to verified B3 DINOv2 feature cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/classical_baselines_2026_09_15",
        help="Output root for C series artifacts",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        help="List of models to run",
    )
    parser.add_argument(
        "--metadata-modes",
        action="store_true",
        help="Run metadata counterfactual evaluation for C1, C2, and C5",
    )
    parser.add_argument(
        "--skip-dinov3-gpu",
        action="store_true",
        help="Skip DINOv3 GPU extraction if running CPU smoke tests",
    )
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    git_rev = get_git_revision()

    # Step 1: Load and verify dataset and locked outer folds
    print("Verifying and loading dataset and locked folds...")
    df = load_dataset_and_folds(
        args.data_dir, args.fold_file, verify_hashes=True
    )

    data_validation_dict = {
        "dataset_path": str(Path(args.data_dir).resolve()),
        "train_csv_sha256": EXPECTED_TRAIN_SHA256,
        "fold_file_path": str(Path(args.fold_file).resolve()),
        "fold_file_sha256": EXPECTED_FOLDS_SHA256,
        "num_images": len(df),
        "fold_distribution": df["fold"].value_counts().sort_index().to_dict(),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "data_and_fold_validation.json", "w") as f:
        json.dump(data_validation_dict, f, indent=2)

    # Save configs
    save_configs(output_dir)

    features_dict = {}

    # Step 2: Build or load handcrafted feature cache (CPU) if needed by C3, C4, C5
    if any(m in args.models for m in ["C3", "C4", "C5"]):
        print("\nLoading/Building handcrafted image features...")
        left_h, right_h, cross_h, meta_h = build_or_load_handcrafted_cache(
            df, args.data_dir, cache_dir
        )
        features_dict["handcrafted_cross"] = cross_h

    # Step 3: Load DINOv2 B3 cache if needed by C6
    if "C6" in args.models:
        print("\nLoading and verifying B3 DINOv2 Base features...")
        dinov2_cross, dinov2_meta = load_dinov2_b3_cache(df, args.b3_cache_dir)
        features_dict["dinov2_cross"] = dinov2_cross

    # Step 4: Build or load DINOv3 Large cache on GPU if needed by C7, C8
    if any(m in args.models for m in ["C7", "C8"]) and not args.skip_dinov3_gpu:
        print("\nLoading/Building DINOv3 Large feature cache...")
        dinov3_cross, dinov3_meta = build_or_load_dinov3_cache(
            df, args.data_dir, cache_dir, batch_size=8, git_revision=git_rev
        )
        features_dict["dinov3_cross"] = dinov3_cross

    # Run models
    all_oof_dfs = []
    all_fold_metric_rows = []
    all_inner_candidate_rows = []
    all_selected_hyperparam_rows = []

    for m in args.models:
        oof_m, fold_m, inner_m, select_m = run_model_5fold_cv(
            m, df, features_dict, output_dir
        )
        all_oof_dfs.append(oof_m)
        all_fold_metric_rows.extend(fold_m)
        all_inner_candidate_rows.extend(inner_m)
        all_selected_hyperparam_rows.extend(select_m)

    # Combine OOF predictions
    master_oof_df = pd.concat(all_oof_dfs, ignore_index=True)
    master_oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Save inner candidate scores
    inner_df = pd.DataFrame(all_inner_candidate_rows)
    inner_df.to_csv(output_dir / "inner_candidate_scores.csv", index=False)

    # Save selected hyperparameters
    selected_df = pd.DataFrame(all_selected_hyperparam_rows)
    selected_df.to_csv(
        output_dir / "selected_hyperparameters.csv", index=False
    )

    # Save fold metrics
    fold_metrics_df = pd.DataFrame(all_fold_metric_rows)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Compute aggregate metrics and per-target metrics
    agg_rows = []
    per_target_rows = []

    for m in args.models:
        sub_folds = fold_metrics_df[fold_metrics_df["model"] == m]
        mean_fold_r2 = float(sub_folds["weighted_r2"].mean())
        std_fold_r2 = float(sub_folds["weighted_r2"].std(ddof=0))

        # Compute pooled OOF score across all 357 predictions
        sub_oof = master_oof_df[master_oof_df["model"] == m]
        y_true_pooled = sub_oof[
            [f"true_{t}" for t in ALL_TARGETS]
        ].to_numpy()
        y_pred_pooled = sub_oof[
            [f"pred_{t}" for t in ALL_TARGETS]
        ].to_numpy()
        pooled_weighted_r2, pooled_per_target = compute_weighted_r2(
            y_true_pooled, y_pred_pooled
        )

        agg_rows.append(
            {
                "model": m,
                "mean_fold_raw_weighted_r2": mean_fold_r2,
                "std_fold_raw_weighted_r2_ddof0": std_fold_r2,
                "pooled_oof_raw_weighted_r2": pooled_weighted_r2,
                "pooled_green_r2": pooled_per_target[0],
                "pooled_dead_r2": pooled_per_target[1],
                "pooled_clover_r2": pooled_per_target[2],
                "pooled_gdm_r2": pooled_per_target[3],
                "pooled_total_r2": pooled_per_target[4],
            }
        )

        # Target breakdown row
        for t_idx, target_col in enumerate(ALL_TARGETS):
            target_metric_key = f"r2_{target_col.split('_')[1].lower()}"
            if "total" in target_col.lower():
                target_metric_key = "r2_total"
            elif "gdm" in target_col.lower():
                target_metric_key = "r2_gdm"

            fold_scores = sub_folds[target_metric_key].tolist()
            per_target_rows.append(
                {
                    "model": m,
                    "target": target_col,
                    "fold_0": fold_scores[0],
                    "fold_1": fold_scores[1],
                    "fold_2": fold_scores[2],
                    "fold_3": fold_scores[3],
                    "fold_4": fold_scores[4],
                    "mean_fold": float(np.mean(fold_scores)),
                    "std_fold_ddof0": float(np.std(fold_scores, ddof=0)),
                    "pooled_oof": pooled_per_target[t_idx],
                }
            )

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    per_target_df = pd.DataFrame(per_target_rows)
    per_target_df.to_csv(output_dir / "per_target_metrics.csv", index=False)

    # Step 5: Metadata modes evaluation
    if args.metadata_modes:
        meta_models = [m for m in ["C1", "C2", "C5"] if m in args.models]
        if meta_models:
            meta_oof_dfs = []
            meta_metric_rows = []
            for mm in meta_models:
                m_oof, m_metrics = run_metadata_modes_evaluation(
                    mm, df, features_dict, output_dir
                )
                meta_oof_dfs.append(m_oof)
                meta_metric_rows.extend(m_metrics)

            combined_meta_oof = pd.concat(meta_oof_dfs, ignore_index=True)
            combined_meta_oof.to_csv(
                output_dir / "metadata_modes_oof_predictions.csv", index=False
            )

            meta_metrics_df = pd.DataFrame(meta_metric_rows)
            meta_metrics_df.to_csv(
                output_dir / "metadata_modes_metrics.csv", index=False
            )

    # Save cache and model SHA256 manifest
    manifest = {}
    for p in output_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(output_dir))
            manifest[rel] = sha256_file(p)

    with open(output_dir / "sha256_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save run metadata
    run_meta = {
        "master_seed": MASTER_SEED,
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": time.time() - start_time,
        "git_revision": git_rev,
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sys.modules["sklearn"].__version__,
        "xgboost_version": xgb.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "models_executed": args.models,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=======================================================")
    print("C SERIES BASELINE RUN COMPLETE!")
    print(f"Results and checkpoints archived at: {output_dir}")
    print("=======================================================\n")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
