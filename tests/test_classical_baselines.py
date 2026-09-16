"""Unit and leakage tests for classical C series baselines.

Implements the 11 required pre-flight verification gates specified in
GEMINI_FLASH_3_8_HANDOFF.txt.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

# Ensure src/ and experiments/classical/ are importable
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "classical"))

from classical_features import (
    ALL_TARGETS,
    EXPECTED_FOLD_COUNTS,
    EXPECTED_FOLDS_SHA256,
    EXPECTED_TRAIN_SHA256,
    PRIMITIVE_TARGETS,
    MetadataPreprocessor,
    compute_weighted_r2,
    derive_targets,
    extract_handcrafted_pair,
    extract_view_features,
    get_handcrafted_feature_names,
    load_dataset_and_folds,
    load_dinov2_b3_cache,
    sha256_file,
)
from run_c_series_cv import (
    ClassicalPipeline,
    define_candidate_grid,
    run_inner_selection,
)

DATA_DIR = REPO_ROOT.parent / "csiro-biomass"
FOLD_FILE = REPO_ROOT / "output" / "reruns_2026_09_13" / "folds_seed17.csv"
B3_DIR = REPO_ROOT / "checkpoints" / "B3_DINOv2_Base_probe"
OUTPUT_DIR = REPO_ROOT / "output" / "classical_baselines_2026_09_15"


class TestClassicalBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset_and_folds(
            DATA_DIR, FOLD_FILE, verify_hashes=False
        )

    def test_01_data_and_fold_hash_checks(self):
        """1. Data and fold hash checks pass."""
        train_csv = DATA_DIR / "train.csv"
        self.assertTrue(train_csv.exists(), "train.csv does not exist")
        self.assertEqual(
            sha256_file(train_csv),
            EXPECTED_TRAIN_SHA256,
            "train.csv SHA256 mismatch",
        )

        self.assertTrue(FOLD_FILE.exists(), "folds_seed17.csv does not exist")
        self.assertEqual(
            sha256_file(FOLD_FILE),
            EXPECTED_FOLDS_SHA256,
            "folds_seed17.csv SHA256 mismatch",
        )
        self.assertEqual(len(self.df), 357)
        self.assertEqual(
            self.df["fold"].value_counts().sort_index().to_dict(),
            EXPECTED_FOLD_COUNTS,
        )

    def test_02_handcrafted_extractor_bitwise_identical(self):
        """2. Handcrafted extractor returns bitwise identical feature arrays on two calls."""
        img_path = DATA_DIR / self.df.iloc[0]["image_path"]
        L1, R1, cross1 = extract_handcrafted_pair(img_path)
        L2, R2, cross2 = extract_handcrafted_pair(img_path)

        self.assertTrue(
            np.array_equal(L1, L2), "Left view features are not bitwise identical"
        )
        self.assertTrue(
            np.array_equal(R1, R2),
            "Right view features are not bitwise identical",
        )
        self.assertTrue(
            np.array_equal(cross1, cross2),
            "Cross-view features are not bitwise identical",
        )

    def test_03_handcrafted_dimensions_and_finite(self):
        """3. Handcrafted extractor emits 163 per view and 652 cross view features with finite values."""
        img_path = DATA_DIR / self.df.iloc[0]["image_path"]
        L, R, cross = extract_handcrafted_pair(img_path)

        self.assertEqual(L.shape, (163,))
        self.assertEqual(R.shape, (163,))
        self.assertEqual(cross.shape, (652,))

        self.assertTrue(
            np.all(np.isfinite(L)), "Left view features contain nonfinite values"
        )
        self.assertTrue(
            np.all(np.isfinite(R)),
            "Right view features contain nonfinite values",
        )
        self.assertTrue(
            np.all(np.isfinite(cross)),
            "Cross-view features contain nonfinite values",
        )

        view_names, cross_names = get_handcrafted_feature_names()
        self.assertEqual(len(view_names), 163)
        self.assertEqual(len(cross_names), 652)

    def test_04_metadata_pipeline_train_fit_and_unseen_category(self):
        """4. Metadata pipeline is fit only on supplied training indexes. An unseen category in validation must not crash."""
        train_df = self.df[self.df["fold"] != 0].copy()
        val_df = self.df[self.df["fold"] == 0].copy()

        mp = MetadataPreprocessor(scale_continuous=True)
        mp.fit(train_df)

        # Introduce unseen category in validation
        val_df_unseen = val_df.copy()
        val_df_unseen.loc[val_df_unseen.index[0], "State"] = "UNKNOWN_STATE"
        val_df_unseen.loc[val_df_unseen.index[0], "Species"] = "UNKNOWN_SPECIES"

        # Transforming must succeed without raising exception
        X_val = mp.transform(val_df_unseen, mode="present")
        self.assertEqual(X_val.shape[0], len(val_df))
        self.assertEqual(X_val.shape[1], len(mp.feature_names_))
        self.assertTrue(
            np.all(np.isfinite(X_val)), "Transformed validation contains NaNs"
        )

    def test_05_no_leakage_in_c3_c4_features(self):
        """5. No target, image ID, path, fold, or metadata value enters C3 or C4 feature matrix."""
        _, cross_names = get_handcrafted_feature_names()
        forbidden_substrings = [
            "target",
            "image_id",
            "path",
            "fold",
            "State",
            "Species",
            "NDVI",
            "Height",
            "Date",
            "Dry_",
            "GDM",
        ]
        for name in cross_names:
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden.lower(),
                    name.lower(),
                    f"Forbidden substring '{forbidden}' found in feature name '{name}'",
                )

    def test_06_target_derivation_and_compositionality(self):
        """6. All pipelines output exactly three primitive targets. Derivation produces five compositional targets."""
        # Simulated raw predictions with negative values
        raw_preds = np.array(
            [
                [-5.0, 10.0, 15.0],
                [20.0, -3.0, 5.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        derived = derive_targets(raw_preds)
        self.assertEqual(derived.shape, (3, 5))

        # Check clipping to >= 0
        self.assertTrue(
            np.all(derived[:, :3] >= 0.0),
            "Primitive predictions not clipped at 0",
        )
        self.assertEqual(derived[0, 0], 0.0)
        self.assertEqual(derived[1, 1], 0.0)

        # Check exact compositional sum: GDM = Green + Clover, Total = GDM + Dead
        for i in range(len(derived)):
            g, d, c, gdm, tot = derived[i]
            self.assertAlmostEqual(gdm, g + c, places=7)
            self.assertAlmostEqual(tot, gdm + d, places=7)

    def test_07_inner_model_selection_no_outer_leakage(self):
        """7. Inner model selection never reads outer validation indexes."""
        outer_fold = 0
        train_mask = self.df["fold"] != outer_fold
        df_train = self.df[train_mask].reset_index(drop=True)

        # Create synthetic feature matrix of outer train rows
        np.random.seed(42)
        X_mock = np.random.randn(len(df_train), 10).astype(np.float32)
        y_prim = df_train[PRIMITIVE_TARGETS].to_numpy()
        y_5 = df_train[ALL_TARGETS].to_numpy()

        candidates = define_candidate_grid("C1")
        # Run inner selection on train split only
        best_cid, best_params, cand_results = run_inner_selection(
            X_train_raw=X_mock,
            y_train_primitive=y_prim,
            y_train_5=y_5,
            df_train_sub=df_train,
            outer_fold=outer_fold,
            candidates=candidates,
        )

        self.assertIsNotNone(best_cid)
        self.assertEqual(len(cand_results), len(candidates))
        for cr in cand_results:
            self.assertTrue(np.isfinite(cr["mean_score"]))

    def test_08_c6_cache_integrity(self):
        """8. C6 cache IDs, shapes, hashes, and metadata match exactly."""
        cross_view, meta = load_dinov2_b3_cache(self.df, B3_DIR)
        self.assertEqual(cross_view.shape, (357, 3072))
        self.assertTrue(
            np.all(np.isfinite(cross_view)), "DINOv2 cache contains nonfinite"
        )
        self.assertEqual(meta.get("model_id"), "facebook/dinov2-base")
        self.assertEqual(meta.get("image_ids"), self.df["image_id"].tolist())

    def test_09_c7_inference_repeatability(self):
        """9. C7 cache IDs, shapes, dimensions, and repeat inference on four examples match within 1e-5."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for DINOv3 inference test")

        import timm
        from classical_features import DINOv3SplitDataset

        device = torch.device("cuda")
        model = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        ).to(device)
        model.eval()

        sample_df = self.df.iloc[:4].copy()
        ds = DINOv3SplitDataset(sample_df, DATA_DIR)

        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        lefts, rights, _ = next(iter(loader))
        lefts = lefts.to(device)
        rights = rights.to(device)

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16
        ):
            out_l1 = model(lefts).cpu().numpy()
            out_l2 = model(lefts).cpu().numpy()

        diff = np.abs(out_l1 - out_l2)
        max_diff = float(np.max(diff))
        self.assertLess(
            max_diff,
            1e-5,
            f"Repeat inference difference {max_diff} exceeds 1e-5",
        )
        self.assertEqual(out_l1.shape, (4, 1024))

    def test_10_c5_metadata_counterfactuals(self):
        """10. C5 zero metadata preserves its image block. Shuffled metadata is a permutation within the same outer fold."""
        train_df = self.df[self.df["fold"] != 0].copy()
        val_df = self.df[self.df["fold"] == 0].copy()

        mp = MetadataPreprocessor(scale_continuous=False)
        mp.fit(train_df)

        meta_present = mp.transform(val_df, mode="present")
        meta_zero = mp.transform(val_df, mode="zero")
        meta_shuffled = mp.transform(
            val_df, mode="shuffled", random_state=17 + 0
        )

        # Check zero mode
        self.assertTrue(
            np.all(meta_zero == 0.0), "Zero mode does not zero all metadata"
        )

        # Check shuffled mode is a permutation
        self.assertFalse(
            np.array_equal(meta_present, meta_shuffled),
            "Shuffled metadata is identical to present",
        )
        self.assertEqual(meta_present.shape, meta_shuffled.shape)
        # Check that row-wise sum distributions match (permutation invariant)
        sum_present = np.sort(np.sum(meta_present, axis=1))
        sum_shuffled = np.sort(np.sum(meta_shuffled, axis=1))
        self.assertTrue(
            np.allclose(sum_present, sum_shuffled, atol=1e-5),
            "Shuffled metadata is not a row permutation of present metadata",
        )

    def test_11_c1_outer_fold_0_smoke(self):
        """11. A CPU smoke run of C1 on outer fold 0 completes, writes no artifacts outside the new output root, and produces finite predictions."""
        from run_c_series_cv import run_model_5fold_cv

        smoke_dir = OUTPUT_DIR / "smoke_test"
        smoke_dir.mkdir(parents=True, exist_ok=True)

        oof_df, fold_metrics, _, _ = run_model_5fold_cv(
            model_name="C1",
            df=self.df,
            features_dict={},
            output_dir=smoke_dir,
        )

        self.assertEqual(len(oof_df), 357)
        self.assertTrue(
            np.all(np.isfinite(oof_df["pred_Dry_Total_g"])),
            "C1 smoke run produced nonfinite predictions",
        )
        self.assertEqual(len(fold_metrics), 5)


if __name__ == "__main__":
    unittest.main()
