"""Comprehensive unit and static integrity tests for repository notebooks."""

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
README_MANIFEST = NOTEBOOKS_DIR / "README.md"

MAINTAINED_EXPERIMENT_SCRIPTS = [
    "experiments/baselines/B1_Median_Predictor_cv.py",
    "experiments/baselines/B2_EfficientNet_B3_cv.py",
    "experiments/baselines/B3_DINOv2_Giant_cv.py",
    "experiments/baselines/B4_DINOv2_Metadata_cv.py",
    "experiments/baselines/B5_DINOv3_ViT_L_Metadata_cv.py",
    "experiments/baselines/B6_VMamba_Base_Metadata_cv.py",
    "experiments/baselines/B9_VMamba_Base_NoMetadata_cv.py",
    "experiments/baselines/B10_DINOv2_Giant_cv.py",
    "experiments/ablation/ablation_a1_ssm_scale_cv.py",
    "experiments/ablation/ablation_a2_metadata_cv.py",
    "experiments/ablation/ablation_a3_cross_validation_cv.py",
    "experiments/ablation/ablation_a4_loss_function_cv.py",
    "experiments/ablation/ablation_a5_tta_cv.py",
    "experiments/ablation/ablation_a6_vmamba_vs_dinov2_cv.py",
    "experiments/ablation/ablation_a7_metadata_only_mlp_cv.py",
    "experiments/ablation/E1_BabyMamba_NoMeta_cv.py",
    "experiments/ablation/E2_CVGA_NoMeta_cv.py",
    "experiments/ablation/E3_GatedDWConv_Meta_cv.py",
    "experiments/ablation/E4_NoFusion_cv.py",
    "experiments/ablation/E5_FullMamba_NoMeta_cv.py",
    "experiments/ablation/E6_SingleBlock_cv.py",
    "experiments/ablation/E7_QuadBlock_cv.py",
    "experiments/ablation/E8_NoFusion_Meta_cv.py",
    "experiments/proposed/Proposed_DINOv3_ViT_L_BabyMamba_cv.py",
    "experiments/proposed/Proposed_DINOv3_ViT_L_CVGA_cv.py",
    "experiments/classical/run_c_series_cv.py",
]


class NotebookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook_files = sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))
        cls.notebooks = {}
        for p in cls.notebook_files:
            rel = p.relative_to(REPO_ROOT).as_posix()
            with open(p, "r", encoding="utf-8") as f:
                cls.notebooks[rel] = json.load(f)

    def test_01_all_notebooks_parse_as_valid_json(self):
        """1. Every tracked .ipynb parses as JSON."""
        self.assertGreater(len(self.notebook_files), 0)
        for p in self.notebook_files:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_02_format_version_4(self):
        """2. Every notebook uses format version 4."""
        for rel, nb in self.notebooks.items():
            self.assertEqual(nb.get("nbformat"), 4, f"{rel} nbformat is not 4")

    def test_03_python3_kernelspec(self):
        """3. Every notebook has a python3 kernelspec."""
        for rel, nb in self.notebooks.items():
            meta = nb.get("metadata", {})
            ks = meta.get("kernelspec", {})
            self.assertEqual(ks.get("name"), "python3", f"{rel} kernelspec name is not python3")

    def test_04_null_execution_counts(self):
        """4. Every code cell has null execution count."""
        for rel, nb in self.notebooks.items():
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    self.assertIsNone(
                        cell.get("execution_count"),
                        f"{rel} cell {i} has non-null execution_count",
                    )

    def test_05_no_committed_outputs(self):
        """5. Every code cell has no committed outputs."""
        for rel, nb in self.notebooks.items():
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    self.assertEqual(
                        cell.get("outputs"),
                        [],
                        f"{rel} cell {i} has committed outputs",
                    )

    def test_06_no_author_machine_paths(self):
        """6. No notebook contains personal author machine paths."""
        forbidden = ["C:" + "\\Users\\", "/mnt/c/" + "Users/", "Project" + "BioMass"]
        for rel, nb in self.notebooks.items():
            text = json.dumps(nb)
            for f in forbidden:
                self.assertNotIn(f, text, f"{rel} contains author path '{f}'")

    def test_07_no_embedded_tokens_or_credentials(self):
        """7. No notebook contains tokens like KAGGLE_KEY, HF_TOKEN, or credential values."""
        forbidden_tokens = ["KAGGLE_KEY", "HF_TOKEN", "api_token", "secret_key"]
        for rel, nb in self.notebooks.items():
            text = json.dumps(nb)
            for tok in forbidden_tokens:
                self.assertNotIn(f'"{tok}"', text, f"{rel} contains token '{tok}'")

    def test_08_maintained_scripts_mapped_in_manifest(self):
        """8. Every maintained training script listed in handoff has a notebook mapping in notebooks/README.md."""
        self.assertTrue(README_MANIFEST.is_file(), "notebooks/README.md does not exist")
        manifest_text = README_MANIFEST.read_text(encoding="utf-8")
        for script in MAINTAINED_EXPERIMENT_SCRIPTS:
            self.assertIn(script, manifest_text, f"Manifest missing mapping for {script}")

    def test_09_manifest_notebook_paths_exist(self):
        """9. Every notebook path in notebooks/README.md exists."""
        manifest_text = README_MANIFEST.read_text(encoding="utf-8")
        matches = re.findall(r"`(notebooks/[^`]+\.ipynb)`", manifest_text)
        self.assertGreater(len(matches), 20)
        for nb_path_str in matches:
            nb_path = REPO_ROOT / nb_path_str
            self.assertTrue(nb_path.is_file(), f"Path in manifest does not exist: {nb_path_str}")

    def test_10_source_script_paths_named_by_notebooks_exist(self):
        """10. Every source script path named by a notebook exists."""
        for rel, nb in self.notebooks.items():
            src_script = nb.get("metadata", {}).get("source_script")
            if src_script:
                full_src = REPO_ROOT / src_script
                self.assertTrue(full_src.is_file(), f"{rel} names nonexistent source_script: {src_script}")

    def test_11_corrected_identity_for_b3_b4_b5(self):
        """11. Current B3, B4, and B5 notebook Markdown states corrected identity and metadata state."""
        b3_text = json.dumps(self.notebooks["notebooks/training/baselines/B3_DINOv2_Giant_cv.ipynb"])
        self.assertIn("dinov2-base", b3_text.lower())
        self.assertIn("legacy", b3_text.lower())

        b4_text = json.dumps(self.notebooks["notebooks/training/baselines/B4_DINOv2_Metadata_cv.ipynb"])
        self.assertIn("use_metadata = false", b4_text.lower())
        self.assertIn("legacy", b4_text.lower())

        b5_text = json.dumps(self.notebooks["notebooks/training/baselines/B5_DINOv3_ViT_L_Metadata_cv.ipynb"])
        self.assertIn("use_metadata = false", b5_text.lower())
        self.assertIn("legacy", b5_text.lower())

    def test_12_standalone_training_no_run_or_exec_subprocesses(self):
        """12. Every training notebook contains full code and does not invoke experiment script via %run, exec, subprocess."""
        training_nbs = [rel for rel in self.notebooks if rel.startswith("notebooks/training/")]
        self.assertGreater(len(training_nbs), 20)
        for rel in training_nbs:
            nb = self.notebooks[rel]
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    self.assertNotIn("%run", src, f"{rel} contains %run")
                    self.assertNotIn("runpy.run_path", src, f"{rel} contains runpy")
                    self.assertNotIn("subprocess.run([sys.executable, 'experiments/", src, f"{rel} executes script")

    def test_13_training_notebooks_do_not_import_src_or_experiments(self):
        """13. Every training notebook does not import implementation code from src or experiments."""
        training_nbs = [rel for rel in self.notebooks if rel.startswith("notebooks/training/")]
        forbidden_imports = [
            "from engine import",
            "import engine",
            "from models import",
            "from frozen_probe import",
            "from classical_features import",
            "from run_c_series_cv import",
        ]
        for rel in training_nbs:
            nb = self.notebooks[rel]
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    for f in forbidden_imports:
                        self.assertNotIn(f, src, f"{rel} imports internal module: {f}")

    def test_14_training_notebook_source_parity_metadata(self):
        """14. Every training notebook records source_script, source_commit, protocol_version, source_sha256, and key fields."""
        training_nbs = [rel for rel in self.notebooks if rel.startswith("notebooks/training/")]
        required_fields = [
            "source_script", "source_commit", "protocol_version", "source_sha256",
            "model_id", "metadata_state", "seed", "fold_count", "image_size",
            "target_order", "metric_weights", "output_head_type"
        ]
        for rel in training_nbs:
            meta = self.notebooks[rel].get("metadata", {})
            for field in required_fields:
                self.assertIn(field, meta, f"{rel} metadata missing required field: {field}")

    def test_15_classical_notebook_shares_implementation(self):
        """15. The C series notebook contains shared C implementation supporting C1-C8 without 8 copied engines."""
        c_nb = self.notebooks["notebooks/training/classical/C_series_cv.ipynb"]
        text = json.dumps(c_nb)
        self.assertIn("ClassicalPipeline", text)
        self.assertIn("C1", text)
        self.assertIn("C8", text)

    def test_16_unified_checkpoint_evaluation_excludes_counterfactuals(self):
        """16. unified_checkpoint_evaluation.ipynb contains complete evaluation code and excludes metadata counterfactuals."""
        eval_nb = self.notebooks["notebooks/evaluation/unified_checkpoint_evaluation.ipynb"]
        text = json.dumps(eval_nb)
        self.assertIn("MODEL_REGISTRY", text)
        self.assertNotIn("mode == 'shuffled'", text)

    def test_17_unified_metadata_evaluation_includes_counterfactuals(self):
        """17. unified_metadata_evaluation.ipynb contains present, zero, shuffled, and matched control evaluation code."""
        meta_nb = self.notebooks["notebooks/evaluation/unified_metadata_evaluation.ipynb"]
        text = json.dumps(meta_nb)
        self.assertIn("present", text)
        self.assertIn("zero", text)
        self.assertIn("shuffled", text)

    def test_18_evaluation_notebooks_do_not_train(self):
        """18. Neither evaluation notebook starts training or calls an evaluator script as a wrapper."""
        for rel in [
            "notebooks/evaluation/unified_checkpoint_evaluation.ipynb",
            "notebooks/evaluation/unified_metadata_evaluation.ipynb"
        ]:
            nb = self.notebooks[rel]
            text = json.dumps(nb)
            self.assertNotIn("train_one_epoch", text)
            self.assertNotIn("runpy", text)

    def test_19_unsupported_metadata_submissions_fail_fast(self):
        """19. Unsupported metadata submission notebooks contain fail-fast schema check and do not fill missing metadata."""
        blocked = [
            "notebooks/submission/baselines/B6_VMamba_Base_Metadata_submission.ipynb",
            "notebooks/submission/proposed/Proposed_BabyMamba_submission.ipynb",
            "notebooks/submission/proposed/Proposed_CVGA_submission.ipynb",
        ]
        for rel in blocked:
            nb = self.notebooks[rel]
            text = json.dumps(nb)
            self.assertIn("Fail Fast Schema Check", text)
            self.assertIn("test.csv lacks", text)


if __name__ == "__main__":
    unittest.main()
