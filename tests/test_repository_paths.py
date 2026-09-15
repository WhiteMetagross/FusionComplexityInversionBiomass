"""Repository path smoke tests that require only the Python standard library."""

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_DIRS = {
    ".git",
    "checkpoints",
    "csiro-biomass",
    "output",
    "pretrained",
    "results",
    "third_party",
    "zips",
}


class RepositoryPathTests(unittest.TestCase):
    def test_shared_experiment_imports_resolve_to_src(self):
        scripts = []
        for path in (REPO_ROOT / "experiments").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "from engine import" in source:
                scripts.append(path)
                self.assertIn("parents[2] / 'src'", source, path)

        self.assertEqual(len(scripts), 21)
        self.assertTrue((REPO_ROOT / "src" / "engine.py").is_file())
        self.assertTrue((REPO_ROOT / "src" / "models.py").is_file())

    def test_no_author_machine_or_retired_tree_paths(self):
        legacy_markers = (
            "Project" + "BioMass",
            "local_" + "training_cv",
            "/mnt/c/" + "Users/",
            "C:" + "\\Users\\",
        )
        checked_suffixes = {".md", ".py", ".sh"}

        for path in REPO_ROOT.rglob("*"):
            if LOCAL_ARTIFACT_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            if path.suffix not in checked_suffixes:
                continue
            source = path.read_text(encoding="utf-8")
            for marker in legacy_markers:
                self.assertNotIn(marker, source, path)

    def test_expected_runtime_directories_are_consistent(self):
        setup = (REPO_ROOT / "src" / "utils" / "setup_deps.sh").read_text(
            encoding="utf-8"
        )
        models = (REPO_ROOT / "src" / "models.py").read_text(encoding="utf-8")

        self.assertIn('PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"', setup)
        self.assertIn('CACHE_DIR="$PROJ_ROOT/pretrained"', setup)
        self.assertIn("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", models)


if __name__ == "__main__":
    unittest.main()
