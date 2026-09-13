import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frozen_probe import DualViewFrozenProbe, ProbeConfig, load_fold_data, weighted_r2
from models import BiomassModelTimm


def test_probe_outputs_are_nonnegative_and_compositional():
    model = DualViewFrozenProbe(input_dim=32, projection_dim=16, dropout=0.0).eval()
    output = model(torch.randn(4, 32), torch.randn(4, 32))
    assert output.shape == (4, 5)
    assert torch.all(output >= 0)
    torch.testing.assert_close(output[:, 3], output[:, 0] + output[:, 2])
    torch.testing.assert_close(output[:, 4], output[:, 3] + output[:, 1])


def test_weighted_r2_is_one_for_exact_predictions():
    targets = np.arange(25, dtype=np.float32).reshape(5, 5)
    score, per_target = weighted_r2(targets, targets.copy())
    assert score == 1.0
    np.testing.assert_array_equal(per_target, np.ones(5))


def test_fold_file_is_complete(tmp_path):
    data_dir = ROOT.parent / "csiro-biomass"
    config = ProbeConfig(
        run_name="test",
        model_id="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "output"),
        fold_file=str(tmp_path / "folds.csv"),
    )
    frame = load_fold_data(config)
    assert len(frame) == 357
    assert sorted(frame["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert frame.groupby("fold").size().sum() == 357


def test_backbone_adapter_accepts_spatial_and_vector_features():
    spatial = torch.randn(2, 8, 4, 4)
    vector = torch.randn(2, 8)
    assert BiomassModelTimm._as_tokens(spatial).shape == (2, 16, 8)
    assert BiomassModelTimm._as_tokens(vector).shape == (2, 1, 8)
