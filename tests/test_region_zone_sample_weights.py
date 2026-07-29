"""三区域损失样本权重测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "model_training"
    / "run_train_cooperative_monostatic_cnn.py"
)


@pytest.fixture(scope="module")
def train_mod():
    module_name = "run_train_cooperative_monostatic_cnn_zone_weight_test"
    spec = importlib.util.spec_from_file_location(module_name, _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_region_zone_sample_weights_assignment(train_mod) -> None:
    xy = torch.tensor(
        [
            [0.0, 0.0],  # center
            [0.8, 0.0],  # side
            [0.8, 0.8],  # corner
        ],
        dtype=torch.float32,
    )
    weights = train_mod._region_zone_sample_weights(
        xy, center_weight=1.0, side_weight=3.0, corner_weight=2.0
    )
    assert weights is not None
    assert torch.allclose(
        weights, torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)
    )


def test_region_zone_sample_weights_all_ones_returns_none(train_mod) -> None:
    xy = torch.tensor([[0.0, 0.0], [0.8, 0.8]], dtype=torch.float32)
    weights = train_mod._region_zone_sample_weights(
        xy, center_weight=1.0, side_weight=1.0, corner_weight=1.0
    )
    assert weights is None
