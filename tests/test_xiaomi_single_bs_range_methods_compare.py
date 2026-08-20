"""单站测距 methods_compare 指标与脚本辅助函数测试。"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from isac.xiaomi_models import (
    SingleBsRangeCNN,
    load_single_bs_range_cnn_checkpoint,
    save_single_bs_range_cnn_checkpoint,
)


def _load_compare_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "xiaomi_model_training"
        / "run_usrp_ofdm_single_bs_range_methods_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        "xiaomi_single_bs_range_methods_compare", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rmse_mae_from_preds_all_valid() -> None:
    mod = _load_compare_module()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 4.0])
    metrics = mod.rmse_mae_from_preds(pred, target)
    assert metrics["n_total"] == 3
    assert metrics["n_valid"] == 3
    assert metrics["fail_rate"] == 0.0
    assert metrics["mae_m"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse_m"] == pytest.approx(np.sqrt(1.0 / 3.0))
    # abs_err = [0, 0, 1]
    assert metrics["min_abs_err_m"] == pytest.approx(0.0)
    assert metrics["max_abs_err_m"] == pytest.approx(1.0)
    assert metrics["var_abs_err_m"] == pytest.approx(np.var([0.0, 0.0, 1.0]))


def test_rmse_mae_from_preds_with_nan_failures() -> None:
    mod = _load_compare_module()
    pred = np.array([1.0, np.nan, 3.0])
    target = np.array([1.0, 2.0, 3.5])
    metrics = mod.rmse_mae_from_preds(pred, target)
    assert metrics["n_total"] == 3
    assert metrics["n_valid"] == 2
    assert metrics["fail_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["mae_m"] == pytest.approx(0.25)
    assert metrics["rmse_m"] == pytest.approx(np.sqrt(0.25 / 2))
    # abs_err = [0, 0.5]
    assert metrics["min_abs_err_m"] == pytest.approx(0.0)
    assert metrics["max_abs_err_m"] == pytest.approx(0.5)
    assert metrics["var_abs_err_m"] == pytest.approx(np.var([0.0, 0.5]))


def test_rmse_mae_all_failed() -> None:
    mod = _load_compare_module()
    pred = np.array([np.nan, np.nan])
    target = np.array([1.0, 2.0])
    metrics = mod.rmse_mae_from_preds(pred, target)
    assert metrics["n_valid"] == 0
    assert metrics["fail_rate"] == 1.0
    assert np.isnan(metrics["rmse_m"])
    assert np.isnan(metrics["mae_m"])
    assert np.isnan(metrics["min_abs_err_m"])
    assert np.isnan(metrics["max_abs_err_m"])
    assert np.isnan(metrics["var_abs_err_m"])


def test_parse_methods_order_and_unique() -> None:
    mod = _load_compare_module()
    assert mod._parse_methods("cnn,music,cnn,esprit") == ["cnn", "music", "esprit"]


def test_write_summary_csv(tmp_path: Path) -> None:
    mod = _load_compare_module()
    summary_path = tmp_path / "summary.csv"
    rows = [
        {
            "method": "music",
            "n_valid": 2,
            "fail_rate": 0.0,
            "rmse_m": 0.1,
            "mae_m": 0.05,
            "min_abs_err_m": 0.01,
            "max_abs_err_m": 0.09,
            "var_abs_err_m": 0.001,
            "n_total": 2,
        }
    ]
    mod._write_summary_csv(summary_path, rows)
    with summary_path.open(encoding="utf-8") as csv_f:
        loaded = list(csv.DictReader(csv_f))
    assert loaded[0]["method"] == "music"
    assert loaded[0]["n_valid"] == "2"
    assert loaded[0]["rmse_m"] == "0.100"
    assert loaded[0]["mae_m"] == "0.050"
    assert loaded[0]["min_abs_err_m"] == "0.010"
    assert loaded[0]["max_abs_err_m"] == "0.090"
    assert loaded[0]["var_abs_err_m"] == "0.001"


def test_cnn_checkpoint_roundtrip_for_compare(tmp_path: Path) -> None:
    model = SingleBsRangeCNN(in_channels=2, base_channels=8, num_layers=2, dropout=0.0)
    ckpt_path = tmp_path / "ckpt.pth"
    save_single_bs_range_cnn_checkpoint(
        ckpt_path,
        model,
        feature_mode="real_imag",
        range_roi=(0.0, 4.0),
        range_bin_step=0.15,
    )
    loaded_model, ckpt = load_single_bs_range_cnn_checkpoint(
        ckpt_path, map_location="cpu"
    )
    assert int(ckpt["in_channels"]) == 2
    y = loaded_model(torch.randn(2, 2, 32))
    assert y.shape == (2,)
