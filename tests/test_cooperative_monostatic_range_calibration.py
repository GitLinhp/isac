"""Cooperative monostatic 距离偏置校准测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isac_imp.cooperative_monostatic_range_calibration import (
    DEFAULT_CALIB_ROI_DEV0,
    DEFAULT_CALIB_ROI_DEV1,
    RangeBiasCalibRoi,
    apply_range_bias,
    biases_from_calib_result,
    calib_roi_mask,
    correct_monostatic_range_pair,
    fit_dual_dev_range_biases,
    fit_range_bias_1d,
    load_range_bias_calib,
    resolve_loaded_range_biases,
    save_range_bias_calib,
    target_in_calib_roi,
    true_quasi_monostatic_range_m,
)

DEV0_XY = (0.0, -2.0)
DEV1_XY = (-2.0, 0.0)
# 测试用共址天线：path_sum/2 退化为中点距离
TX0_XY = RX0_XY = DEV0_XY
TX1_XY = RX1_XY = DEV1_XY


def _fit_kwargs() -> dict:
    return {
        "tx0_xy": TX0_XY,
        "rx0_xy": RX0_XY,
        "tx1_xy": TX1_XY,
        "rx1_xy": RX1_XY,
        "dev0_xy": DEV0_XY,
        "dev1_xy": DEV1_XY,
    }


def test_target_in_calib_roi_defaults() -> None:
    assert target_in_calib_roi(0.0, 0.0, DEFAULT_CALIB_ROI_DEV0) is True
    assert target_in_calib_roi(0.5, 1.0, DEFAULT_CALIB_ROI_DEV0) is True
    assert target_in_calib_roi(0.6, 0.0, DEFAULT_CALIB_ROI_DEV0) is False
    assert target_in_calib_roi(0.0, 0.0, DEFAULT_CALIB_ROI_DEV1) is True
    assert target_in_calib_roi(1.0, 0.4, DEFAULT_CALIB_ROI_DEV1) is True
    assert target_in_calib_roi(0.0, 0.6, DEFAULT_CALIB_ROI_DEV1) is False


def test_fit_range_bias_1d_recovers_injected_bias() -> None:
    r_true = np.array([2.0, 2.5, 3.0], dtype=np.float64)
    bias = 0.07
    r_est = r_true - bias
    mask = np.ones_like(r_true, dtype=bool)

    best_bias, best_mae = fit_range_bias_1d(
        r_est,
        r_true,
        mask,
        search_min=-0.5,
        search_max=0.5,
        step=0.01,
    )

    assert best_bias == pytest.approx(bias, abs=0.011)
    assert best_mae == pytest.approx(0.0, abs=1e-9)


def test_fit_ignores_samples_outside_roi() -> None:
    rows = []
    for x, y, r0_bias in ((0.0, 0.0, 0.10), (0.8, 0.8, 0.90)):
        true_r0 = true_quasi_monostatic_range_m((x, y), TX0_XY, RX0_XY)
        rows.append(
            {
                "true_x_m": x,
                "true_y_m": y,
                "r_dev0_m": true_r0 - r0_bias,
                "r_dev1_m": 2.0,
            }
        )
    df = pd.DataFrame(rows)
    result = fit_dual_dev_range_biases(
        df,
        **_fit_kwargs(),
        roi_dev0=DEFAULT_CALIB_ROI_DEV0,
        roi_dev1=RangeBiasCalibRoi(-5, 5, -5, 5),
        search_min=-1.0,
        search_max=1.0,
        step=0.01,
    )

    assert result.bias_dev0_m == pytest.approx(0.10, abs=0.011)
    assert result.n_dev0 == 1


def test_save_and_load_calib_json(tmp_path: Path) -> None:
    rows = [
        {
            "true_x_m": 0.0,
            "true_y_m": 0.0,
            "r_dev0_m": true_quasi_monostatic_range_m((0.0, 0.0), TX0_XY, RX0_XY) - 0.05,
            "r_dev1_m": true_quasi_monostatic_range_m((0.0, 0.0), TX1_XY, RX1_XY) - 0.03,
        }
    ]
    df = pd.DataFrame(rows)
    result = fit_dual_dev_range_biases(
        df,
        **_fit_kwargs(),
        search_min=-1.0,
        search_max=1.0,
        step=0.01,
    )
    path = tmp_path / "calib.json"
    save_range_bias_calib(path, result)
    loaded = load_range_bias_calib(path)

    assert loaded.bias_dev0_m == pytest.approx(result.bias_dev0_m)
    assert loaded.bias_dev1_m == pytest.approx(result.bias_dev1_m)
    assert loaded.tx0_xy == result.tx0_xy
    assert json.loads(path.read_text(encoding="utf-8"))["n_dev0"] == result.n_dev0


def test_fit_path_sum_truth_differs_from_colocated() -> None:
    """±5 cm TX/RX 时 path_sum/2 与中点距离不同，偏置应相对前者拟合。"""
    from isac_imp.cooperative_monostatic_pipeline import (
        DEFAULT_DEV0_RX_XY,
        DEFAULT_DEV0_TX_XY,
        DEFAULT_DEV1_RX_XY,
        DEFAULT_DEV1_TX_XY,
    )

    target = (0.0, 0.0)
    r_true = true_quasi_monostatic_range_m(
        target, DEFAULT_DEV0_TX_XY, DEFAULT_DEV0_RX_XY
    )
    assert r_true != pytest.approx(2.0, abs=1e-9)
    df = pd.DataFrame(
        [
            {
                "true_x_m": 0.0,
                "true_y_m": 0.0,
                "r_dev0_m": r_true - 0.12,
                "r_dev1_m": true_quasi_monostatic_range_m(
                    target, DEFAULT_DEV1_TX_XY, DEFAULT_DEV1_RX_XY
                )
                - 0.08,
            }
        ]
    )
    result = fit_dual_dev_range_biases(
        df,
        tx0_xy=DEFAULT_DEV0_TX_XY,
        rx0_xy=DEFAULT_DEV0_RX_XY,
        tx1_xy=DEFAULT_DEV1_TX_XY,
        rx1_xy=DEFAULT_DEV1_RX_XY,
        search_min=-1.0,
        search_max=1.0,
        step=0.01,
    )
    assert result.bias_dev0_m == pytest.approx(0.12, abs=0.011)
    assert result.bias_dev1_m == pytest.approx(0.08, abs=0.011)


def test_apply_range_bias_adds_cal_columns() -> None:
    df = pd.DataFrame(
        {
            "r_dev0_m": [2.0],
            "r_dev1_m": [3.0],
        }
    )
    out = apply_range_bias(df, bias_dev0_m=0.1, bias_dev1_m=-0.2)
    assert out.loc[0, "r_dev0_cal_m"] == pytest.approx(2.1)
    assert out.loc[0, "r_dev1_cal_m"] == pytest.approx(2.8)
    assert out.loc[0, "r_dev0_m"] == pytest.approx(2.0)


def test_calib_roi_mask_shape() -> None:
    x = np.array([0.0, 0.6], dtype=np.float64)
    y = np.array([0.0, 0.0], dtype=np.float64)
    mask = calib_roi_mask(x, y, DEFAULT_CALIB_ROI_DEV0)
    assert mask.tolist() == [True, False]


def test_correct_monostatic_range_pair_scalar_and_nan() -> None:
    r0_cal, r1_cal = correct_monostatic_range_pair(
        2.0, 3.0, bias_dev0_m=0.1, bias_dev1_m=-0.2
    )
    assert r0_cal == pytest.approx(2.1)
    assert r1_cal == pytest.approx(2.8)

    r0_nan, r1_nan = correct_monostatic_range_pair(
        float("nan"), 3.0, bias_dev0_m=0.1, bias_dev1_m=-0.2
    )
    assert np.isnan(r0_nan)
    assert r1_nan == pytest.approx(2.8)


def test_resolve_loaded_range_biases(tmp_path: Path) -> None:
    path = tmp_path / "calib.json"
    save_range_bias_calib(
        path,
        fit_dual_dev_range_biases(
            pd.DataFrame(
                [
                    {
                        "true_x_m": 0.0,
                        "true_y_m": 0.0,
                        "r_dev0_m": true_quasi_monostatic_range_m(
                            (0.0, 0.0), TX0_XY, RX0_XY
                        )
                        - 0.05,
                        "r_dev1_m": true_quasi_monostatic_range_m(
                            (0.0, 0.0), TX1_XY, RX1_XY
                        )
                        - 0.03,
                    }
                ]
            ),
            **_fit_kwargs(),
            search_min=-1.0,
            search_max=1.0,
            step=0.01,
        ),
    )
    args = argparse.Namespace(calib_json=path)
    biases = resolve_loaded_range_biases(args)
    loaded = load_range_bias_calib(path)

    assert biases == biases_from_calib_result(loaded)
