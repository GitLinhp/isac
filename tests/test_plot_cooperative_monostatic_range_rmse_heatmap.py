"""Cooperative monostatic 双 dev 距离 MAE 热力图绘图测试。"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _load_plot_module():
    plot_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "plot_cooperative_monostatic_range_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_range_mae_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plot_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="plot_mod")
def fixture_plot_mod():
    return _load_plot_module()


DEV0_XY = (0.0, -2.0)
DEV1_XY = (-2.0, 0.0)


def _synthetic_range_df() -> pd.DataFrame:
    rows = []
    sample_idx = 0
    points = [
        (0.0, 0.0, 2.1, 1.9),
        (-1.0, -1.0, 3.0, 2.0),
        (0.8, -0.6, 2.5, 2.8),
    ]
    for x, y, r0, r1 in points:
        rows.append(
            {
                "sample_idx": sample_idx,
                "session_index": sample_idx,
                "frame_index": 0,
                "true_x_m": x,
                "true_y_m": y,
                "r_dev0_m": r0,
                "r_dev1_m": r1,
            }
        )
        sample_idx += 1
    return pd.DataFrame(rows)


def _synthetic_range_csv(path: Path) -> None:
    df = _synthetic_range_df()
    df["est_x_m"] = df["true_x_m"]
    df["est_y_m"] = df["true_y_m"]
    df["rmse_xy_m"] = 0.5
    df.to_csv(path, index=False)


def test_true_monostatic_range_m(plot_mod) -> None:
    assert plot_mod.true_monostatic_range_m((0.0, 0.0), DEV0_XY) == pytest.approx(2.0)
    assert plot_mod.true_monostatic_range_m((0.0, 0.0), DEV1_XY) == pytest.approx(2.0)
    assert plot_mod.true_monostatic_range_m((-1.0, -1.0), DEV0_XY) == pytest.approx(
        math.hypot(-1.0, 1.0)
    )


def test_add_per_dev_range_abs_errors(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    _synthetic_range_csv(csv_path)
    df = pd.read_csv(csv_path)
    out = plot_mod.add_per_dev_range_abs_errors(
        df, dev0_xy=DEV0_XY, dev1_xy=DEV1_XY
    )

    assert out.loc[0, plot_mod.ABS_ERR_COL_DEV0] == pytest.approx(0.1)
    assert out.loc[0, plot_mod.ABS_ERR_COL_DEV1] == pytest.approx(0.1)
    true_r0 = plot_mod.true_monostatic_range_m((-1.0, -1.0), DEV0_XY)
    assert out.loc[1, plot_mod.ABS_ERR_COL_DEV0] == pytest.approx(abs(3.0 - true_r0))


def test_build_range_mae_grid_10cm(plot_mod, tmp_path: Path) -> None:
    df = plot_mod.add_per_dev_range_abs_errors(
        _synthetic_range_df(), dev0_xy=DEV0_XY, dev1_xy=DEV1_XY
    )
    heatmap_mod = plot_mod._load_heatmap_module()
    xs, ys, z0, _ = heatmap_mod.build_rmse_grid_10cm_interpolated(
        df, value_col=plot_mod.ABS_ERR_COL_DEV0
    )

    x0_idx = int(np.where(np.isclose(xs, 0.0))[0][0])
    y0_idx = int(np.where(np.isclose(ys, 0.0))[0][0])
    assert z0[y0_idx, x0_idx] == pytest.approx(0.1)
    assert xs.shape == (21,)
    assert ys.shape == (21,)
    assert z0.shape == (21, 21)


def test_from_df_matches_from_csv(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    png_csv = tmp_path / "from_csv.png"
    png_df = tmp_path / "from_df.png"
    _synthetic_range_csv(csv_path)
    df = pd.read_csv(csv_path)

    summary_csv = plot_mod.plot_range_mae_heatmap_dual_dev_from_csv(
        csv_path,
        png_csv,
        method="music",
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
    )
    summary_df = plot_mod.plot_range_mae_heatmap_dual_dev_from_df(
        df,
        png_df,
        method="music",
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
        data_source=str(csv_path),
    )

    assert png_csv.is_file()
    assert png_df.is_file()
    assert summary_csv["total_rows"] == summary_df["total_rows"]
    assert summary_csv["mean_mae_dev0_m"] == pytest.approx(summary_df["mean_mae_dev0_m"])
    assert summary_csv["mean_mae_dev1_m"] == pytest.approx(summary_df["mean_mae_dev1_m"])


def test_plot_range_mae_heatmap_dual_dev_writes_png(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    png_path = tmp_path / "range_mae_heatmap_dev.png"
    _synthetic_range_csv(csv_path)

    summary = plot_mod.plot_range_mae_heatmap_dual_dev_from_csv(
        csv_path,
        png_path,
        method="music",
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
    )

    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert summary["total_rows"] == 3
    assert summary["filled_cells_dev0"] > 0
    assert summary["filled_cells_dev1"] > 0
    assert np.isfinite(summary["mean_mae_dev0_m"])
    assert np.isfinite(summary["mean_mae_dev1_m"])


def test_cli_rejects_h5_and_csv_together(plot_mod, tmp_path: Path) -> None:
    with patch.object(
        sys,
        "argv",
        [
            "plot_range_mae_heatmap",
            "--h5-path",
            str(tmp_path / "data.h5"),
            "--input-csv",
            str(tmp_path / "data.csv"),
        ],
    ):
        with pytest.raises(SystemExit):
            plot_mod.argument_parser()


def test_evaluate_range_estimates_to_dataframe_mock(plot_mod, tmp_path: Path) -> None:
    h5_path = tmp_path / "fake.h5"
    h5_path.touch()
    mock_rows = [
        {
            "sample_idx": 0,
            "session_index": 0,
            "frame_index": 0,
            "true_x_m": 0.0,
            "true_y_m": 0.0,
            "r_dev0_m": 2.1,
            "r_dev1_m": 1.9,
        }
    ]

    with patch.object(
        plot_mod,
        "_load_eval_module",
    ) as load_eval:
        eval_mod = load_eval.return_value
        eval_mod._evaluate_per_frame.return_value = mock_rows
        df = plot_mod.evaluate_range_estimates_to_dataframe(
            h5_path,
            method="music",
            options=plot_mod.RangeEvalOptions(show_progress=False),
        )

    assert len(df) == 1
    assert df.loc[0, "r_dev0_m"] == pytest.approx(2.1)
    eval_mod._evaluate_per_frame.assert_called_once()


def test_main_h5_path_mock(plot_mod, tmp_path: Path) -> None:
    h5_path = tmp_path / "dataset.h5"
    h5_path.touch()
    out_png = tmp_path / "out.png"
    mock_df = _synthetic_range_df()

    with patch.object(
        plot_mod,
        "evaluate_range_estimates_to_dataframe",
        return_value=mock_df,
    ) as evaluate:
        with patch.object(
            sys,
            "argv",
            [
                "plot_range_mae_heatmap",
                "--h5-path",
                str(h5_path),
                "--method",
                "esprit",
                "--output-png",
                str(out_png),
                "--no-progress",
            ],
        ):
            plot_mod.main()

    evaluate.assert_called_once()
    assert out_png.is_file()


def test_main_calibrate_range_mock(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "range.csv"
    out_png = tmp_path / "calib_out.png"
    _synthetic_range_csv(csv_path)

    from isac_imp.cooperative_monostatic_range_calibration import (
        DEFAULT_CALIB_ROI_DEV0,
        DEFAULT_CALIB_ROI_DEV1,
        RangeBiasCalibResult,
        apply_range_bias,
    )

    mock_result = RangeBiasCalibResult(
        bias_dev0_m=0.05,
        bias_dev1_m=-0.03,
        mae_dev0_m=0.1,
        mae_dev1_m=0.08,
        n_dev0=10,
        n_dev1=10,
        roi_dev0=DEFAULT_CALIB_ROI_DEV0,
        roi_dev1=DEFAULT_CALIB_ROI_DEV1,
        search_min_m=-2.0,
        search_max_m=2.0,
        step_m=0.01,
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
    )

    with patch.object(
        plot_mod,
        "resolve_range_bias_calibration",
        return_value=(apply_range_bias(
            pd.read_csv(csv_path), bias_dev0_m=0.05, bias_dev1_m=-0.03
        ), mock_result),
    ):
        with patch.object(
            sys,
            "argv",
            [
                "plot_range_mae_heatmap",
                "--input-csv",
                str(csv_path),
                "--output-png",
                str(out_png),
                "--calibrate-range",
            ],
        ):
            plot_mod.main()

    assert out_png.is_file()


def test_default_output_png_includes_h5_parent_tag(plot_mod, tmp_path: Path) -> None:
    h5_path = tmp_path / "cooperative_monostatic_measurement0" / "dataset.h5"
    out = plot_mod._default_output_png("music", h5_path=h5_path)
    assert "cooperative_monostatic_measurement0" in out.name
