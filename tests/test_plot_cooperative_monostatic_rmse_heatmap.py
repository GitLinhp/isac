"""Cooperative monostatic RMSE 热力图绘图测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isac_imp.record_target_metadata import is_inner_target_xy_m


def _load_plot_module():
    plot_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "plot_cooperative_monostatic_music_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="plot_mod")
def fixture_plot_mod():
    return _load_plot_module()


def _synthetic_rmse_csv(path: Path) -> None:
    rows = []
    sample_idx = 0
    for y in (-1.0, -0.8, -0.6):
        for x in (-1.0, -0.8, -0.6):
            for frame in range(2):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "session_index": sample_idx // 2,
                        "frame_index": frame,
                        "true_x_m": x,
                        "true_y_m": y,
                        "est_x_m": x,
                        "est_y_m": y,
                        "r_dev0_m": 1.0,
                        "r_dev1_m": 1.0,
                        "rmse_xy_m": abs(x) + abs(y) + frame * 0.01,
                    }
                )
                sample_idx += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_rmse_grid_shape_and_values(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)
    xs, ys, z = plot_mod.build_rmse_grid(df)

    assert xs.shape == (3,)
    assert ys.shape == (3,)
    assert z.shape == (3, 3)

    x_idx = int(np.where(np.isclose(xs, -1.0))[0][0])
    y_idx = int(np.where(np.isclose(ys, -1.0))[0][0])
    assert z[y_idx, x_idx] == pytest.approx(2.005, abs=1e-6)


def test_plot_rmse_heatmap_from_csv_writes_png(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    png_path = tmp_path / "heatmap.png"
    _synthetic_rmse_csv(csv_path)

    summary = plot_mod.plot_rmse_heatmap_from_csv(csv_path, png_path)

    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert summary["filled_cells"] == 9
    assert summary["total_rows"] == 18


def _synthetic_inner_outer_rmse_csv(path: Path) -> None:
    rows = []
    sample_idx = 0
    points = [
        (0.0, 0.0, 0.5),
        (-1.0, -1.0, 3.0),
        (0.8, -0.6, 2.5),
    ]
    for x, y, rmse in points:
        rows.append(
            {
                "sample_idx": sample_idx,
                "session_index": sample_idx,
                "frame_index": 0,
                "true_x_m": x,
                "true_y_m": y,
                "est_x_m": x,
                "est_y_m": y,
                "r_dev0_m": 1.0,
                "r_dev1_m": 1.0,
                "rmse_xy_m": rmse,
            }
        )
        sample_idx += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_rmse_grid_outer_shape_and_masking(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_inner_outer_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)
    xs, ys, z = plot_mod.build_rmse_grid_outer(df)

    assert xs.shape == (11,)
    assert ys.shape == (11,)
    assert z.shape == (11, 11)
    assert np.allclose(np.diff(xs), plot_mod.OUTER_GRID_STEP_M)

    x0_idx = int(np.where(np.isclose(xs, 0.0))[0][0])
    y0_idx = int(np.where(np.isclose(ys, 0.0))[0][0])
    assert np.isnan(z[y0_idx, x0_idx])
    assert not is_inner_target_xy_m(-1.0, -1.0)
    x_neg_idx = int(np.where(np.isclose(xs, -1.0))[0][0])
    assert z[x_neg_idx, x_neg_idx] == pytest.approx(3.0)


def test_plot_rmse_heatmap_outer_from_csv_writes_png(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    png_path = tmp_path / "heatmap_outer.png"
    _synthetic_inner_outer_rmse_csv(csv_path)

    summary = plot_mod.plot_rmse_heatmap_outer_from_csv(csv_path, png_path)

    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert summary["filled_cells"] == 2
    assert summary["total_rows"] == 3


def test_build_rmse_grid_inner_shape_and_values(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_inner_outer_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)
    xs, ys, z = plot_mod.build_rmse_grid_inner(df)

    assert xs.shape == (11,)
    assert ys.shape == (11,)
    assert z.shape == (11, 11)
    assert np.allclose(np.diff(xs), plot_mod.INNER_GRID_STEP_M)

    x0_idx = int(np.where(np.isclose(xs, 0.0))[0][0])
    y0_idx = int(np.where(np.isclose(ys, 0.0))[0][0])
    assert z[y0_idx, x0_idx] == pytest.approx(0.5)
    assert int(np.isfinite(z).sum()) == 1


def _synthetic_full_inner_grid_csv(path: Path) -> None:
    rows = []
    sample_idx = 0
    axis = [round(-0.5 + i * 0.1, 1) for i in range(11)]
    for y in axis:
        for x in axis:
            rmse = float("nan") if (x, y) == (-0.2, 0.1) else 1.0 + abs(x) + abs(y)
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "session_index": sample_idx,
                    "frame_index": 0,
                    "true_x_m": x,
                    "true_y_m": y,
                    "est_x_m": x,
                    "est_y_m": y,
                    "r_dev0_m": 1.0,
                    "r_dev1_m": 1.0,
                    "rmse_xy_m": rmse,
                }
            )
            sample_idx += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_rmse_grid_inner_fills_full_11x11_except_invalid(
    plot_mod, tmp_path: Path
) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_full_inner_grid_csv(csv_path)
    df = pd.read_csv(csv_path)
    _, _, z = plot_mod.build_rmse_grid_inner(df)

    assert z.shape == (11, 11)
    assert int(np.isfinite(z).sum()) == 120


def test_build_rmse_grid_10cm_shape(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_inner_outer_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)
    xs, ys, z, measured_invalid_mask = plot_mod.build_rmse_grid_10cm(df)

    assert xs.shape == (21,)
    assert ys.shape == (21,)
    assert z.shape == (21, 21)
    assert measured_invalid_mask.shape == (21, 21)
    assert np.allclose(np.diff(xs), plot_mod.UNIFIED_GRID_STEP_M)

    x0_idx = int(np.where(np.isclose(xs, 0.0))[0][0])
    y0_idx = int(np.where(np.isclose(ys, 0.0))[0][0])
    assert z[y0_idx, x0_idx] == pytest.approx(0.5)


def _synthetic_outer_20cm_corners_csv(path: Path) -> None:
    rows = []
    sample_idx = 0
    points = [
        (0.0, 0.0, 1.0),
        (-1.0, -1.0, 4.0),
        (-1.0, 1.0, 3.0),
        (1.0, -1.0, 5.0),
        (1.0, 1.0, 6.0),
    ]
    for x, y, rmse in points:
        rows.append(
            {
                "sample_idx": sample_idx,
                "session_index": sample_idx,
                "frame_index": 0,
                "true_x_m": x,
                "true_y_m": y,
                "est_x_m": x,
                "est_y_m": y,
                "r_dev0_m": 1.0,
                "r_dev1_m": 1.0,
                "rmse_xy_m": rmse,
            }
        )
        sample_idx += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_interpolate_outer_fills_gaps(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_outer_20cm_corners_csv(csv_path)
    df = pd.read_csv(csv_path)
    xs, ys, z, measured_invalid_mask = plot_mod.build_rmse_grid_10cm(df)
    z_interp, interpolated_cells = plot_mod.interpolate_outer_rmse_gaps(
        z,
        xs,
        ys,
        measured_invalid_mask=measured_invalid_mask,
    )

    assert interpolated_cells > 0
    x_idx = int(np.where(np.isclose(xs, -0.9))[0][0])
    y_idx = int(np.where(np.isclose(ys, -0.9))[0][0])
    assert np.isfinite(z_interp[y_idx, x_idx])


def test_interpolate_skips_measured_invalid(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_full_inner_grid_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "sample_idx": 999,
                        "session_index": 999,
                        "frame_index": 0,
                        "true_x_m": -1.0,
                        "true_y_m": -1.0,
                        "est_x_m": -1.0,
                        "est_y_m": -1.0,
                        "r_dev0_m": 1.0,
                        "r_dev1_m": 1.0,
                        "rmse_xy_m": 4.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rows.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    xs, ys, z, measured_invalid_mask = plot_mod.build_rmse_grid_10cm(df)
    z_interp, _ = plot_mod.interpolate_outer_rmse_gaps(
        z,
        xs,
        ys,
        measured_invalid_mask=measured_invalid_mask,
    )

    x_idx = int(np.where(np.isclose(xs, -0.2))[0][0])
    y_idx = int(np.where(np.isclose(ys, 0.1))[0][0])
    assert np.isnan(z_interp[y_idx, x_idx])


def test_plot_rmse_heatmap_combined_from_csv_writes_png(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    png_path = tmp_path / "heatmap_combined.png"
    _synthetic_inner_outer_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)

    _, _, z_interp, interpolated_cells = plot_mod.build_rmse_grid_10cm_interpolated(df)
    expected_filled = int(np.isfinite(z_interp).sum())

    summary = plot_mod.plot_rmse_heatmap_combined_from_csv(csv_path, png_path)

    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert summary["filled_cells"] == expected_filled
    assert summary["interpolated_cells"] == interpolated_cells
    assert summary["interpolated_cells"] > 0
    assert summary["total_rows"] == 3


def test_empirical_cdf_monotonic(plot_mod) -> None:
    values = np.array([1.0, 3.0, 2.0, np.nan], dtype=np.float64)
    x_cdf, y_cdf = plot_mod._empirical_cdf(values)

    assert np.allclose(x_cdf, [1.0, 2.0, 3.0])
    assert np.allclose(y_cdf, [1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert y_cdf[-1] == pytest.approx(1.0)


def test_split_rmse_by_region(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    _synthetic_inner_outer_rmse_csv(csv_path)
    df = pd.read_csv(csv_path)

    by_region = plot_mod._split_rmse_by_region(df)

    assert by_region["global"].size == 3
    assert by_region["inner"].size == 1
    assert by_region["outer"].size == 2
    assert np.isfinite(by_region["global"]).sum() == 3
    assert by_region["inner"][0] == pytest.approx(0.5)


def test_plot_rmse_cdf_from_csv_writes_png(plot_mod, tmp_path: Path) -> None:
    csv_path = tmp_path / "rmse.csv"
    png_path = tmp_path / "rmse_cdf.png"
    _synthetic_inner_outer_rmse_csv(csv_path)

    summary = plot_mod.plot_rmse_cdf_from_csv(
        csv_path,
        png_path,
        title="Test RMSE CDF",
    )

    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert summary["total_rows"] == 3
    assert summary["global_valid"] == 3
    assert summary["inner_valid"] == 1
    assert summary["outer_valid"] == 2
    assert summary["curves_plotted"] == 3

