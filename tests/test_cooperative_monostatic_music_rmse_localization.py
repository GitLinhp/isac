"""MUSIC/ESPRIT 定位与距离偏置修正集成测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from isac_imp.cooperative_monostatic_pipeline import (
    localize_xy_from_two_ranges,
    localize_xy_from_two_ranges_with_bias,
)

DEV0_XY = (0.0, -2.0)
DEV1_XY = (-2.0, 0.0)
TRUE_XY = (0.0, 0.0)


def _load_music_rmse_module():
    plot_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_music_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("music_rmse", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="music_mod")
def fixture_music_mod():
    return _load_music_rmse_module()


def test_localize_with_bias_matches_manual_calibrated_ranges() -> None:
    r0_raw = 2.05
    r1_raw = 2.03
    bias0, bias1 = -0.07, -0.09
    r0_cal = r0_raw + bias0
    r1_cal = r1_raw + bias1

    est_bias = localize_xy_from_two_ranges_with_bias(
        DEV0_XY,
        r0_raw,
        DEV1_XY,
        r1_raw,
        bias_dev0_m=bias0,
        bias_dev1_m=bias1,
    )
    est_manual = localize_xy_from_two_ranges(
        DEV0_XY,
        r0_cal,
        DEV1_XY,
        r1_cal,
    )

    assert est_bias[0] == pytest.approx(est_manual[0])
    assert est_bias[1] == pytest.approx(est_manual[1])


def test_localize_sample_applies_bias(music_mod) -> None:
    r0_raw = 2.05
    r1_raw = 2.03
    bias0, bias1 = -0.07, -0.09

    est_raw_x, est_raw_y, _ = music_mod._localize_sample(
        r0_raw,
        r1_raw,
        TRUE_XY,
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
    )
    est_cal_x, est_cal_y, _ = music_mod._localize_sample(
        r0_raw,
        r1_raw,
        TRUE_XY,
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
        bias_dev0_m=bias0,
        bias_dev1_m=bias1,
    )

    assert est_cal_x != pytest.approx(est_raw_x, abs=1e-6) or est_cal_y != pytest.approx(
        est_raw_y, abs=1e-6
    )


def test_build_eval_row_writes_cal_columns(music_mod) -> None:
    row = music_mod._build_eval_row(
        sample_idx=0,
        session_index=0,
        frame_index=0,
        true_x=0.0,
        true_y=0.0,
        r0=2.0,
        r1=2.5,
        dev0_xy=DEV0_XY,
        dev1_xy=DEV1_XY,
        range_biases=(-0.1, 0.05),
    )

    assert row["r_dev0_m"] == pytest.approx(2.0)
    assert row["r_dev1_m"] == pytest.approx(2.5)
    assert row["r_dev0_cal_m"] == pytest.approx(1.9)
    assert row["r_dev1_cal_m"] == pytest.approx(2.55)
