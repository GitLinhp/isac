"""Cooperative monostatic ESPRIT RMSE 评估脚本测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from isac_imp.cooperative_monostatic_pipeline import grc_cooperative_processing_params


def _load_eval_module():
    eval_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_esprit_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("esprit_rmse_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="eval_mod")
def fixture_eval_mod():
    return _load_eval_module()


def test_rmse_stats_counts_nan(eval_mod) -> None:
    rmses = np.array([1.0, 2.0, np.nan], dtype=np.float64)
    stats = eval_mod._rmse_stats(rmses)
    assert stats["samples"] == 3
    assert stats["valid"] == 2
    assert stats["nan"] == 1
    assert stats["mean"] == pytest.approx(1.5)


def test_print_summary_inner_outer(capsys, eval_mod) -> None:
    rows = [
        {"true_x_m": 0.0, "true_y_m": 0.0, "rmse_xy_m": 1.0},
        {"true_x_m": 0.2, "true_y_m": 0.0, "rmse_xy_m": 2.0},
        {"true_x_m": 0.8, "true_y_m": 0.0, "rmse_xy_m": 3.0},
        {"true_x_m": 0.0, "true_y_m": 0.8, "rmse_xy_m": float("nan")},
    ]
    eval_mod._print_summary(rows)
    out = capsys.readouterr().out

    assert "ESPRIT localization RMSE summary" in out
    assert "global" in out
    assert "inner (|x|,|y| <= 0.5 m)" in out
    assert "outer" in out


def test_esprit_range_from_divide_cpi_smoke(eval_mod) -> None:
    params = grc_cooperative_processing_params()
    vlen_divide = int(params["vlen_divide_cpi"])
    rng = np.random.default_rng(0)
    divide_cpi = (rng.normal(size=vlen_divide) + 1j * rng.normal(size=vlen_divide)).astype(
        np.complex64
    )

    r_m = eval_mod._esprit_range_from_divide_cpi(
        divide_cpi,
        proc_params=params,
        range_roi=params["range_roi"],
    )

    assert np.isfinite(r_m) or np.isnan(r_m)
