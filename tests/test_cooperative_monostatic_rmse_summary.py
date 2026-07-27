"""Cooperative monostatic RMSE 内外侧汇总测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from isac_imp.record_target_metadata import is_inner_target_xy_m


def _load_eval_module():
    eval_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_music_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("music_rmse_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="eval_mod")
def fixture_eval_mod():
    return _load_eval_module()


def test_is_inner_target_xy_m_boundary() -> None:
    assert is_inner_target_xy_m(0.5, 0.0) is True
    assert is_inner_target_xy_m(0.0, -0.5) is True
    assert is_inner_target_xy_m(0.6, 0.0) is False
    assert is_inner_target_xy_m(0.0, 0.6) is False
    assert is_inner_target_xy_m(-0.5, 0.5) is True


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

    assert "MUSIC localization RMSE summary" in out
    assert "global" in out
    assert "inner (|x|,|y| <= 0.5 m)" in out
    assert "outer" in out
    assert out.index("inner (|x|,|y| <= 0.5 m)") < out.index("outer")

    inner_pos = out.index("inner (|x|,|y| <= 0.5 m)")
    outer_pos = out.index("outer")
    inner_block = out[inner_pos:outer_pos]
    outer_block = out[outer_pos:]
    assert "2" in inner_block
    assert "2" in outer_block
    assert "1" in outer_block
