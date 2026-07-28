"""Cooperative monostatic CNN RMSE 评估脚本测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from isac.models import CooperativeMonostaticCNN
from isac_imp.cooperative_monostatic_pipeline import DEFAULT_RANGE_ROI
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
)

_VLEN = 32768


def _load_eval_module():
    eval_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_cnn_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("cnn_rmse_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_h5(path: Path, *, n_sessions: int = 2, frames_per_session: int = 2) -> None:
    total = n_sessions * frames_per_session
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV0,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV1,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        target = np.zeros((total, 3), dtype=np.float64)
        for s in range(n_sessions):
            start = s * frames_per_session
            end = start + frames_per_session
            target[start:end, 0] = s * 0.1
            target[start:end, 1] = s * 0.2
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        session_index = np.repeat(
            np.arange(n_sessions, dtype=np.int32),
            frames_per_session,
        )
        f.create_dataset(DATASET_KEY_SESSION_INDEX, data=session_index)
        frame_index = np.tile(
            np.arange(frames_per_session, dtype=np.int32),
            n_sessions,
        )
        f.create_dataset(DATASET_KEY_FRAME_INDEX, data=frame_index)


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

    assert "CNN localization RMSE summary" in out
    assert "global" in out
    assert "inner (|x|,|y| <= 0.5 m)" in out
    assert "outer" in out
    assert out.index("inner (|x|,|y| <= 0.5 m)") < out.index("outer")


def test_evaluate_per_frame_smoke(tmp_path: Path, eval_mod) -> None:
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=2, frames_per_session=2)

    model = CooperativeMonostaticCNN(in_channels=4)
    model.eval()
    rows = eval_mod._evaluate_per_frame(
        h5_path,
        model,
        "cpu",
        proc_params=__import__(
            "isac_imp.cooperative_monostatic_pipeline",
            fromlist=["grc_cooperative_processing_params"],
        ).grc_cooperative_processing_params(),
        range_roi=DEFAULT_RANGE_ROI,
        frame_indices=[0, 1, 2, 3],
        batch_size=2,
        show_progress=False,
    )
    assert len(rows) == 4
    for row in rows:
        assert np.isfinite(row["est_x_m"])
        assert np.isfinite(row["est_y_m"])
        assert np.isfinite(row["rmse_xy_m"])


def test_evaluate_aggregate_session(eval_mod) -> None:
    per_frame = [
        {
            "session_index": 0,
            "frame_index": 0,
            "true_x_m": 0.0,
            "true_y_m": 0.0,
            "est_x_m": 0.1,
            "est_y_m": 0.0,
            "rmse_xy_m": 0.1,
        },
        {
            "session_index": 0,
            "frame_index": 1,
            "true_x_m": 0.0,
            "true_y_m": 0.0,
            "est_x_m": 0.3,
            "est_y_m": 0.0,
            "rmse_xy_m": 0.3,
        },
    ]
    rows = eval_mod._evaluate_aggregate_session(per_frame)
    assert len(rows) == 1
    assert rows[0]["session_index"] == 0
    assert rows[0]["est_x_m"] == pytest.approx(0.2)
    assert rows[0]["rmse_xy_m"] == pytest.approx(0.2)
