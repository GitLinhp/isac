"""Run1 full + Run2 10% val / holdout split helpers."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    session_train_val_split_by_region,
)

_VLEN = 64
_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = (
    _ROOT / "script/model_training/run_train_cooperative_monostatic_cnn.py"
)
_EVAL_SCRIPT = _ROOT / "script/experiment/run_cooperative_monostatic_cnn_rmse.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="train_mod")
def fixture_train_mod():
    return _load_module(_TRAIN_SCRIPT, "run_train_coop_cnn_tv_test")


@pytest.fixture(name="eval_mod")
def fixture_eval_mod():
    return _load_module(_EVAL_SCRIPT, "cnn_rmse_eval_tv_test")


def _write_region_h5(path: Path, *, n_regions: int = 9, sessions_per_region: int = 4) -> int:
    """Synthetic H5 with one target per session on a 3x3 grid."""
    frames_per_session = 2
    centers = [
        (-0.8, -0.8),
        (0.0, -0.8),
        (0.8, -0.8),
        (-0.8, 0.0),
        (0.0, 0.0),
        (0.8, 0.0),
        (-0.8, 0.8),
        (0.0, 0.8),
        (0.8, 0.8),
    ]
    sessions: list[tuple[float, float]] = []
    for r in range(n_regions):
        x, y = centers[r]
        for _ in range(sessions_per_region):
            sessions.append((x, y))
    n_sessions = len(sessions)
    total = n_sessions * frames_per_session
    rng = np.random.default_rng(0)
    target = np.zeros((total, 3), dtype=np.float64)
    session_index = np.zeros(total, dtype=np.int32)
    frame_index = np.zeros(total, dtype=np.int32)
    for s, (x, y) in enumerate(sessions):
        start = s * frames_per_session
        end = start + frames_per_session
        target[start:end, 0] = x
        target[start:end, 1] = y
        session_index[start:end] = s
        frame_index[start:end] = np.arange(frames_per_session, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV0,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV1,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        f.create_dataset(DATASET_KEY_SESSION_INDEX, data=session_index)
        f.create_dataset(DATASET_KEY_FRAME_INDEX, data=frame_index)
    return total


def test_test_val_ratio_cli_defaults(train_mod) -> None:
    args = train_mod._build_arg_parser().parse_args([])
    assert args.test_val_ratio is None
    assert args.use_test_h5 is False


def test_test_val_ratio_requires_use_test_h5(train_mod) -> None:
    args = argparse.Namespace(test_val_ratio=0.1)
    with pytest.raises(ValueError, match="--use-test-h5"):
        train_mod._validate_test_val_ratio_args(args, use_external_test=False)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_test_val_ratio_illegal(train_mod, bad: float) -> None:
    args = argparse.Namespace(test_val_ratio=bad)
    with pytest.raises(ValueError, match="0 < ratio < 1"):
        train_mod._validate_test_val_ratio_args(args, use_external_test=True)


def test_test_val_ratio_ok(train_mod) -> None:
    args = argparse.Namespace(test_val_ratio=0.1)
    train_mod._validate_test_val_ratio_args(args, use_external_test=True)


def test_eval_train_split_only_complement(tmp_path: Path, eval_mod) -> None:
    h5_path = tmp_path / "run2_synth.h5"
    total = _write_region_h5(h5_path)
    val_idx = eval_mod._resolve_frame_indices(
        h5_path,
        max_samples=None,
        session_index=None,
        val_only=True,
        train_split_only=False,
        val_ratio=0.1,
        seed=42,
    )
    hold_idx = eval_mod._resolve_frame_indices(
        h5_path,
        max_samples=None,
        session_index=None,
        val_only=False,
        train_split_only=True,
        val_ratio=0.1,
        seed=42,
    )
    assert set(val_idx).isdisjoint(hold_idx)
    assert len(val_idx) + len(hold_idx) == total
    assert 0 < len(val_idx) < total
    # ~10% sessions → frames should be well below half
    assert len(val_idx) < total * 0.35


def test_eval_val_and_train_split_mutex(eval_mod, tmp_path: Path) -> None:
    h5_path = tmp_path / "run2_synth.h5"
    _write_region_h5(h5_path)
    with pytest.raises(ValueError, match="不能同时"):
        eval_mod._resolve_frame_indices(
            h5_path,
            max_samples=None,
            session_index=None,
            val_only=True,
            train_split_only=True,
            val_ratio=0.1,
            seed=42,
        )


def test_external_val_subset_disjoint_from_holdout(tmp_path: Path) -> None:
    """Mirrors training use-test-h5 + test-val-ratio=0.1 selection."""
    h5_path = tmp_path / "run2_synth.h5"
    _write_region_h5(h5_path)
    with h5py.File(h5_path, "r") as f:
        sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        targets = np.asarray(f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64)
    hold_idx, val_idx, _ = session_train_val_split_by_region(
        sessions, targets, 0.1, seed=42
    )
    assert set(hold_idx.tolist()).isdisjoint(set(val_idx.tolist()))
    assert len(val_idx) < len(hold_idx)
