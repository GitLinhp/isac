"""Run2 session 抽样辅助函数测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_region_train():
    path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "model_training"
        / "run_train_cooperative_monostatic_region_cnn.py"
    )
    spec = importlib.util.spec_from_file_location("region_train_session_helpers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec for dataclasses on 3.13
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sample_sessions_by_frac_partition() -> None:
    mod = _load_region_train()
    sessions = np.repeat(np.arange(100, dtype=np.int64), 5)
    aug, hold = mod.sample_sessions_by_frac(sessions, frac=0.1, seed=42)
    assert len(aug) == 10
    assert len(hold) == 90
    assert set(aug.tolist()).isdisjoint(set(hold.tolist()))
    assert set(aug.tolist()) | set(hold.tolist()) == set(range(100))


def test_frames_for_sessions() -> None:
    mod = _load_region_train()
    session_indices = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    frames = mod.frames_for_sessions(session_indices, [2])
    assert frames.tolist() == [3, 4, 5]
