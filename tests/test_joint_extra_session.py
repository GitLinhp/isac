"""Joint 训练脚本 extra-session 辅助函数测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_joint_train():
    path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "model_training"
        / "run_train_cooperative_monostatic_two_stage_joint.py"
    )
    spec = importlib.util.spec_from_file_location("joint_train_session_helpers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sample_sessions_by_frac_partition() -> None:
    mod = _load_joint_train()
    sessions = np.repeat(np.arange(100, dtype=np.int64), 5)
    aug, hold = mod.sample_sessions_by_frac(sessions, frac=0.1, seed=42)
    assert len(aug) == 10
    assert len(hold) == 90
    assert set(aug.tolist()).isdisjoint(set(hold.tolist()))
    assert set(aug.tolist()) | set(hold.tolist()) == set(range(100))


def test_frames_for_sessions() -> None:
    mod = _load_joint_train()
    session_indices = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    frames = mod.frames_for_sessions(session_indices, [2])
    assert frames.tolist() == [3, 4, 5]


def test_read_write_session_list(tmp_path: Path) -> None:
    mod = _load_joint_train()
    path = tmp_path / "sessions.txt"
    mod._write_session_list(path, [3, 7, 11])
    assert mod._read_session_list(path) == [3, 7, 11]


def test_extra_h5_cli_flags_present() -> None:
    mod = _load_joint_train()
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "--extra-h5",
            "data/extra.h5",
            "--extra-session-list",
            "split/aug.txt",
            "--extra-val-session-list",
            "split/val.txt",
            "--extra-session-frac",
            "0.1",
            "--extra-session-seed",
            "7",
            "--early-stop-on",
            "extra_val",
            "--freeze-fine",
            "--extra-oversample",
            "3",
        ]
    )
    assert args.extra_h5.name == "extra.h5"
    assert args.extra_session_list.name == "aug.txt"
    assert args.extra_val_session_list.name == "val.txt"
    assert args.extra_session_frac == 0.1
    assert args.extra_session_seed == 7
    assert args.early_stop_on == "extra_val"
    assert args.freeze_fine is True
    assert args.extra_oversample == 3


def test_split_session_list_train_val() -> None:
    mod = _load_joint_train()
    sessions = list(range(22))
    train, val = mod.split_session_list_train_val(sessions, n_val=4, seed=43)
    assert len(val) == 4
    assert len(train) == 18
    assert set(train.tolist()).isdisjoint(set(val.tolist()))
    assert set(train.tolist()) | set(val.tolist()) == set(sessions)


def test_split_session_list_train_val_rejects_bad_n() -> None:
    mod = _load_joint_train()
    try:
        mod.split_session_list_train_val([1, 2, 3], n_val=3, seed=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_extra_oversample_cli_rejects_zero() -> None:
    mod = _load_joint_train()
    parser = mod._build_arg_parser()
    args = parser.parse_args(["--extra-oversample", "0"])
    assert args.extra_oversample == 0
