"""Cooperative monostatic HDF5 数据集构建与加载测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    CooperativeMonostaticDataset,
    _read_divide_cpi_file,
    build_cooperative_monostatic_h5,
    summarize_cooperative_monostatic_h5,
)
from isac_imp.record_target_metadata import COOPERATIVE_TARGET_CSV, CSV_COLUMNS

_VLEN = 8
_N_FRAMES = 3


def _write_binary_frames(path: Path, n_frames: int, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("wb") as f:
        for _ in range(n_frames):
            frame = rng.standard_normal(_VLEN).astype(np.complex64)
            f.write(frame.tobytes())


def _write_csv(parent: Path, rows: list[dict[str, str]]) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_experiment_dir(tmp_path: Path) -> Path:
    sessions = [
        {
            "recorded_at_utc": "2026-07-26T10:00:00+00:00",
            "target_x_cm": "100.0",
            "target_y_cm": "-50.0",
            "dev0_file": "dev0/divide_profiles_001",
            "dev1_file": "dev1/divide_profiles_001",
            "record_max_frames": str(_N_FRAMES),
        },
        {
            "recorded_at_utc": "2026-07-26T10:01:00+00:00",
            "target_x_cm": "-20.0",
            "target_y_cm": "30.0",
            "dev0_file": "dev0/divide_profiles_002",
            "dev1_file": "dev1/divide_profiles_002",
            "record_max_frames": str(_N_FRAMES),
        },
    ]
    _write_csv(tmp_path, sessions)
    for idx, row in enumerate(sessions, start=1):
        _write_binary_frames(tmp_path / row["dev0_file"], _N_FRAMES, seed=idx)
        _write_binary_frames(tmp_path / row["dev1_file"], _N_FRAMES, seed=idx + 100)
    return tmp_path


def test_build_load_shapes_and_labels(tmp_path: Path) -> None:
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "dataset.h5"

    build_cooperative_monostatic_h5(
        parent, h5_path, vlen=_VLEN, compression=None, target_z_m=0.0, show_progress=False
    )

    ds = CooperativeMonostaticDataset.load(h5_path)
    assert len(ds) == 2 * _N_FRAMES
    assert ds.profiles_dev0.shape == (2 * _N_FRAMES, _VLEN)
    assert ds.profiles_dev1.shape == (2 * _N_FRAMES, _VLEN)
    assert ds.target_position.shape == (2 * _N_FRAMES, 3)
    assert int(ds.attrs["num_sessions"]) == 2
    assert int(ds.attrs["frames_per_session"]) == _N_FRAMES

    np.testing.assert_allclose(ds.target_position[0], [1.0, -0.5, 0.0], rtol=0.0, atol=1e-6)
    assert int(ds.session_index[0]) == 0
    assert int(ds.frame_index[0]) == 0

    np.testing.assert_allclose(
        ds.target_position[_N_FRAMES],
        [-0.2, 0.3, 0.0],
        rtol=0.0,
        atol=1e-6,
    )
    assert int(ds.session_index[_N_FRAMES]) == 1
    assert int(ds.frame_index[_N_FRAMES]) == 0


def test_frame_count_mismatch_raises(tmp_path: Path) -> None:
    parent = tmp_path / "exp"
    row = {
        "recorded_at_utc": "2026-07-26T10:00:00+00:00",
        "target_x_cm": "0.0",
        "target_y_cm": "0.0",
        "dev0_file": "dev0/divide_profiles_001",
        "dev1_file": "dev1/divide_profiles_001",
        "record_max_frames": "3",
    }
    _write_csv(parent, [row])
    _write_binary_frames(parent / row["dev0_file"], _N_FRAMES, seed=1)
    _write_binary_frames(parent / row["dev1_file"], _N_FRAMES - 1, seed=2)

    with pytest.raises(ValueError, match="frame count mismatch"):
        build_cooperative_monostatic_h5(
            parent,
            tmp_path / "bad.h5",
            vlen=_VLEN,
            compression=None,
            show_progress=False,
        )


def test_read_divide_cpi_file_bulk_shape(tmp_path: Path) -> None:
    path = tmp_path / "dev0" / "divide_profiles_001"
    _write_binary_frames(path, _N_FRAMES, seed=7)
    block = _read_divide_cpi_file(path, vlen=_VLEN)
    assert block.shape == (_N_FRAMES, _VLEN)
    assert block.dtype == np.complex64


def test_summarize_h5_without_full_load(tmp_path: Path) -> None:
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "summary.h5"
    build_cooperative_monostatic_h5(
        parent, h5_path, vlen=_VLEN, compression=None, show_progress=False
    )

    summary = summarize_cooperative_monostatic_h5(h5_path)
    assert summary["total_frames"] == 2 * _N_FRAMES
    assert summary["profiles_dev0_shape"] == (2 * _N_FRAMES, _VLEN)
    assert summary["num_sessions"] == 2
    assert summary["frames_per_session"] == _N_FRAMES


def test_getitem_returns_torch_tensors(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "getitem.h5"
    build_cooperative_monostatic_h5(parent, h5_path, vlen=_VLEN, compression=None, show_progress=False)

    ds = CooperativeMonostaticDataset.load(h5_path)
    sample0 = ds[0]
    assert sample0["profiles_dev0"].shape == (_VLEN,)
    assert sample0["profiles_dev1"].shape == (_VLEN,)
    assert sample0["target_position"].shape == (3,)
    np.testing.assert_allclose(
        sample0["target_position"].numpy(),
        [1.0, -0.5, 0.0],
        rtol=0.0,
        atol=1e-6,
    )
    assert int(sample0["session_index"]) == 0
    assert int(sample0["frame_index"]) == 0
