"""单站 OFDM 测距 HDF5 数据集构建与加载测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    SingleBsRangeDataset,
    build_usrp_ofdm_single_bs_range_h5,
    infer_single_bs_range_vlen,
    summarize_usrp_ofdm_single_bs_range_h5,
)
from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    MONO_RANGE_TARGET_CSV_COLUMNS,
    round_mono_range_labels,
)

_VLEN = 8
_N_FRAMES = 3


def _write_binary_frames(path: Path, n_frames: int, *, seed: int, vlen: int = _VLEN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("wb") as f:
        for _ in range(n_frames):
            frame = rng.standard_normal(vlen).astype(np.complex64)
            f.write(frame.tobytes())


def _write_csv(parent: Path, rows: list[dict[str, str]]) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=MONO_RANGE_TARGET_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_experiment_dir(tmp_path: Path) -> Path:
    sessions = [
        {
            "recorded_at_utc": "2026-08-20T06:47:37+00:00",
            "target_range_m": "0.49999999999999994",
            "data_file": "divide_profiles_001",
            "record_max_frames": str(_N_FRAMES),
        },
        {
            "recorded_at_utc": "2026-08-20T06:48:10+00:00",
            "target_range_m": "1.25",
            "data_file": "divide_profiles_002",
            "record_max_frames": str(_N_FRAMES),
        },
    ]
    _write_csv(tmp_path, sessions)
    for idx, row in enumerate(sessions, start=1):
        _write_binary_frames(tmp_path / row["data_file"], _N_FRAMES, seed=idx)
    return tmp_path


def test_build_load_shapes_and_rounded_labels(tmp_path: Path) -> None:
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "dataset.h5"

    build_usrp_ofdm_single_bs_range_h5(
        parent, h5_path, vlen=_VLEN, compression=None, show_progress=False
    )

    ds = SingleBsRangeDataset.load(h5_path)
    assert len(ds) == 2 * _N_FRAMES
    assert ds.profiles.shape == (2 * _N_FRAMES, _VLEN)
    assert ds.target_range.shape == (2 * _N_FRAMES,)
    assert int(ds.attrs["num_sessions"]) == 2
    assert int(ds.attrs["frames_per_session"]) == _N_FRAMES
    assert str(ds.attrs["label_axes"]) == "target_range"

    np.testing.assert_allclose(ds.target_range[0], 0.50, rtol=0.0, atol=1e-12)
    assert int(ds.session_index[0]) == 0
    assert int(ds.frame_index[0]) == 0

    np.testing.assert_allclose(ds.target_range[_N_FRAMES], 1.25, rtol=0.0, atol=1e-12)
    assert int(ds.session_index[_N_FRAMES]) == 1
    assert int(ds.frame_index[_N_FRAMES]) == 0


def test_round_mono_range_labels_writes_two_decimals(tmp_path: Path) -> None:
    parent = _make_experiment_dir(tmp_path)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    round_mono_range_labels(csv_path, ndigits=2)

    with csv_path.open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert rows[0]["target_range_m"] == "0.50"
    assert rows[1]["target_range_m"] == "1.25"


def test_infer_vlen_prefers_16384_when_ambiguous(tmp_path: Path) -> None:
    path = tmp_path / "divide_profiles_001"
    _write_binary_frames(path, 2, seed=1, vlen=16384)
    assert infer_single_bs_range_vlen(path) == 16384


def test_infer_vlen_8192(tmp_path: Path) -> None:
    path = tmp_path / "divide_profiles_001"
    _write_binary_frames(path, 3, seed=1, vlen=8192)
    assert infer_single_bs_range_vlen(path) == 8192


def test_missing_binary_raises(tmp_path: Path) -> None:
    parent = tmp_path / "exp"
    row = {
        "recorded_at_utc": "2026-08-20T06:47:37+00:00",
        "target_range_m": "0.50",
        "data_file": "divide_profiles_001",
        "record_max_frames": "3",
    }
    _write_csv(parent, [row])

    with pytest.raises(FileNotFoundError, match="missing"):
        build_usrp_ofdm_single_bs_range_h5(
            parent,
            tmp_path / "bad.h5",
            vlen=_VLEN,
            compression=None,
            show_progress=False,
        )


def test_summarize_h5_without_full_load(tmp_path: Path) -> None:
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "summary.h5"
    build_usrp_ofdm_single_bs_range_h5(
        parent, h5_path, vlen=_VLEN, compression=None, show_progress=False
    )

    summary = summarize_usrp_ofdm_single_bs_range_h5(h5_path)
    assert summary["total_frames"] == 2 * _N_FRAMES
    assert summary["profiles_shape"] == (2 * _N_FRAMES, _VLEN)
    assert summary["num_sessions"] == 2
    assert summary["frames_per_session"] == _N_FRAMES


def test_getitem_returns_torch_tensors(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    parent = _make_experiment_dir(tmp_path)
    h5_path = tmp_path / "getitem.h5"
    build_usrp_ofdm_single_bs_range_h5(
        parent, h5_path, vlen=_VLEN, compression=None, show_progress=False
    )

    ds = SingleBsRangeDataset.load(h5_path)
    sample0 = ds[0]
    assert sample0["profiles"].shape == (_VLEN,)
    assert float(sample0["target_range"]) == pytest.approx(0.50)
    assert int(sample0["session_index"]) == 0
    assert int(sample0["frame_index"]) == 0
