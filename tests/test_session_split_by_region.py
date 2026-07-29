"""九宫格区域 train/val 划分测试。"""

from __future__ import annotations

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
from isac_imp.record_target_metadata import (
    target_region_index_xy_m,
    target_region_name,
    target_zone_name_xy_m,
)

_VLEN = 32768

# (x_m, y_m) -> region name
_REGION_CASES = [
    ((0.0, 0.0), "C"),
    ((0.5, 0.5), "C"),
    ((0.8, 0.0), "E"),
    ((-0.8, 0.0), "W"),
    ((0.0, 0.8), "N"),
    ((0.0, -0.8), "S"),
    ((-0.8, 0.8), "NW"),
    ((0.8, 0.8), "NE"),
    ((-0.8, -0.8), "SW"),
    ((0.8, -0.8), "SE"),
]

# (x_m, y_m) -> zone name (center / side / corner)
_ZONE_CASES = [
    ((0.0, 0.0), "center"),
    ((0.5, 0.5), "center"),
    ((0.8, 0.0), "side"),
    ((-0.8, 0.0), "side"),
    ((0.0, 0.8), "side"),
    ((0.0, -0.8), "side"),
    ((-0.8, 0.8), "corner"),
    ((0.8, 0.8), "corner"),
    ((-0.8, -0.8), "corner"),
    ((0.8, -0.8), "corner"),
]


@pytest.mark.parametrize(("xy", "name"), _REGION_CASES)
def test_target_region_index_xy_m(xy: tuple[float, float], name: str) -> None:
    region_id = target_region_index_xy_m(xy[0], xy[1])
    assert target_region_name(region_id) == name


@pytest.mark.parametrize(("xy", "zone"), _ZONE_CASES)
def test_target_zone_name_xy_m(xy: tuple[float, float], zone: str) -> None:
    assert target_zone_name_xy_m(xy[0], xy[1]) == zone


def _region_xy_m(region_name: str) -> tuple[float, float]:
    mapping = {
        "SW": (-0.8, -0.8),
        "W": (-0.8, 0.0),
        "NW": (-0.8, 0.8),
        "S": (0.0, -0.8),
        "C": (0.0, 0.0),
        "N": (0.0, 0.8),
        "SE": (0.8, -0.8),
        "E": (0.8, 0.0),
        "NE": (0.8, 0.8),
    }
    return mapping[region_name]


def _write_region_h5(
    path: Path,
    *,
    sessions_per_region: dict[str, int],
    frames_per_session: int = 2,
) -> None:
    session_specs: list[tuple[int, tuple[float, float]]] = []
    session_index = 0
    for region_name, count in sessions_per_region.items():
        xy = _region_xy_m(region_name)
        for _ in range(count):
            session_specs.append((session_index, xy))
            session_index += 1

    total = len(session_specs) * frames_per_session
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
        session_index_arr = np.zeros(total, dtype=np.int32)
        frame_index_arr = np.zeros(total, dtype=np.int32)
        for sess_idx, (sess, (x_m, y_m)) in enumerate(session_specs):
            start = sess_idx * frames_per_session
            end = start + frames_per_session
            target[start:end, 0] = x_m
            target[start:end, 1] = y_m
            session_index_arr[start:end] = sess
            frame_index_arr[start:end] = np.arange(frames_per_session, dtype=np.int32)
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        f.create_dataset(DATASET_KEY_SESSION_INDEX, data=session_index_arr)
        f.create_dataset(DATASET_KEY_FRAME_INDEX, data=frame_index_arr)


def test_session_train_val_split_by_region_no_leak(tmp_path: Path) -> None:
    h5_path = tmp_path / "coop.h5"
    sessions_per_region = {name: 4 for name in ("SW", "W", "NW", "S", "C", "N", "SE", "E", "NE")}
    _write_region_h5(h5_path, sessions_per_region=sessions_per_region)

    with h5py.File(h5_path, "r") as f:
        session_indices = f[DATASET_KEY_SESSION_INDEX][:]
        target_position = f[DATASET_KEY_TARGET_POSITION][:]

    train_idx, val_idx, split_info = session_train_val_split_by_region(
        session_indices,
        target_position,
        0.2,
        seed=7,
    )
    train_sessions = set(session_indices[train_idx].tolist())
    val_sessions = set(session_indices[val_idx].tolist())
    assert train_sessions.isdisjoint(val_sessions)
    assert train_idx.size + val_idx.size == session_indices.size
    assert sum(info["val"] for info in split_info.values()) == len(val_sessions)
    for info in split_info.values():
        if info["train"] + info["val"] > 0:
            assert info["train"] + info["val"] >= 1


def test_session_train_val_split_single_session_region_all_train(tmp_path: Path) -> None:
    h5_path = tmp_path / "coop.h5"
    _write_region_h5(
        h5_path,
        sessions_per_region={"C": 1, "E": 4},
        frames_per_session=3,
    )
    with h5py.File(h5_path, "r") as f:
        session_indices = f[DATASET_KEY_SESSION_INDEX][:]
        target_position = f[DATASET_KEY_TARGET_POSITION][:]

    train_idx, val_idx, split_info = session_train_val_split_by_region(
        session_indices,
        target_position,
        0.2,
        seed=0,
    )
    assert split_info[4] == {"train": 1, "val": 0}
    center_sessions = {0}
    assert center_sessions.isdisjoint(set(session_indices[val_idx].tolist()))
    assert center_sessions.issubset(set(session_indices[train_idx].tolist()))


def test_session_frames_stay_in_same_split(tmp_path: Path) -> None:
    h5_path = tmp_path / "coop.h5"
    _write_region_h5(h5_path, sessions_per_region={"E": 3}, frames_per_session=5)
    with h5py.File(h5_path, "r") as f:
        session_indices = f[DATASET_KEY_SESSION_INDEX][:]
        target_position = f[DATASET_KEY_TARGET_POSITION][:]

    train_idx, val_idx, _ = session_train_val_split_by_region(
        session_indices,
        target_position,
        0.34,
        seed=1,
    )
    for split_indices in (train_idx, val_idx):
        for sess in np.unique(session_indices[split_indices]):
            all_frames = np.where(session_indices == sess)[0]
            split_frames = split_indices[session_indices[split_indices] == sess]
            np.testing.assert_array_equal(np.sort(all_frames), np.sort(split_frames))
