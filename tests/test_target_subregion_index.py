"""0.5 m 间隔 4×4=16 子区域索引与局部坐标测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isac_imp.record_target_metadata import (
    SUBREGION_CELL_SIZE_M,
    SUBREGION_CORNER_IDS,
    SUBREGION_COUNT,
    SUBREGION_GRID_MAX_M,
    SUBREGION_GRID_MIN_M,
    SUBREGION_GRID_N,
    global_xy_from_subregion_local,
    is_subregion_corner_xy_m,
    target_local_offset_xy_m,
    target_subregion_center_xy_m,
    target_subregion_index_xy_m,
)


def test_subregion_constants() -> None:
    assert SUBREGION_CELL_SIZE_M == 0.5
    assert SUBREGION_GRID_MIN_M == -1.0
    assert SUBREGION_GRID_MAX_M == 1.0
    assert SUBREGION_GRID_N == 4
    assert SUBREGION_COUNT == 16
    assert SUBREGION_CORNER_IDS == frozenset({0, 3, 12, 15})


def test_is_subregion_corner_xy_m() -> None:
    for sid in (0, 3, 12, 15):
        cx, cy = target_subregion_center_xy_m(sid)
        assert is_subregion_corner_xy_m(cx, cy)
    for sid in (1, 5, 10, 14):
        cx, cy = target_subregion_center_xy_m(sid)
        assert not is_subregion_corner_xy_m(cx, cy)

# (x_m, y_m) -> expected subregion_id
# 轴区间: [-1,-0.5)/[-0.5,0)/[0,0.5)/[0.5,1]；中心 ±0.75, ±0.25
_INDEX_CASES = [
    # 各格中心
    ((-0.75, -0.75), 0),
    ((-0.25, -0.75), 1),
    ((0.25, -0.75), 2),
    ((0.75, -0.75), 3),
    ((-0.75, -0.25), 4),
    ((-0.25, -0.25), 5),
    ((0.25, -0.25), 6),
    ((0.75, -0.25), 7),
    ((-0.75, 0.25), 8),
    ((-0.25, 0.25), 9),
    ((0.25, 0.25), 10),
    ((0.75, 0.25), 11),
    ((-0.75, 0.75), 12),
    ((-0.25, 0.75), 13),
    ((0.25, 0.75), 14),
    ((0.75, 0.75), 15),
    # 边界：左闭右开；上边界 clip 到最后一格
    ((-1.0, -1.0), 0),
    ((-0.5, -1.0), 1),
    ((-0.5001, -1.0), 0),
    ((0.0, -1.0), 2),
    ((0.5, -1.0), 3),
    ((1.0, 1.0), 15),
    ((0.4999, 0.4999), 10),  # 仍在 [0, 0.5) × [0, 0.5)
]


@pytest.mark.parametrize(("xy", "expected"), _INDEX_CASES)
def test_target_subregion_index_xy_m(
    xy: tuple[float, float], expected: int
) -> None:
    assert target_subregion_index_xy_m(xy[0], xy[1]) == expected


def test_boundary_axis_bands() -> None:
    """单轴区间 [-1,-0.5)/[-0.5,0)/[0,0.5)/[0.5,1.0]。"""
    assert target_subregion_index_xy_m(-1.0, -0.75) == 0
    assert target_subregion_index_xy_m(-0.51, -0.75) == 0
    assert target_subregion_index_xy_m(-0.5, -0.75) == 1
    assert target_subregion_index_xy_m(-0.01, -0.75) == 1
    assert target_subregion_index_xy_m(0.0, -0.75) == 2
    assert target_subregion_index_xy_m(0.49, -0.75) == 2
    assert target_subregion_index_xy_m(0.5, -0.75) == 3
    assert target_subregion_index_xy_m(1.0, -0.75) == 3


def test_target_subregion_center_xy_m() -> None:
    assert target_subregion_center_xy_m(0) == pytest.approx((-0.75, -0.75))
    assert target_subregion_center_xy_m(5) == pytest.approx((-0.25, -0.25))
    assert target_subregion_center_xy_m(10) == pytest.approx((0.25, 0.25))
    assert target_subregion_center_xy_m(15) == pytest.approx((0.75, 0.75))
    with pytest.raises(ValueError, match="subregion_id"):
        target_subregion_center_xy_m(16)
    with pytest.raises(ValueError, match="subregion_id"):
        target_subregion_center_xy_m(-1)


def test_local_offset_and_roundtrip() -> None:
    x_m, y_m = 0.15, -0.35
    sid = target_subregion_index_xy_m(x_m, y_m)
    dx, dy = target_local_offset_xy_m(x_m, y_m, sid)
    gx, gy = global_xy_from_subregion_local(sid, dx, dy)
    assert gx == pytest.approx(x_m)
    assert gy == pytest.approx(y_m)
    half = SUBREGION_CELL_SIZE_M / 2.0
    assert abs(dx) <= half + 1e-9
    assert abs(dy) <= half + 1e-9


def test_local_offset_auto_infer_id() -> None:
    # (-0.25, -0.25) 是 id=5 的中心
    dx, dy = target_local_offset_xy_m(-0.25, -0.25)
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


@pytest.mark.parametrize("sid", range(SUBREGION_COUNT))
def test_every_cell_center_maps_to_itself(sid: int) -> None:
    cx, cy = target_subregion_center_xy_m(sid)
    assert target_subregion_index_xy_m(cx, cy) == sid
    assert math.isfinite(cx) and math.isfinite(cy)


def test_session_train_val_split_by_subregion_no_leak() -> None:
    from isac_imp.data_collection.cooperative_monostatic_dataset import (
        session_train_val_split_by_subregion,
    )

    sessions: list[int] = []
    targets: list[list[float]] = []
    sess = 0
    for sid in range(SUBREGION_COUNT):
        cx, cy = target_subregion_center_xy_m(sid)
        for _ in range(3):
            for _f in range(2):
                sessions.append(sess)
                targets.append([cx, cy, 0.0])
            sess += 1
    session_indices = np.asarray(sessions, dtype=np.int64)
    target_position = np.asarray(targets, dtype=np.float64)
    train_idx, val_idx, split_info = session_train_val_split_by_subregion(
        session_indices, target_position, 0.3, seed=11
    )
    train_s = set(session_indices[train_idx].tolist())
    val_s = set(session_indices[val_idx].tolist())
    assert train_s.isdisjoint(val_s)
    assert train_idx.size + val_idx.size == session_indices.size
    assert len(split_info) == SUBREGION_COUNT
    for info in split_info.values():
        assert info["train"] + info["val"] == 3
        assert info["val"] >= 1
        assert info["train"] >= 1


def test_filter_frame_indices_exclude_subregion_corners() -> None:
    from isac_imp.data_collection.cooperative_monostatic_dataset import (
        filter_frame_indices_exclude_subregion_corners,
    )

    # 16 格中心各一帧
    target_position = np.asarray(
        [list(target_subregion_center_xy_m(sid)) + [0.0] for sid in range(16)],
        dtype=np.float64,
    )
    all_idx = np.arange(16, dtype=np.int64)
    kept = filter_frame_indices_exclude_subregion_corners(all_idx, target_position)
    assert set(kept.tolist()) == set(range(16)) - SUBREGION_CORNER_IDS
    assert kept.size == 12
    # 仅非角格应原样保留
    non_corner = np.asarray([1, 5, 10, 14], dtype=np.int64)
    assert np.array_equal(
        filter_frame_indices_exclude_subregion_corners(non_corner, target_position),
        non_corner,
    )


def test_session_train_val_split_excludes_corner_subregions() -> None:
    from isac_imp.data_collection.cooperative_monostatic_dataset import (
        session_train_val_split_by_subregion,
    )

    sessions: list[int] = []
    targets: list[list[float]] = []
    sess = 0
    for sid in range(SUBREGION_COUNT):
        cx, cy = target_subregion_center_xy_m(sid)
        for _ in range(2):
            sessions.append(sess)
            targets.append([cx, cy, 0.0])
            sess += 1
    session_indices = np.asarray(sessions, dtype=np.int64)
    target_position = np.asarray(targets, dtype=np.float64)
    train_idx, val_idx, split_info = session_train_val_split_by_subregion(
        session_indices,
        target_position,
        0.5,
        seed=0,
        exclude_corner_subregions=True,
    )
    assert train_idx.size + val_idx.size == 12 * 2  # 12 non-corner cells × 2 sessions
    for sid in SUBREGION_CORNER_IDS:
        assert split_info[sid] == {"train": 0, "val": 0}
    for sid in set(range(SUBREGION_COUNT)) - SUBREGION_CORNER_IDS:
        assert split_info[sid]["train"] + split_info[sid]["val"] == 2
    for idx in np.concatenate([train_idx, val_idx]).tolist():
        x_m, y_m = float(target_position[idx, 0]), float(target_position[idx, 1])
        assert not is_subregion_corner_xy_m(x_m, y_m)
