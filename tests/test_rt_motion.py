"""射线追踪目标 trajectory 运动逻辑单元测试。"""

import numpy as np
import pytest

pytest.importorskip("sionna.rt")

from isac.channel.rt.rt_target import RTTarget
from isac.channel.rt.rt_transceiver import RTTransceiver
from isac.channel.rt.trajectory import Trajectory
from isac.data_structures.params.rt_scene_params import TrajectoryParams, TargetParams


def _trajectory_dict(
    *,
    points: list[list[float]] | None = None,
    velocity: float = 2.0,
):
    return {
        "points": points if points is not None else [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "velocity": velocity,
    }


def _build_dummy_target(trajectory_dict: dict | None = None) -> RTTarget:
    class DummyTarget(RTTarget):
        @property
        def position(self):
            return self._dummy_position

        @position.setter
        def position(self, value):
            self._dummy_position = value

        @property
        def velocity(self):
            return self._dummy_velocity

        @velocity.setter
        def velocity(self, value):
            self._dummy_velocity = value

    tgt = DummyTarget.__new__(DummyTarget)
    tgt._dummy_position = [0.0, 0.0, 0.0]
    tgt._dummy_velocity = [0.0, 0.0, 0.0]
    tgt._pending_attributes = {k: None for k in RTTarget._SCENE_OBJECT_ATTRIBUTES}
    tgt._acceleration = None
    tgt._trajectory_params = (
        TrajectoryParams.from_dict(trajectory_dict)
        if trajectory_dict is not None
        else None
    )
    tgt._trajectory_done = False
    tgt._trajectory = Trajectory()
    tgt._pos_now = None
    tgt._velocity_now = None
    return tgt


def test_trajectory_params_roundtrip():
    mp = TrajectoryParams.from_dict(_trajectory_dict())
    assert len(mp.points) == 2


def test_trajectory_params_missing_points_raises():
    with pytest.raises(ValueError, match="trajectory.points 是必选配置项"):
        TrajectoryParams.from_dict({"velocity": 1.0})


def test_target_params_parses_trajectory_success():
    target = TargetParams.from_dict(
        {
            "fname": "car",
            "trajectory": _trajectory_dict(
                points=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                velocity=1.2,
            ),
        }
    )
    assert target.fname == "car"
    assert target.trajectory is not None
    assert target.trajectory.points == [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    assert target.trajectory.velocity == 1.2


def test_target_params_trajectory_type_error():
    with pytest.raises(TypeError, match="targets\\.\\*\\.trajectory 必须为 dict 或省略"):
        TargetParams.from_dict({"trajectory": "invalid"})


def test_rttarget_parse_trajectory_kwargs_success():
    kwargs = {
        "trajectory_points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "trajectory_velocity": 1.5,
    }
    trajectory = RTTarget._parse_trajectory_from_kwargs(kwargs)
    assert trajectory is not None
    assert trajectory.points == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert trajectory.velocity == 1.5
    assert kwargs == {}


def test_rttarget_partial_trajectory_kwargs_raises():
    with pytest.raises(ValueError, match="缺少 trajectory_points"):
        RTTarget._parse_trajectory_from_kwargs({"trajectory_velocity": 2.0})


def test_trajectory_interpolation_correctness():
    tgt = _build_dummy_target(
        _trajectory_dict(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            velocity=4.0,
        )
    )
    tgt.generate_motion_path()
    _, _, p1, v1, done = tgt.move(0.5)
    np.testing.assert_allclose(p1, [2.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(v1, [4.0, 0.0, 0.0], atol=1e-9)
    assert done is False


def test_reach_end_stops_motion():
    tgt = _build_dummy_target(
        _trajectory_dict(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            velocity=2.0,
        )
    )
    tgt.generate_motion_path()
    _, _, p1, v1, done = tgt.move(6.0)
    np.testing.assert_allclose(p1, [10.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(v1, [0.0, 0.0, 0.0], atol=1e-9)
    assert done is True


def test_single_point_trajectory_velocity_is_zero():
    tgt = _build_dummy_target(_trajectory_dict(points=[[1.0, 2.0, 3.0]], velocity=2.0))
    tgt.generate_motion_path()
    _, _, p1, v1, done = tgt.move(0.5)
    np.testing.assert_allclose(p1, [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(v1, [0.0, 0.0, 0.0], atol=1e-9)
    assert done is True


def test_trajectory_params_looping_mode_is_rejected():
    with pytest.raises(ValueError, match="trajectory\\.looping_mode 已废弃"):
        TrajectoryParams.from_dict(
            {
                "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "velocity": 1.0,
                "looping_mode": "mirror",
            }
        )


def test_generate_motion_path_without_trajectory_raises():
    tgt = _build_dummy_target(None)
    with pytest.raises(ValueError, match="未配置 trajectory"):
        tgt.generate_motion_path()


def test_rttarget_move_writes_list_to_scene_object():
    tgt = _build_dummy_target(
        _trajectory_dict(points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], velocity=2.0)
    )
    tgt.generate_motion_path()
    p0, v0, _, _, _ = tgt.move(0.5)
    assert isinstance(tgt.position, list)
    assert isinstance(tgt.velocity, list)
    assert len(p0) == 3 and len(v0) == 3


def test_rttransceiver_setup_is_decoupled_from_auto_move():
    class DummyDevice:
        def __init__(self):
            self.position = [0.0, 0.0, 0.0]
            self.velocity = [0.0, 0.0, 0.0]

    trx = RTTransceiver.__new__(RTTransceiver)
    trx.tx = DummyDevice()
    trx.rx = DummyDevice()
    trx._trajectory_params = TrajectoryParams.from_dict(
        _trajectory_dict(points=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], velocity=1.0)
    )
    trx._trajectory_done = False
    trx._trajectory = Trajectory()
    trx._pos_now = None
    trx._velocity_now = None

    # 仅配置 trajectory 不会自动推进或覆盖设备状态。
    assert trx._pos_now is None
    assert trx.tx.position == [0.0, 0.0, 0.0]
    assert trx.rx.position == [0.0, 0.0, 0.0]

    trx.generate_motion_path()
    np.testing.assert_allclose(trx.tx.position, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(trx.rx.position, [0.0, 0.0, 0.0], atol=1e-9)

    _, _, p1, v1, done = trx.move(0.5)
    np.testing.assert_allclose(p1, [0.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(v1, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(trx.tx.position, [0.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(trx.rx.position, [0.5, 0.0, 0.0], atol=1e-9)
    assert done is False
