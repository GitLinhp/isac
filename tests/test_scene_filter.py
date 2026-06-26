"""SceneFilter 与收发机–障碍物包围盒校验单元测试。"""

import numpy as np
import pytest

from isac.channel.rt.scene_filter import (
    SceneFilter,
    _point_inside_aabb,
    validate_transceivers_not_in_obstacles,
)


class _Mesh:
    def __init__(self, box_min, box_max):
        self._min = np.asarray(box_min, dtype=np.float64)
        self._max = np.asarray(box_max, dtype=np.float64)

    def bbox(self):
        return type("B", (), {"min": self._min, "max": self._max})()


class _Obj:
    def __init__(self, box_min, box_max):
        self.mi_mesh = _Mesh(box_min, box_max)


def test_point_inside_aabb():
    assert _point_inside_aabb(
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0]),
    )
    assert not _point_inside_aabb(
        np.array([3.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0]),
    )


def test_validate_transceiver_raises_when_inside_obstacle():
    class _Tc:
        position = [5.0, 5.0, 5.0]

    class _Scene:
        objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
        transceivers = {"bs1": _Tc()}
        scene_params = None

    with pytest.raises(ValueError, match="收发机 'bs1'"):
        validate_transceivers_not_in_obstacles(_Scene())


def test_validate_transceiver_passes_when_clear():
    class _Tc:
        position = [50.0, 0.0, 0.0]

    class _Scene:
        objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
        transceivers = {"bs1": _Tc()}
        scene_params = None

    validate_transceivers_not_in_obstacles(_Scene())


def test_scene_filter_mc_sampling_unchanged():
    class _Scene:
        objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
        transceivers = {}

    filt = SceneFilter(_Scene(), safe_margin=1.0)
    assert filt.is_valid(np.array([50.0, 0.0, 0.0]))
    assert not filt.is_valid(np.array([5.0, 5.0, 5.0]))
