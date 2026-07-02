"""RTSceneFilter 与收发机–障碍物包围盒校验单元测试。"""

import numpy as np
import pytest

from sionna.rt import Scene

from isac.channel.rt.rt_simulator import RTSimulator
from isac.channel.rt.rt_scene_filter import RTSceneFilter


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
    class _Scene:
        objects = {"building_1": _Obj([0, 0, 0], [2, 2, 2])}
        transceivers = {}

    filt = RTSceneFilter(_Scene(), safe_margin=0.0)
    assert not filt(np.array([1.0, 1.0, 1.0]))
    assert filt(np.array([3.0, 1.0, 1.0]))


def test_validate_transceiver_raises_when_inside_obstacle():
    class _Tc:
        position = [5.0, 5.0, 5.0]

    sim = object.__new__(RTSimulator)
    sim.scene = object.__new__(Scene)
    sim.scene._scene_objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
    sim.transceivers = {"bs1": _Tc()}

    with pytest.raises(ValueError, match="收发机 'bs1'"):
        sim.validate_transceivers_not_in_obstacles()


def test_validate_transceiver_passes_when_clear():
    class _Tc:
        position = [50.0, 0.0, 0.0]

    sim = object.__new__(RTSimulator)
    sim.scene = object.__new__(Scene)
    sim.scene._scene_objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
    sim.transceivers = {"bs1": _Tc()}

    sim.validate_transceivers_not_in_obstacles()


def test_scene_filter_mc_sampling_unchanged():
    class _Scene:
        objects = {"building_1": _Obj([0, 0, 0], [10, 10, 10])}
        transceivers = {}

    filt = RTSceneFilter(_Scene(), safe_margin=1.0)
    assert filt(np.array([50.0, 0.0, 0.0]))
    assert not filt(np.array([5.0, 5.0, 5.0]))
