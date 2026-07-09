"""RTSimulator 路径与目标交互类型判断测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sionna.phy.config
from sionna.rt.constants import InteractionType

from isac import PROJECT_ROOT
from isac.channel.rt.rt_simulator import (
    RTSimulator,
    _normalize_interaction_type,
)
from isac.collection import RoiKinematicsSampler
from isac.system import System
from isac.utils import set_random_seed
from isac.utils.numerical import cartesian_direction_to_yaw_pitch_roll

_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@pytest.fixture
def collection_system() -> System:
    sionna.phy.config.device = "cpu"
    set_random_seed(42)
    return System(_CONFIG, device="cpu")


def test_normalize_interaction_type_string_alias() -> None:
    assert _normalize_interaction_type("specular") == InteractionType.SPECULAR
    assert _normalize_interaction_type("  DIFFUSE ") == InteractionType.DIFFUSE


def test_normalize_interaction_type_int_passthrough() -> None:
    assert _normalize_interaction_type(InteractionType.REFRACTION) == (
        InteractionType.REFRACTION
    )


def test_normalize_interaction_type_unknown_raises() -> None:
    with pytest.raises(ValueError, match="未知 interaction_type"):
        _normalize_interaction_type("invalid_type")


def test_paths_intersect_object_with_interaction_match() -> None:
    sim = object.__new__(RTSimulator)
    object_id = 7
    sim.paths = lambda: SimpleNamespace(
        objects=np.array([[[[object_id, 99]]]]),
        interactions=np.array([[[[InteractionType.SPECULAR, InteractionType.DIFFUSE]]]]),
    )

    assert sim.paths_intersect_object_with_interaction(object_id, "specular") is True
    assert sim.paths_intersect_object_with_interaction(object_id, "diffuse") is False


def test_paths_intersect_target_with_interaction_delegates() -> None:
    sim = object.__new__(RTSimulator)
    target = SimpleNamespace(object_id=3)
    sim.paths = lambda: SimpleNamespace(
        objects=np.array([[[[3]]]]),
        interactions=np.array([[[[InteractionType.SPECULAR]]]]),
    )

    assert sim.paths_intersect_target_with_interaction(target, InteractionType.SPECULAR)
    assert not sim.paths_intersect_target_with_interaction(
        target, InteractionType.DIFFUSE
    )


def test_paths_intersect_target_with_interaction_integration(
    collection_system: System,
) -> None:
    rt_simulator = collection_system.components.rt_simulator
    sampler = RoiKinematicsSampler(
        roi=[-2.5, 2.5, -4.5, 4.5],
        position_sampling_mode="uniform",
        speed_range=[0.1, 3.0],
        speed_sampling_mode="uniform",
        num_samples=50,
    )
    _, target = next(iter(rt_simulator.rt_targets.items()))

    for _ in range(30):
        pos, vel, _ = sampler.pop()
        ori = cartesian_direction_to_yaw_pitch_roll(vel.reshape(1, 3))[0]
        target(position=pos, velocity=vel, orientation=ori)
        if not rt_simulator.scene_filter(pos):
            continue
        rt_simulator.paths(update=True)

        has_target = rt_simulator.paths_intersect_target(target)
        has_specular = rt_simulator.paths_intersect_target_with_interaction(
            target, "specular"
        )
        has_diffuse = rt_simulator.paths_intersect_target_with_interaction(
            target, InteractionType.DIFFUSE
        )

        if not has_target:
            assert not has_specular
            assert not has_diffuse
        else:
            assert has_specular
            return

    pytest.skip("30 次采样内未找到 paths_intersect_target 为 True 的位姿")
