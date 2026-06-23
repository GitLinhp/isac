"""RCSTarget 单元测试。"""

import pytest

from isac.channel import RCSTarget
from isac.data_structures import RCSTargetParams


def test_from_dict_defaults() -> None:
    t = RCSTarget.from_dict({})
    assert t.range_m == 100.0
    assert t.velocity_mps == 0.0
    assert t.rcs == 1e25
    assert t.azimuth_deg == 0.0
    assert t.position_rx_m == 0.0


def test_from_dict_parses_toml_keys() -> None:
    t = RCSTarget.from_dict(
        {
            "range_m": 200.0,
            "velocity_mps": 12.5,
            "rcs": 1e20,
            "azimuth_deg": 15.0,
            "position_rx_m": 0.5,
        }
    )
    assert t.range_m == 200.0
    assert t.velocity_mps == 12.5
    assert t.rcs == 1e20
    assert t.azimuth_deg == 15.0
    assert t.position_rx_m == 0.5


def test_update_partial_fields() -> None:
    t = RCSTarget()
    t.update(velocity_mps=5.0)
    assert t.velocity_mps == 5.0
    assert t.range_m == 100.0
    assert t.rcs == 1e25

    t.update(range_m=150.0, rcs=2e24, azimuth_deg=30.0)
    assert t.range_m == 150.0
    assert t.rcs == 2e24
    assert t.azimuth_deg == 30.0


def test_as_dict_roundtrip() -> None:
    t = RCSTarget(
        range_m=80.0,
        velocity_mps=-3.0,
        rcs=1e18,
        azimuth_deg=10.0,
        position_rx_m=0.25,
    )
    assert t.as_dict() == {
        "range_m": 80.0,
        "velocity_mps": -3.0,
        "rcs": 1e18,
        "azimuth_deg": 10.0,
        "position_rx_m": 0.25,
    }


def test_from_params() -> None:
    params = RCSTargetParams(
        range_m=120.0,
        velocity_mps=4.0,
        rcs=1e22,
        azimuth_deg=5.0,
        position_rx_m=1.0,
    )
    t = RCSTarget.from_params(params)
    assert t.as_dict() == {
        "range_m": params.range_m,
        "velocity_mps": params.velocity_mps,
        "rcs": params.rcs,
        "azimuth_deg": params.azimuth_deg,
        "position_rx_m": params.position_rx_m,
    }


def test_rejects_vector_range_m() -> None:
    with pytest.raises(ValueError, match="range_m 仅支持标量输入"):
        RCSTarget(range_m=[100.0, 200.0])


def test_rejects_vector_azimuth_deg() -> None:
    with pytest.raises(ValueError, match="azimuth_deg 仅支持标量输入"):
        RCSTarget(azimuth_deg=[0.0, 90.0])
