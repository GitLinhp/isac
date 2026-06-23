"""SystemParams.from_dict 解析约定测试。"""

import pytest

from isac.data_structures import SystemParams


def test_empty_section_parses_as_none() -> None:
    params = SystemParams.from_dict(
        {
            "carrier_frequency": 6e9,
            "ofdm": {"fft_size": 1024, "num_symbols": 512},
            "channel": {"type": "rt"},
            "rt_scene": {"filename": "test"},
            "music": {},
            "windows": {},
        }
    )
    assert params.music is None
    assert params.windows is None


def test_rcs_scene_nested_target() -> None:
    params = SystemParams.from_dict(
        {
            "carrier_frequency": 6e9,
            "ofdm": {"fft_size": 2048, "num_symbols": 512},
            "channel": {"type": "rcs"},
            "rcs_scene": {
                "self_coupling_db": -10.0,
                "target": {
                    "range_m": 95.0,
                    "velocity_mps": 5.0,
                    "rcs": 1e25,
                },
            },
        }
    )
    assert params.rcs_scene is not None
    assert params.rcs_scene.target.range_m == 95.0
    assert params.rcs_scene.self_coupling_db == -10.0


def test_rcs_scene_requires_target_subsection() -> None:
    with pytest.raises(ValueError, match="rcs_scene.target"):
        SystemParams.from_dict(
            {
                "channel": {"type": "rcs"},
                "rcs_scene": {"self_coupling_db": -10.0},
            }
        )


def test_rcs_scene_flat_target_keys_rejected() -> None:
    with pytest.raises(ValueError, match="rcs_scene.target"):
        SystemParams.from_dict(
            {
                "channel": {"type": "rcs"},
                "rcs_scene": {
                    "range_m": 95.0,
                    "velocity_mps": 5.0,
                },
            }
        )
