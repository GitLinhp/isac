"""SystemParams.from_dict 解析约定测试。"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from isac.data_structures import SystemParams
from isac.data_structures.params.channel_params.rt_simulator_params import TargetParams

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_rt_target_module():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    rt_pkg_path = _SRC / "isac" / "channel" / "rt"
    for name, path in (
        ("isac", _SRC / "isac"),
        ("isac.channel", _SRC / "isac" / "channel"),
        ("isac.channel.rt", rt_pkg_path),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg

    rt_pkg = sys.modules["isac.channel.rt"]
    rt_pkg.RT_SCENES_DIR = rt_pkg_path / "scenes"

    spec = importlib.util.spec_from_file_location(
        "isac.channel.rt.rt_target",
        rt_pkg_path / "rt_target.py",
        submodule_search_locations=[str(rt_pkg_path)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "isac.channel.rt"
    sys.modules["isac.channel.rt.rt_target"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


RTTarget = _load_rt_target_module().RTTarget


def test_empty_section_parses_as_none() -> None:
    params = SystemParams.from_dict(
        {
            "carrier_frequency": 6e9,
            "ofdm": {"fft_size": 1024, "num_symbols": 512},
            "channel": {"type": "rt"},
            "rt_simulator": {"filename": "test"},
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


def test_target_params_scaling_default() -> None:
    params = TargetParams.from_dict({"fname": "cube", "material": "car_material"})
    assert params.scaling == 1.0


def test_target_params_scaling_scalar() -> None:
    params = TargetParams.from_dict(
        {"fname": "cube", "material": "car_material", "scaling": 2.0}
    )
    assert params.scaling == 2.0


def test_target_params_scaling_vector() -> None:
    params = TargetParams.from_dict(
        {"fname": "cube", "material": "car_material", "scaling": [1.0, 1.0, 2.0]}
    )
    assert params.scaling == [1.0, 1.0, 2.0]


@pytest.fixture
def capture_rt_target_setattr(monkeypatch):
    captured: dict[str, object] = {}
    real_setattr = setattr

    def _setattr(obj, name, value):
        if isinstance(obj, RTTarget):
            captured[name] = value
            return
        real_setattr(obj, name, value)

    monkeypatch.setattr("builtins.setattr", _setattr)
    return captured


def test_rt_target_call_scaling_scalar_uses_float(capture_rt_target_setattr) -> None:
    target = object.__new__(RTTarget)
    target(scaling=1.0)
    assert capture_rt_target_setattr["scaling"] == 1.0
    assert isinstance(capture_rt_target_setattr["scaling"], float)


def test_rt_target_call_scaling_vector_uses_xyz_list(capture_rt_target_setattr) -> None:
    target = object.__new__(RTTarget)
    target(scaling=[1, 1, 2])
    assert capture_rt_target_setattr["scaling"] == [1.0, 1.0, 2.0]


def test_rt_target_call_scaling_invalid_length_raises() -> None:
    target = object.__new__(RTTarget)
    with pytest.raises(ValueError, match="scaling 须为标量或长度为 3 的向量"):
        target(scaling=[1, 2])


def test_rt_simulator_render_params_from_dict() -> None:
    from isac.data_structures.params.channel_params.rt_simulator_params import (
        RenderParams,
        RTSimulatorParams,
    )

    render = RenderParams.from_dict({"clip_at": 3.4, "with_paths": False})
    assert render.clip_at == 3.4
    assert render.with_paths is False

    params = RTSimulatorParams.from_dict(
        {"filename": "empty_room", "render": {"clip_at": 3.4}}
    )
    assert params.render is not None
    assert params.render.clip_at == 3.4
    assert params.render.with_paths is True


def test_rt_simulator_init_with_default_target_scaling() -> None:
    from isac.system import System

    system = System("simulation/sensing/sensing_monostatic.toml", device="cpu")
    target = system.components.rt_simulator.rt_targets["car"]
    assert target.scaling is not None


def test_rt_simulator_rx_tx_states_skip_missing_role() -> None:
    from isac.channel.rt.rt_simulator import RTSimulator
    from isac.channel.rt.rt_transceiver import RTTransceiver

    sim = object.__new__(RTSimulator)
    sim.transceivers = {
        "bs1": RTTransceiver(
            name="bs1",
            position=[0, 0, 0],
            transceiver_type=["rx"],
        ),
        "bs2": RTTransceiver(
            name="bs2",
            position=[1, 0, 0],
            transceiver_type=["tx"],
        ),
    }

    rx_states = sim.rx_states
    tx_states = sim.tx_states

    assert list(rx_states.keys()) == ["bs1_rx"]
    assert list(tx_states.keys()) == ["bs2_tx"]


def test_bistatic_config_rx_tx_states() -> None:
    from isac.system import System

    system = System("simulation/sensing/sensing_bistatic.toml", device="cpu")
    rt = system.components.rt_simulator

    assert "bs1_rx" in rt.rx_states
    assert "bs2_tx" in rt.tx_states
    assert "bs2_rx" not in rt.rx_states
    assert "bs1_tx" not in rt.tx_states


def test_rt_transceiver_rx_position_offset_colocated() -> None:
    """同逻辑名 tx+rx 时 RX 应用偏移，TX 保持名义位置，间距仍 ≤ 共址阈值。"""
    import numpy as np

    from isac.channel.rt.rt_transceiver import RTTransceiver
    from isac.sensing.geometry import MONOSTATIC_TX_RX_EPS_M

    nominal = [-4.5, -2.5, 1.5]
    offset = [0.0, 0.0, 0.05]
    tc = RTTransceiver(
        name="bs1",
        position=nominal,
        transceiver_type=["tx", "rx"],
        rx_position_offset=offset,
    )
    assert tc.tx is not None and tc.rx is not None
    tx_pos = np.asarray(tc.tx.position, dtype=np.float64).reshape(3)
    rx_pos = np.asarray(tc.rx.position, dtype=np.float64).reshape(3)
    assert np.allclose(tx_pos, nominal)
    assert np.allclose(rx_pos, np.asarray(nominal) + np.asarray(offset))
    assert np.allclose(tc.position, nominal)
    sep = float(np.linalg.norm(rx_pos - tx_pos))
    assert sep == pytest.approx(0.05)
    assert sep <= MONOSTATIC_TX_RX_EPS_M


def test_rt_transceiver_rx_offset_ignored_when_rx_only() -> None:
    """仅 RX 时忽略 rx_position_offset，避免误伤分离拓扑。"""
    import numpy as np

    from isac.channel.rt.rt_transceiver import RTTransceiver

    tc = RTTransceiver(
        name="bs1",
        position=[1.0, 2.0, 3.0],
        transceiver_type=["rx"],
        rx_position_offset=[0.0, 0.0, 0.05],
    )
    assert tc.tx is None and tc.rx is not None
    rx_pos = np.asarray(tc.rx.position, dtype=np.float64).reshape(3)
    assert np.allclose(rx_pos, [1.0, 2.0, 3.0])
