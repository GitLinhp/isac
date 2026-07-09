"""DelayDopplerSpectrum：无 [windows] 或子窗为 None 时不加窗。"""

from __future__ import annotations

import sionna.phy.config
import torch
from sionna.phy.ofdm import ResourceGrid

from isac.data_structures.params.system_params import SystemParams
from isac.data_structures.system_components import SystemComponents
from isac.sensing.spectrum import DelayDopplerSpectrum
from isac.sensing.spectrum.dd_spectrum_roi import DelayDopplerRoi
from isac.system import System
from isac.utils import set_random_seed
from isac import PROJECT_ROOT

_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


def _base_config_dict() -> dict:
    return {
        "carrier_frequency": 6e9,
        "ofdm": {
            "num_symbols": 512,
            "fft_size": 2048,
            "subcarrier_spacing": 30e3,
            "cyclic_prefix_length": 512,
            "l_min": -6,
        },
        "mti": {"filter_order": 1},
        "dd_spectrum_roi": {"max_range_m": 30.0, "max_velocity_mps": 5.0},
    }


def _resource_grid() -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=512,
        fft_size=2048,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=512,
        device="cpu",
    )


def _build_dd_from_params(params: SystemParams) -> DelayDopplerSpectrum:
    kwargs = SystemComponents._build_sensing(params, _resource_grid(), "cpu")
    dd = kwargs.get("delay_doppler_spectrum")
    assert dd is not None
    return dd


def test_dd_spectrum_built_without_windows_section() -> None:
    params = SystemParams.from_dict(_base_config_dict())
    assert params.windows is None

    dd = _build_dd_from_params(params)
    assert dd.delay_window is None
    assert dd.doppler_window is None


def test_dd_spectrum_delay_window_only() -> None:
    cfg = _base_config_dict()
    cfg["windows"] = {
        "delay_window": {"type": "hamming"},
    }
    params = SystemParams.from_dict(cfg)
    assert params.windows is not None

    dd = _build_dd_from_params(params)
    assert dd.delay_window == {"type": "hamming"}
    assert dd.doppler_window is None


def test_dd_spectrum_forward_shape_without_windows() -> None:
    sionna.phy.config.device = "cpu"
    set_random_seed(42)
    system = System(_CONFIG, device="cpu")
    dd = system.components.delay_doppler_spectrum
    assert dd is not None
    assert dd.delay_window is None
    assert dd.doppler_window is None

    rg = system.components.sensing_performance.rg
    h_freq = torch.zeros(
        rg.num_ofdm_symbols,
        rg.fft_size,
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    h_dd = dd(h_freq)
    assert h_dd.ndim == 2
    assert h_dd.shape[0] > 0
    assert h_dd.shape[1] > 0


def test_dd_spectrum_forward_matches_rectangular_window() -> None:
    """无窗配置与显式 delay/doppler_window=None 输出一致。"""
    params = SystemParams.from_dict(_base_config_dict())
    kwargs = SystemComponents._build_sensing(params, _resource_grid(), "cpu")
    sp = kwargs["sensing_performance"]
    roi = kwargs["dd_spectrum_roi"]
    assert sp is not None
    assert roi is not None

    dd_none = DelayDopplerSpectrum(
        sensing_performance=sp,
        delay_window=None,
        doppler_window=None,
        dd_spectrum_roi=roi,
        device=torch.device("cpu"),
    )
    dd_default = DelayDopplerSpectrum(
        sensing_performance=sp,
        dd_spectrum_roi=roi,
        device=torch.device("cpu"),
    )

    h_freq = torch.randn(
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        dtype=torch.complex64,
    )
    out_none = dd_none(h_freq)
    out_default = dd_default(h_freq)
    assert out_none.shape == out_default.shape
    assert torch.allclose(out_none, out_default)
