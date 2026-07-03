"""DelayDopplerSpectrum ROI 物理量 ↔ bin 换算测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.sensing.delay_doppler_spectrum import DelayDopplerSpectrum


def _sp() -> SimpleNamespace:
    return SimpleNamespace(
        range_resolution=2.44,
        velocity_resolution=1.171,
        rg=SimpleNamespace(num_ofdm_symbols=512, fft_size=2048),
    )


def _dd(max_range_m: float = 310.0, max_velocity_mps: float = 150.0) -> DelayDopplerSpectrum:
    return DelayDopplerSpectrum(
        _sp(),
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
    )


def test_delay_and_doppler_bins():
    dd = _dd()
    assert dd.roi_delay_bins() == 128
    assert dd.roi_doppler_half_bins() == 128


def test_crop_shape_with_physical_roi():
    dd = _dd()
    h_dd = torch.randn(512, 2048, dtype=torch.complex64)
    cropped = dd.crop(h_dd)
    assert cropped.shape == (256, 128)


def test_limits_on_full_grid():
    dd = _dd()
    max_range_m, max_velocity_mps = dd.roi_limits(torch.zeros(512, 2048))
    assert max_range_m == pytest.approx(309.96, rel=1e-3)
    assert max_velocity_mps == pytest.approx(149.888, rel=1e-3)


def test_invalid_physical_values():
    with pytest.raises(ValueError):
        DelayDopplerSpectrum(_sp(), max_range_m=0.0, max_velocity_mps=10.0).crop(
            torch.zeros(512, 2048)
        )


def test_call_crops_to_roi_shape():
    dd = _dd()
    h_freq = torch.randn(512, 2048, dtype=torch.complex64)
    h_dd = dd(h_freq)
    expected = dd.crop(torch.zeros(512, 2048, dtype=torch.complex64))
    assert h_dd.shape == expected.shape
    assert dd._roi_slices is not None
