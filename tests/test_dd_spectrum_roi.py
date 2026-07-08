"""DelayDopplerSpectrum ROI 物理量 ↔ bin 换算测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.sensing.spectrum import DelayDopplerSpectrum


def _sp() -> SimpleNamespace:
    return SimpleNamespace(
        range_resolution_monostatic=2.44,
        velocity_resolution_monostatic=1.171,
        rg=SimpleNamespace(num_ofdm_symbols=512, fft_size=2048),
    )


def _dd(max_range_m: float = 310.0, max_velocity_mps: float = 150.0) -> DelayDopplerSpectrum:
    return DelayDopplerSpectrum(
        _sp(),
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
    )


def _expected_roi_shape(dd: DelayDopplerSpectrum) -> tuple[int, int]:
    assert dd.dd_spectrum_roi is not None
    h_full = torch.zeros(512, 2048, dtype=torch.complex64)
    dop_start, dop_end, _, delay_end = dd.dd_spectrum_roi.bin_slices(h_full)
    return dop_end - dop_start, delay_end


def test_delay_and_doppler_bins():
    dd = _dd()
    assert dd.dd_spectrum_roi is not None
    assert dd.dd_spectrum_roi.delay_bin_count() == 128
    assert dd.dd_spectrum_roi.doppler_half_bins() == 128


def test_call_output_shape():
    dd = _dd()
    h_freq = torch.randn(512, 2048, dtype=torch.complex64)
    h_dd = dd(h_freq)
    assert h_dd.shape == _expected_roi_shape(dd)


def test_limits_on_full_grid():
    dd = _dd()
    sp = _sp()
    assert dd.dd_spectrum_roi is not None
    dop_start, dop_end, _, delay_end = dd.dd_spectrum_roi.bin_slices(torch.zeros(512, 2048))
    max_range_m = (delay_end - 1) * sp.range_resolution_monostatic
    max_velocity_mps = ((dop_end - dop_start) // 2) * sp.velocity_resolution_monostatic
    assert max_range_m == pytest.approx(309.96, rel=1e-3)
    assert max_velocity_mps == pytest.approx(149.888, rel=1e-3)


def test_effective_physical_limits():
    dd = _dd(max_range_m=157.5, max_velocity_mps=25.6)
    assert dd.dd_spectrum_roi is not None
    max_range_m, max_velocity_mps = dd.dd_spectrum_roi.effective_physical_limits()
    assert max_range_m == pytest.approx(64 * 2.44, rel=1e-3)
    assert max_velocity_mps == pytest.approx(22 * 1.171, rel=1e-3)


def test_invalid_physical_values():
    dd = DelayDopplerSpectrum(_sp(), max_range_m=0.0, max_velocity_mps=10.0)
    with pytest.raises(ValueError):
        dd(torch.zeros(512, 2048, dtype=torch.complex64))


def test_call_crops_to_roi_shape():
    dd = _dd()
    h_freq = torch.randn(512, 2048, dtype=torch.complex64)
    h_dd = dd(h_freq)
    assert h_dd.shape == _expected_roi_shape(dd)
    assert dd.dd_spectrum_roi is not None
    assert dd.dd_spectrum_roi.slices is not None
