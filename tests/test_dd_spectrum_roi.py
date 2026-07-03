"""DelayDopplerRoi 物理量 ↔ bin 换算测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.sensing.dd_spectrum_roi import DelayDopplerRoi


def _sp() -> SimpleNamespace:
    return SimpleNamespace(range_resolution=2.44, velocity_resolution=1.171)


def test_delay_and_doppler_bins():
    roi = DelayDopplerRoi(max_range_m=310.0, max_velocity_mps=150.0)
    sp = _sp()
    assert roi.delay_bins(sp) == 128
    assert roi.doppler_half_bins(sp) == 128


def test_crop_shape_with_physical_roi():
    roi = DelayDopplerRoi(max_range_m=310.0, max_velocity_mps=150.0)
    sp = _sp()
    h_dd = torch.randn(512, 2048, dtype=torch.complex64)
    cropped = roi.crop(h_dd, sp)
    assert cropped.shape == (256, 128)


def test_limits_on_full_grid():
    sp = _sp()
    roi = DelayDopplerRoi(max_range_m=310.0, max_velocity_mps=150.0)
    max_range_m, max_velocity_mps = roi.limits(
        torch.zeros(512, 2048), sp
    )
    assert max_range_m == pytest.approx(309.96, rel=1e-3)
    assert max_velocity_mps == pytest.approx(149.888, rel=1e-3)


def test_invalid_physical_values():
    with pytest.raises(ValueError):
        DelayDopplerRoi(max_range_m=0.0, max_velocity_mps=10.0)
