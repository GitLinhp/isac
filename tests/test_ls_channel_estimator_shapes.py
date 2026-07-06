"""LSChannelEstimator 输入形状规范化。"""

import pytest
import torch
from sionna.phy.ofdm import ResourceGrid

from isac.sensing.ls_channel_estimator import LSChannelEstimator


def _make_estimator(s: int = 4, f: int = 8) -> LSChannelEstimator:
    rg = ResourceGrid(
        num_ofdm_symbols=s,
        fft_size=f,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=2,
        num_guard_carriers=[0, 0],
        dc_null=False,
        pilot_pattern=None,
        pilot_ofdm_symbol_indices=[],
    )
    return LSChannelEstimator(rg)


def test_ls_channel_estimator_2d_inputs():
    est = _make_estimator()
    x = torch.ones(4, 8, dtype=torch.complex64)
    y = torch.ones(4, 8, dtype=torch.complex64) * 2
    h = est(x, y)
    assert h.shape == (4, 8)
    assert torch.allclose(h, torch.full((4, 8), 2.0, dtype=torch.complex64))


def test_ls_channel_estimator_3d_leading_one_squeezes():
    est = _make_estimator()
    x = torch.ones(1, 4, 8, dtype=torch.complex64)
    y = torch.ones(1, 4, 8, dtype=torch.complex64) * 3
    h = est(x, y)
    assert h.shape == (4, 8)


def test_ls_channel_estimator_y_3d_rx_broadcast():
    est = _make_estimator()
    x = torch.ones(4, 8, dtype=torch.complex64)
    y = torch.ones(2, 4, 8, dtype=torch.complex64) * 2
    h = est(x, y)
    assert h.shape == (2, 4, 8)


def test_ls_channel_estimator_rejects_invalid_ndim():
    est = _make_estimator()
    x = torch.ones(2, 2, 4, 8, dtype=torch.complex64)
    y = torch.ones(4, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="x squeeze 后须为 2D"):
        est(x, y)
