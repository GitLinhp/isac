"""短抽头 SIC 单元测试。"""

import numpy as np
import torch

from isac.sensing.clutter import cancel_short_tap_si, suggest_si_num_taps


def test_suggest_si_num_taps_min_and_geometry():
    from scipy.constants import speed_of_light as c

    dt = 8.14e-9
    assert suggest_si_num_taps(0.0, delay_resolution_s=dt) == 2
    # 0.05 m → τ/Δτ≈0.02 → ceil=1 → 1+guard2=3
    assert suggest_si_num_taps(0.05, delay_resolution_s=dt, guard_taps=2) == 3
    # τ = 3 Δτ → n_geom=3 → L=5
    sep = 3 * dt * c
    assert suggest_si_num_taps(sep, delay_resolution_s=dt, guard_taps=2) == 5


def test_cancel_short_tap_si_removes_near_zero_energy():
    rng = np.random.default_rng(0)
    s, f = 4, 256
    # 强近零 SI + 较弱远距离回波
    h_delay = np.zeros((s, f), dtype=np.complex128)
    h_delay[:, 0] = 10.0 + 0.0j
    h_delay[:, 1] = 3.0 + 1.0j
    h_delay[:, 40] = 0.5 + 0.2j
    h_shifted = np.fft.fft(h_delay, axis=-1)
    h_freq = np.fft.ifftshift(h_shifted, axes=-1)

    h_clean = cancel_short_tap_si(h_freq, num_taps=2)
    assert isinstance(h_clean, np.ndarray)
    h_s = np.fft.fftshift(h_clean, axes=-1)
    d = np.fft.ifft(h_s, axis=-1)
    assert np.allclose(d[:, :2], 0.0, atol=1e-10)
    assert np.mean(np.abs(d[:, 40])) > 0.1


def test_cancel_short_tap_si_torch_roundtrip_dtype():
    h = torch.randn(2, 64, dtype=torch.complex64)
    out = cancel_short_tap_si(h, num_taps=3)
    assert isinstance(out, torch.Tensor)
    assert out.shape == h.shape
    assert out.dtype == h.dtype
