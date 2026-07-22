"""频域短抽头自干扰消除（SIC）：从 LS CFR 中减去近零时延分量。

典型用于共址单基地：TX–RX 直射落在前若干时延抽头，目标回波在更长时延。
对 ``h_freq`` 做 IFFT → 保留前 ``num_taps`` 抽头重建 ``H_si`` → ``H_clean = H - H_si``。
等价于时延域将前 ``num_taps`` 抽头置零后再 FFT 回频域。
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
from scipy.constants import speed_of_light as c

ArrayLike = Union[torch.Tensor, np.ndarray]


def suggest_si_num_taps(
    tx_rx_separation_m: float,
    *,
    delay_resolution_s: float,
    guard_taps: int = 2,
    min_taps: int = 2,
) -> int:
    """由 TX–RX 间距与时延分辨率建议 SIC 抽头数。

    ``L = ceil(τ_si / Δτ) + guard``，``τ_si = sep / c``；结果不低于 ``min_taps``。
    """
    if delay_resolution_s <= 0:
        raise ValueError(f"delay_resolution_s 须为正，收到 {delay_resolution_s}")
    sep = max(0.0, float(tx_rx_separation_m))
    tau_si = sep / c
    n_geom = int(math.ceil(tau_si / delay_resolution_s)) if tau_si > 0 else 0
    return max(int(min_taps), n_geom + int(guard_taps))


def cancel_short_tap_si(
    h_freq: ArrayLike,
    *,
    num_taps: int,
) -> ArrayLike:
    """短抽头 SIC：消除频域信道前 ``num_taps`` 个时延抽头对应的 SI。

    时延域约定与 :class:`~isac.sensing.spectrum.DelayDopplerSpectrum` 一致：
    先对子载波维 ``fftshift``，再 ``ifft``；零多普勒/零时延在 tap 0。
    重建 SI 后 ``ifftshift`` 回资源网格频轴顺序。

    参数:
        h_freq: LS 频域信道，形状 ``(S, F)``（或可 squeeze 为此）。
        num_taps: 视为自干扰的时延抽头数（含 tap 0）；须 ``1 <= num_taps < F``。

    返回:
        与输入同类型、同形状的 ``H_clean``。
    """
    if num_taps < 1:
        raise ValueError(f"num_taps 须 >= 1，收到 {num_taps}")

    is_torch = isinstance(h_freq, torch.Tensor)
    if is_torch:
        h = h_freq
        device, dtype = h.device, h.dtype
        h_np = h.detach().cpu().numpy()
    else:
        h_np = np.asarray(h_freq)

    squeezed = False
    if h_np.ndim == 3 and h_np.shape[0] == 1:
        h_np = h_np[0]
        squeezed = True
    if h_np.ndim != 2:
        raise ValueError(f"h_freq 须为 2D (S, F)，收到 shape={h_np.shape}")

    n_sym, fft_size = h_np.shape
    if num_taps >= fft_size:
        raise ValueError(
            f"num_taps={num_taps} 须 < fft_size={fft_size}，否则会消掉全部能量"
        )

    h_c = h_np.astype(np.complex128, copy=False)
    # 与 DelayDopplerSpectrum._transform_freq_to_dd 一致：fftshift → ifft
    h_shifted = np.fft.fftshift(h_c, axes=-1)
    h_delay = np.fft.ifft(h_shifted, axis=-1)
    h_si_delay = np.zeros_like(h_delay)
    h_si_delay[..., :num_taps] = h_delay[..., :num_taps]
    h_si_shifted = np.fft.fft(h_si_delay, axis=-1)
    h_si = np.fft.ifftshift(h_si_shifted, axes=-1)
    h_clean = (h_c - h_si).astype(h_np.dtype, copy=False)

    if squeezed:
        h_clean = h_clean[np.newaxis, ...]

    if is_torch:
        out = torch.from_numpy(h_clean).to(device=device, dtype=dtype)
        return out
    return h_clean
