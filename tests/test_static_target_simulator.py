"""static_target_simulator 向量化回归：与 gr-radar 等价 loop 参考实现对比。"""
from __future__ import annotations

import math

import pytest
import torch

from isac.channel.static_target_simulator import (
    StaticTargetParams,
    _apply_single_target_echo,
    _build_delay_filter,
    _build_doppler_filter,
    static_target_params_from_grc,
    static_target_simulator,
)

_TWO_PI = 2.0 * math.pi
_SAMP_RATE = 30_720_000.0
_DEVICE = torch.device("cpu")
_ATOL = 1e-5
_FULL_FRAME_N = 512 * 2560
# fast chirp 与 loop fmod 在 N≈1.3M 时 complex64 最大偏差约 2^-10
_CHIRP_ATOL = 1e-3
# fast chirp 经 FFT 链后全帧 echo 相对偏差约 1e-3 量级
_ECHO_RTOL = 2e-3
_ECHO_ATOL = 0.5


def _build_doppler_filter_loop_ref(
    n: int,
    doppler_hz: float,
    scale_ampl: float,
    samp_rate: float,
    device: torch.device,
) -> torch.Tensor:
    phase_inc = _TWO_PI * doppler_hz / samp_rate
    filt = torch.empty(n, device=device, dtype=torch.complex64)
    phase = 0.0
    scale = float(scale_ampl)
    for i in range(n):
        filt[i] = scale * complex(math.cos(phase), math.sin(phase))
        phase = math.fmod(phase + phase_inc, _TWO_PI)
    return filt


def _build_delay_filter_loop_ref(
    n: int,
    delay_s: float,
    samp_rate: float,
    *,
    compensate_numpy_ifft: bool,
    device: torch.device,
) -> torch.Tensor:
    filt = torch.empty(n, device=device, dtype=torch.complex64)
    half = n // 2
    fs = float(samp_rate)
    delay = float(delay_s)
    inv_n = 1.0 / float(n) if compensate_numpy_ifft else 1.0
    for i in range(n):
        freq = i * fs / n if i < half else i * fs / n - fs
        phase = math.fmod(_TWO_PI * delay * freq, _TWO_PI)
        filt[i] = (math.cos(phase) - 1j * math.sin(phase)) * inv_n
    return filt


def _apply_single_target_echo_loop_ref(
    tx: torch.Tensor,
    *,
    range_m: float,
    velocity_mps: float,
    rcs: float,
    azimuth_deg: float,
    position_rx_m: float,
    samp_rate: float,
    center_freq: float,
) -> torch.Tensor:
    from isac.channel.static_target_simulator import C_LIGHT, _target_scale_ampl

    n = tx.shape[-1]
    device = tx.device
    doppler_hz = 2.0 * velocity_mps * center_freq / C_LIGHT
    timeshift_s = 2.0 * range_m / C_LIGHT
    azimuth_shift_s = position_rx_m * math.sin(math.radians(azimuth_deg))
    scale_ampl = _target_scale_ampl(range_m, rcs, center_freq)

    doppler_filt = _build_doppler_filter_loop_ref(
        n, doppler_hz, scale_ampl, samp_rate, device
    )
    range_filt = _build_delay_filter_loop_ref(
        n, timeshift_s, samp_rate, compensate_numpy_ifft=False, device=device
    )
    azimuth_filt = _build_delay_filter_loop_ref(
        n, azimuth_shift_s, samp_rate, compensate_numpy_ifft=False, device=device
    )

    y = tx * doppler_filt
    y_fft = torch.fft.fft(y, dim=-1)
    y_fft = y_fft * range_filt * azimuth_filt
    return torch.fft.ifft(y_fft, dim=-1)


def _static_target_simulator_loop_ref(
    tx: torch.Tensor,
    params: StaticTargetParams,
) -> torch.Tensor:
    tx_c = tx.to(torch.complex64)
    ranges = (float(params.range_m),) if isinstance(params.range_m, (int, float)) else tuple(params.range_m)
    velocities = (
        (float(params.velocity_mps),)
        if isinstance(params.velocity_mps, (int, float))
        else tuple(params.velocity_mps)
    )
    rcs_vals = (float(params.rcs),) if isinstance(params.rcs, (int, float)) else tuple(params.rcs)
    azimuths = (
        (float(params.azimuth_deg),)
        if isinstance(params.azimuth_deg, (int, float))
        else tuple(params.azimuth_deg)
    )
    rx_positions = tuple(params.position_rx_m)

    rx_outputs: list[torch.Tensor] = []
    for pos_rx in rx_positions:
        out = torch.zeros_like(tx_c)
        for rng, vel, rcs, az in zip(ranges, velocities, rcs_vals, azimuths, strict=True):
            out = out + _apply_single_target_echo_loop_ref(
                tx_c,
                range_m=rng,
                velocity_mps=vel,
                rcs=rcs,
                azimuth_deg=az,
                position_rx_m=pos_rx,
                samp_rate=float(params.samp_rate),
                center_freq=params.center_freq,
            )
        if params.self_coupling:
            coupling = 10.0 ** (params.self_coupling_db / 20.0)
            out = out + coupling * tx_c
        rx_outputs.append(out)

    if len(rx_outputs) == 1:
        return rx_outputs[0]
    return torch.stack(rx_outputs, dim=0)


@pytest.mark.parametrize("n", [64, 2048, 512 * 2560])
@pytest.mark.parametrize("doppler_hz", [200.0, -350.0, 0.0])
def test_doppler_filter_matches_loop(n: int, doppler_hz: float) -> None:
    scale = 1.23e4
    vec = _build_doppler_filter(n, doppler_hz, scale, _SAMP_RATE, _DEVICE)
    ref = _build_doppler_filter_loop_ref(n, doppler_hz, scale, _SAMP_RATE, _DEVICE)
    atol = _CHIRP_ATOL if n >= _FULL_FRAME_N else _ATOL
    assert torch.allclose(vec, ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize("n", [64, 2048, 512 * 2560])
@pytest.mark.parametrize("delay_s", [1e-6, 6.67e-6, -2e-7])
@pytest.mark.parametrize("compensate_numpy_ifft", [False, True])
def test_delay_filter_matches_loop(
    n: int, delay_s: float, compensate_numpy_ifft: bool
) -> None:
    vec = _build_delay_filter(
        n,
        delay_s,
        _SAMP_RATE,
        compensate_numpy_ifft=compensate_numpy_ifft,
        device=_DEVICE,
    )
    ref = _build_delay_filter_loop_ref(
        n,
        delay_s,
        _SAMP_RATE,
        compensate_numpy_ifft=compensate_numpy_ifft,
        device=_DEVICE,
    )
    assert torch.allclose(vec, ref, atol=_ATOL, rtol=0.0)


@pytest.mark.parametrize("n", [256, 512 * 2560])
def test_apply_single_target_echo(n: int) -> None:
    torch.manual_seed(0)
    tx = torch.randn(n, dtype=torch.complex64, device=_DEVICE)
    kwargs = dict(
        range_m=100.0,
        velocity_mps=5.0,
        rcs=1e25,
        azimuth_deg=10.0,
        position_rx_m=0.5,
        samp_rate=_SAMP_RATE,
        center_freq=6e9,
        rndm_phaseshift=False,
        generator=None,
    )
    vec = _apply_single_target_echo(tx, **kwargs)
    ref = _apply_single_target_echo_loop_ref(tx, **{k: v for k, v in kwargs.items() if k not in ("rndm_phaseshift", "generator")})
    if n >= _FULL_FRAME_N:
        assert torch.allclose(vec, ref, atol=_ECHO_ATOL, rtol=_ECHO_RTOL)
    else:
        assert torch.allclose(vec, ref, atol=_ATOL, rtol=1e-6)


def test_static_target_simulator_end2end() -> None:
    n = 512 * 2560
    torch.manual_seed(42)
    tx = torch.randn(n, dtype=torch.complex64, device=_DEVICE)
    params = static_target_params_from_grc(
        range_m=100.0,
        velocity_mps=5.0,
        rndm_phaseshift=False,
        self_coupling=True,
        self_coupling_db=-10.0,
    )
    vec = static_target_simulator(tx, params)
    ref = _static_target_simulator_loop_ref(tx, params)
    assert torch.allclose(vec, ref, atol=_ECHO_ATOL, rtol=_ECHO_RTOL)
