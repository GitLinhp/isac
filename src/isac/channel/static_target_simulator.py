"""gr-radar static_target_simulator_cc 的 Torch 复现（点目标散射 + FFT 分数时延）。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

C_LIGHT = 299_792_458.0
FOUR_PI_CUBED_SQRT = math.sqrt((4.0 * math.pi) ** 3)


def _as_float_vector(values: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        return (float(values),)
    seq = tuple(float(v) for v in values)
    if not seq:
        raise ValueError(f"{name} 不能为空")
    return seq


@dataclass(frozen=True)
class StaticTargetParams:
    """与 gr-radar static_target_simulator_cc 块参数对应。"""

    range_m: Sequence[float]
    velocity_mps: Sequence[float]
    rcs: Sequence[float]
    azimuth_deg: Sequence[float]
    position_rx_m: Sequence[float]
    samp_rate: int
    center_freq: float
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True

    def __post_init__(self) -> None:
        ranges = _as_float_vector(self.range_m, "range_m")
        velocities = _as_float_vector(self.velocity_mps, "velocity_mps")
        rcs_vals = _as_float_vector(self.rcs, "rcs")
        azimuths = _as_float_vector(self.azimuth_deg, "azimuth_deg")
        rx_positions = _as_float_vector(self.position_rx_m, "position_rx_m")
        n = len(ranges)
        if not (len(velocities) == len(rcs_vals) == len(azimuths) == n):
            raise ValueError("range_m / velocity_mps / rcs / azimuth_deg 长度须一致")
        if not rx_positions:
            raise ValueError("position_rx_m 不能为空")
        if self.samp_rate <= 0:
            raise ValueError("samp_rate 须为正")
        if self.center_freq <= 0:
            raise ValueError("center_freq 须为正")

    @property
    def num_targets(self) -> int:
        return len(_as_float_vector(self.range_m, "range_m"))

    @property
    def num_rx(self) -> int:
        return len(_as_float_vector(self.position_rx_m, "position_rx_m"))


def static_target_params_from_grc(
    *,
    range_m: float | Sequence[float] = 100.0,
    velocity_mps: float | Sequence[float] = 5.0,
    rcs: float | Sequence[float] = 1e25,
    azimuth_deg: float | Sequence[float] = 0.0,
    position_rx_m: Sequence[float] = (0.0,),
    center_freq: float = 6e9,
    samp_rate: int = 30_720_000,
    self_coupling_db: float = -10.0,
    rndm_phaseshift: bool = True,
    self_coupling: bool = True,
) -> StaticTargetParams:
    """与 simulator_ofdm.grc 中 radar_static_target_simulator_cc_0 默认参数对齐。"""
    return StaticTargetParams(
        range_m=range_m,
        velocity_mps=velocity_mps,
        rcs=rcs,
        azimuth_deg=azimuth_deg,
        position_rx_m=position_rx_m,
        samp_rate=samp_rate,
        center_freq=center_freq,
        self_coupling_db=self_coupling_db,
        rndm_phaseshift=rndm_phaseshift,
        self_coupling=self_coupling,
    )



def _build_doppler_filter(
    n: int,
    doppler_hz: float,
    scale_ampl: float,
    samp_rate: float,
    device: torch.device,
) -> torch.Tensor:
    """逐样点相位累加（fmod），与 gr-radar d_filt_doppler 一致。"""
    phase_inc = 2.0 * math.pi * doppler_hz / samp_rate
    filt = torch.empty(n, device=device, dtype=torch.complex64)
    phase = 0.0
    scale = float(scale_ampl)
    for i in range(n):
        filt[i] = scale * complex(math.cos(phase), math.sin(phase))
        phase = math.fmod(phase + phase_inc, 2.0 * math.pi)
    return filt


def _build_delay_filter(
    n: int,
    delay_s: float,
    samp_rate: float,
    *,
    compensate_numpy_ifft: bool,
    device: torch.device,
) -> torch.Tensor:
    """频域分数时延滤波器。

    gr-radar (FFTW) 在 range 滤波器内除以 N 以补偿无归一化 IFFT。
    ``torch.fft.ifft`` 默认含 1/N，故 ``compensate_numpy_ifft=False`` 时不除 N。
    """
    filt = torch.empty(n, device=device, dtype=torch.complex64)
    half = n // 2
    fs = float(samp_rate)
    delay = float(delay_s)
    inv_n = 1.0 / float(n) if compensate_numpy_ifft else 1.0
    for i in range(n):
        freq = i * fs / n if i < half else i * fs / n - fs
        phase = math.fmod(2.0 * math.pi * delay * freq, 2.0 * math.pi)
        filt[i] = (math.cos(phase) - 1j * math.sin(phase)) * inv_n
    return filt


def _target_scale_ampl(range_m: float, rcs: float, center_freq: float) -> float:
    return C_LIGHT * math.sqrt(rcs) / FOUR_PI_CUBED_SQRT / (range_m * range_m) / center_freq


def _apply_single_target_echo(
    tx: torch.Tensor,
    *,
    range_m: float,
    velocity_mps: float,
    rcs: float,
    azimuth_deg: float,
    position_rx_m: float,
    samp_rate: float,
    center_freq: float,
    rndm_phaseshift: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    n = tx.shape[-1]
    device = tx.device

    doppler_hz = 2.0 * velocity_mps * center_freq / C_LIGHT
    timeshift_s = 2.0 * range_m / C_LIGHT
    azimuth_shift_s = position_rx_m * math.sin(math.radians(azimuth_deg))
    scale_ampl = _target_scale_ampl(range_m, rcs, center_freq)

    doppler_filt = _build_doppler_filter(
        n, doppler_hz, scale_ampl, samp_rate, device
    )
    range_filt = _build_delay_filter(
        n,
        timeshift_s,
        samp_rate,
        compensate_numpy_ifft=False,
        device=device,
    )
    azimuth_filt = _build_delay_filter(
        n,
        azimuth_shift_s,
        samp_rate,
        compensate_numpy_ifft=False,
        device=device,
    )

    y = tx * doppler_filt
    y_fft = torch.fft.fft(y, dim=-1)
    y_fft = y_fft * range_filt * azimuth_filt
    y = torch.fft.ifft(y_fft, dim=-1)

    if rndm_phaseshift:
        if generator is not None:
            rand_val = torch.rand((), generator=generator, device=device)
        else:
            rand_val = torch.rand((), device=device)
        phase = torch.exp(1j * (2.0 * math.pi * rand_val).to(torch.complex64))
        y = y * phase

    return y


def static_target_simulator(
    tx: torch.Tensor,
    params: StaticTargetParams,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """模拟 gr-radar static_target_simulator_cc 接收 IQ。

    Parameters
    ----------
    tx:
        时域发射 IQ，末维为样点；支持 ``(N,)`` 或 ``(..., N)``。
    params:
        目标与硬件参数。
    generator:
        随机相位生成器（``rndm_phaseshift=True`` 时可选，用于可复现）。

    Returns
    -------
    torch.Tensor
        单 RX（``num_rx==1``）时与 ``tx`` 同 shape；多 RX 时为 ``(num_rx, *tx.shape)``。
    """
    if tx.shape[-1] == 0:
        raise ValueError("tx 末维样点数须为正")

    tx_c = tx.to(torch.complex64)
    ranges = _as_float_vector(params.range_m, "range_m")
    velocities = _as_float_vector(params.velocity_mps, "velocity_mps")
    rcs_vals = _as_float_vector(params.rcs, "rcs")
    azimuths = _as_float_vector(params.azimuth_deg, "azimuth_deg")
    rx_positions = _as_float_vector(params.position_rx_m, "position_rx_m")

    rx_outputs: list[torch.Tensor] = []
    for pos_rx in rx_positions:
        out = torch.zeros_like(tx_c)
        for k, (rng, vel, rcs, az) in enumerate(
            zip(ranges, velocities, rcs_vals, azimuths, strict=True)
        ):
            echo = _apply_single_target_echo(
                tx_c,
                range_m=rng,
                velocity_mps=vel,
                rcs=rcs,
                azimuth_deg=az,
                position_rx_m=pos_rx,
                samp_rate=float(params.samp_rate),
                center_freq=params.center_freq,
                rndm_phaseshift=params.rndm_phaseshift,
                generator=generator,
            )
            out = out + echo

        if params.self_coupling:
            coupling = 10.0 ** (params.self_coupling_db / 20.0)
            out = out + coupling * tx_c
        rx_outputs.append(out)

    if len(rx_outputs) == 1:
        return rx_outputs[0]
    return torch.stack(rx_outputs, dim=0)
