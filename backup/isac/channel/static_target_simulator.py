"""gr-radar static_target_simulator_cc 的 Torch 复现（点目标散射 + FFT 分数时延）。

处理链（单目标）：时域多普勒 chirp → FFT → 频域距离/方位分数时延滤波 → IFFT；
多目标与多 RX 分别累加回波；可选自耦合直通与随机初相（与 gr-radar 块一致）。
仅接受时域 IQ（末维为样点），不经 OFDM 资源网格。
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from scipy.constants import c

# √(4π)³：雷达方程幅度标定中的 (4π)^(3/2) 因子，见 _target_scale_ampl
_FOUR_PI_CUBED_SQRT = math.sqrt((4.0 * math.pi) ** 3)
_TWO_PI = 2.0 * math.pi


def _as_float_vector(values: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        return (float(values),)
    seq = tuple(float(v) for v in values)
    if not seq:
        raise ValueError(f"{name} 不能为空")
    return seq


@dataclass(frozen=True)
class StaticTargetParams:
    """与 gr-radar static_target_simulator_cc 块参数对应。

    ``range_m`` / ``velocity_mps`` / ``rcs`` / ``azimuth_deg`` 须等长（多目标）；
    ``position_rx_m`` 为各 RX 天线横向位置 (m)，决定方位时延。
    ``self_coupling_db``：自耦合直通幅度 (dB)；``rndm_phaseshift``：每目标随机初相。
    """

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


class StaticTargetSimulator:
    """gr-radar static_target_simulator_cc 的 Torch 复现。

    用法：``StaticTargetSimulator(params)(tx)``；``tx`` 为时域复基带，末维长度须与
    CPI 样点数一致（如 ``num_ofdm_symbols * (fft_size + cp_len)``）。
    """

    def __init__(self, params: StaticTargetParams) -> None:
        self.params = params

    @staticmethod
    def _fft_freq_bins(n: int, samp_rate: float, device: torch.device) -> torch.Tensor:
        """FFT 频域 bin 频率 (Hz)，与 gr-radar range/azimuth 滤波器索引一致。"""
        i = torch.arange(n, device=device, dtype=torch.float64)
        half = n // 2
        fs = float(samp_rate)
        return torch.where(i < half, i * fs / n, i * fs / n - fs)

    @staticmethod
    def _build_delay_filter_from_freq(
        freq: torch.Tensor,
        delay_s: float,
        *,
        compensate_numpy_ifft: bool,
    ) -> torch.Tensor:
        """由预计算的频域 bin 构建分数时延滤波器。"""
        delay = float(delay_s)
        # H(f) = exp(-j 2π f τ)，分数时延 τ 不必为整数样点
        phase = torch.fmod(_TWO_PI * delay * freq, _TWO_PI)
        inv_n = 1.0 / float(freq.numel()) if compensate_numpy_ifft else 1.0
        return (inv_n * (torch.cos(phase) - 1j * torch.sin(phase))).to(torch.complex64)

    @staticmethod
    def _build_doppler_filter(
        n: int,
        doppler_hz: float,
        scale_ampl: float,
        samp_rate: float,
        device: torch.device,
    ) -> torch.Tensor:
        """多普勒时域 chirp：scale · exp(j·2π·f_d·i/f_s)，与 gr-radar 逐步 fmod 相位等价。"""
        phase_inc = _TWO_PI * doppler_hz / samp_rate
        scale = float(scale_ampl)
        if n == 0:
            return torch.empty(0, device=device, dtype=torch.complex64)
        i = torch.arange(n, device=device, dtype=torch.float64)
        return (scale * torch.exp(1j * i * phase_inc)).to(torch.complex64)

    @staticmethod
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
        freq = StaticTargetSimulator._fft_freq_bins(n, samp_rate, device)
        return StaticTargetSimulator._build_delay_filter_from_freq(
            freq, delay_s, compensate_numpy_ifft=compensate_numpy_ifft
        )

    @staticmethod
    def _target_scale_ampl(range_m: float, rcs: float, center_freq: float) -> float:
        """单目标回波幅度标量：c·√σ / ((4π)^(3/2)·R²·f_c)，σ 为 RCS。"""
        return c * math.sqrt(rcs) / _FOUR_PI_CUBED_SQRT / (range_m * range_m) / center_freq

    @staticmethod
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
        """单目标回波：多普勒 × 距离分数时延 × 方位分数时延（与 gr-radar 三步滤波同序）。"""
        n = tx.shape[-1]
        device = tx.device

        # 双程：多普勒 f_d = 2 v f_c / c；距离时延 τ = 2R/c
        doppler_hz = 2.0 * velocity_mps * center_freq / c
        timeshift_s = 2.0 * range_m / c
        # 方位：横向天线位置 × sin(方位角) → 等效时延 (s)
        azimuth_shift_s = position_rx_m * math.sin(math.radians(azimuth_deg))
        scale_ampl = StaticTargetSimulator._target_scale_ampl(range_m, rcs, center_freq)

        doppler_filt = StaticTargetSimulator._build_doppler_filter(
            n, doppler_hz, scale_ampl, samp_rate, device
        )
        freq_bins = StaticTargetSimulator._fft_freq_bins(n, samp_rate, device)
        range_filt = StaticTargetSimulator._build_delay_filter_from_freq(
            freq_bins, timeshift_s, compensate_numpy_ifft=False
        )
        azimuth_filt = StaticTargetSimulator._build_delay_filter_from_freq(
            freq_bins, azimuth_shift_s, compensate_numpy_ifft=False
        )

        # 时域 chirp → 频域乘距离/方位滤波器 → 回时域（compensate_numpy_ifft=False 对齐 torch.ifft）
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

    def __call__(
        self,
        tx: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """模拟 gr-radar static_target_simulator_cc 接收 IQ。

        Parameters
        ----------
        tx:
            时域发射 IQ，末维为样点；支持 ``(N,)`` 或 ``(..., N)``。
        generator:
            随机相位生成器（``rndm_phaseshift=True`` 时可选，用于可复现）。

        Returns
        -------
        torch.Tensor
            单 RX（``num_rx==1``）时与 ``tx`` 同 shape；多 RX 时为 ``(num_rx, *tx.shape)``。
        """
        params = self.params
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
            for rng, vel, rcs_val, az in zip(
                ranges, velocities, rcs_vals, azimuths, strict=True
            ):
                echo = self._apply_single_target_echo(
                    tx_c,
                    range_m=rng,
                    velocity_mps=vel,
                    rcs=rcs_val,
                    azimuth_deg=az,
                    position_rx_m=pos_rx,
                    samp_rate=float(params.samp_rate),
                    center_freq=params.center_freq,
                    rndm_phaseshift=params.rndm_phaseshift,
                    generator=generator,
                )
                out = out + echo

            # 自耦合：未进入散射链的发射副本（默认 -10 dB）
            if params.self_coupling:
                coupling = 10.0 ** (params.self_coupling_db / 20.0)
                out = out + coupling * tx_c
            rx_outputs.append(out)

        if len(rx_outputs) == 1:
            return rx_outputs[0]
        return torch.stack(rx_outputs, dim=0)
