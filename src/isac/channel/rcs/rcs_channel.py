"""gr-radar static_target_simulator_cc 的 Torch 复现（点目标散射 + FFT 分数时延）。

处理链：时域多普勒 chirp → FFT → 频域距离/方位分数时延滤波 → IFFT；
可选自耦合直通与随机初相（与 gr-radar 块一致）。
仅接受时域 IQ（末维为样点），不经 OFDM 资源网格。
"""

import math
from typing import Callable, Optional

import torch
from scipy.constants import c
from sionna.phy.config import Precision

from .rcs_scene import RCSScene
from .rcs_target import RCSTarget
from ..channel import Channel

_FOUR_PI_CUBED_SQRT = math.sqrt((4.0 * math.pi) ** 3)
_TWO_PI = 2.0 * math.pi


class RCSChannel(Channel):
    """gr-radar static_target_simulator_cc 的 Torch 复现。

    用法：``RCSChannel(rcs_scene, center_freq=..., samp_rate=...)(tx)``；``rcs_scene`` 为
    ``Callable[[], RCSScene]``，每次施加信道时读取当前场景与点目标状态（与 ``RTChannel.paths`` 同模式）。
    """

    def __init__(
        self,
        rcs_scene: Callable[[], RCSScene],
        *,
        center_freq: float,
        samp_rate: float,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        if not callable(rcs_scene):
            raise TypeError("rcs_scene 必须是可调用对象：Callable[[], RCSScene]")
        self.rcs_scene = rcs_scene
        if center_freq <= 0:
            raise ValueError("center_freq 须为正")
        if samp_rate <= 0:
            raise ValueError("samp_rate 须为正")
        self.center_freq = float(center_freq)
        self.samp_rate = float(samp_rate)
        self._generator: torch.Generator | None = None

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
        """频域分数时延滤波器。"""
        freq = RCSChannel._fft_freq_bins(n, samp_rate, device)
        return RCSChannel._build_delay_filter_from_freq(
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

        doppler_hz = -2.0 * velocity_mps * center_freq / c
        timeshift_s = 2.0 * range_m / c
        azimuth_shift_s = position_rx_m * math.sin(math.radians(azimuth_deg))
        scale_ampl = RCSChannel._target_scale_ampl(range_m, rcs, center_freq)

        doppler_filt = RCSChannel._build_doppler_filter(
            n, doppler_hz, scale_ampl, samp_rate, device
        )
        freq_bins = RCSChannel._fft_freq_bins(n, samp_rate, device)
        range_filt = RCSChannel._build_delay_filter_from_freq(
            freq_bins, timeshift_s, compensate_numpy_ifft=False
        )
        azimuth_filt = RCSChannel._build_delay_filter_from_freq(
            freq_bins, azimuth_shift_s, compensate_numpy_ifft=False
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

    def _apply_clean(self, inputs: torch.Tensor, domain: str) -> torch.Tensor:
        if domain != "time":
            raise ValueError("channel.type='rcs' 仅支持 domain='time'")

        if inputs.shape[-1] == 0:
            raise ValueError("tx 末维样点数须为正")

        tx_c = inputs.to(torch.complex64)
        scene = self.rcs_scene()
        tgt = scene.target

        out = self._apply_single_target_echo(
            tx_c,
            range_m=tgt.range_m,
            velocity_mps=tgt.velocity_mps,
            rcs=tgt.rcs,
            azimuth_deg=tgt.azimuth_deg,
            position_rx_m=tgt.position_rx_m,
            samp_rate=self.samp_rate,
            center_freq=self.center_freq,
            rndm_phaseshift=scene.rndm_phaseshift,
            generator=self._generator,
        )

        if scene.self_coupling:
            coupling = 10.0 ** (scene.self_coupling_db / 20.0)
            out = out + coupling * tx_c

        return out

    def __call__(
        self,
        inputs: torch.Tensor,
        domain: str = "time",
        *,
        snr_db: Optional[float] = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """模拟 gr-radar static_target_simulator_cc 接收 IQ（可选 AWGN）。"""
        self._generator = generator
        return super().__call__(inputs, domain=domain, snr_db=snr_db)
