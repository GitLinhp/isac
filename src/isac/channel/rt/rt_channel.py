"""Sionna RT 射线追踪信道：多径 CIR/CFR 与时/频域施加。"""

from typing import Callable, Optional, Tuple

import torch
from sionna.phy.channel import (
    ApplyOFDMChannel,
    ApplyTimeChannel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
)
from sionna.phy.config import Precision
from sionna.phy.ofdm import ResourceGrid
from sionna.rt import Paths

from ..channel import Channel


class RTChannel(Channel):
    """RT 信道：CIR/CFR 计算与 Sionna 时/频域卷积施加。"""

    def __init__(
        self,
        rg: ResourceGrid,
        paths: Callable[[], Paths],
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self.rg = rg

        assert callable(paths), "paths 必须是可调用对象：Callable[[], Paths]"
        self.paths = paths

        self._init_properties()
        self._init_components()

    def _init_properties(self) -> None:
        self.l_min, self.l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.l_tot = self.l_max - self.l_min + 1
        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size,
            self.rg.subcarrier_spacing,
        )

    def _init_components(self) -> None:
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.channel_time = ApplyTimeChannel(
            self.rg.num_time_samples, l_tot=self.l_tot, add_awgn=True
        )

    def cir(
        self, num_time_steps: int, sampling_frequency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        paths = self.paths()
        a, tau = paths.cir(
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency,
            normalize_delays=False,
            out_type="torch",
        )
        a = torch.unsqueeze(a, dim=0)
        tau = torch.unsqueeze(tau, dim=0)
        return a, tau

    def cfr(
        self, num_time_steps: int, sampling_frequency: float, out_type: str = "torch"
    ):
        """获取信道频率响应"""
        paths = self.paths()
        return paths.cfr(
            frequencies=self.frequencies,
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency,
            normalize_delays=False,
            normalize=True,
            out_type=out_type,
        )

    @property
    def h_time(self) -> torch.Tensor:
        a, tau = self.cir(
            num_time_steps=self.rg.num_time_samples + self.l_tot - 1,
            sampling_frequency=self.rg.bandwidth,
        )
        return cir_to_time_channel(
            self.rg.bandwidth,
            a,
            tau,
            self.l_min,
            self.l_max,
            normalize=False,
        )

    @property
    def h_freq(self) -> torch.Tensor:
        a, tau = self.cir(
            num_time_steps=self.rg.num_ofdm_symbols,
            sampling_frequency=1 / self.rg.ofdm_symbol_duration,
        )
        return cir_to_ofdm_channel(self.frequencies, a, tau, normalize=False)

    def _apply_clean(self, inputs: torch.Tensor, domain: str) -> torch.Tensor:
        if domain == "time":
            return self.channel_time(inputs, self.h_time, None)
        if domain == "frequency":
            return self.channel_freq(inputs, self.h_freq, None)
        raise ValueError(f"不支持的域: {domain}。支持的值: 'time', 'frequency'")
