import torch
from typing import Callable, Optional, Tuple

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import (
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    ApplyOFDMChannel,
    ApplyTimeChannel,
)
from sionna.rt import Paths

from ..utils.channel_paths import paths_cfr_per_tx_torch
from .awgn import AWGN


class Channel:
    """信道类

    提供信道响应计算和路径分析功能，包括信道冲激响应(CIR)和
    信道频率响应(CFR)的计算。
    """

    def __init__(
        self,
        rg: ResourceGrid,
        paths: Callable[[], Paths],
    ):
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
        self._awgn = AWGN()

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

    def cfr_per_tx(
        self,
        rt_scene: object,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离的 OFDM 频域信道，每个 TX 对应 ``(num_ofdm_symbols, fft_size)``。"""
        return paths_cfr_per_tx_torch(
            self.rg,
            rt_scene,
            device=device,
            dtype=dtype,
        )

    # 参数属性
    @property
    def h_time(self) -> torch.Tensor:
        # 计算信道冲激响应
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

    def __call__(
        self,
        inputs: torch.Tensor,
        domain: str = "time",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """经信道并可选叠加 AWGN。

        未传 ``snr_db`` 时不加噪。传入时按**接收端信号功率**定标噪声：

        ``no = E[|y_clean|^2] / 10^(snr_db/10)``，使配置 ``snr_db`` 为实际接收 SNR (dB)。

        """
        if domain == "time":
            y_clean = self.channel_time(inputs, self.h_time, None)
            if snr_db is None:
                return y_clean
            return self._awgn(y_clean, snr_db)

        elif domain == "frequency":
            y_clean = self.channel_freq(inputs, self.h_freq, None)
            if snr_db is None:
                return y_clean
            return self._awgn(y_clean, snr_db)

        else:
            raise ValueError(f"不支持的域: {domain}。支持的值: 'time', 'frequency'")
