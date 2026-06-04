import math
import torch
from typing import Callable, Optional, Tuple, Union

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import (
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    ApplyOFDMChannel,
    ApplyTimeChannel,
)
from sionna.phy.utils import ebnodb2no
from sionna.rt import Paths


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
        from ..utils.channel_paths import paths_cfr_per_tx_torch

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

    @staticmethod
    def snr_db_to_ebno_db(
        snr_db: float,
        num_bits_per_symbol: int,
        coderate: float = 1.0,
    ) -> float:
        """Es/N0 (dB) → Eb/N0 (dB)，与 ``ebnodb2no`` 内 :math:`E_s/N_0 = E_b/N_0 \\cdot r M` 一致。"""
        if coderate <= 0 or num_bits_per_symbol <= 0:
            raise ValueError("coderate 与 num_bits_per_symbol 须为正")
        return snr_db - 10.0 * math.log10(coderate * num_bits_per_symbol)

    @staticmethod
    def noise_power_from_snr_db(
        snr_db: float,
        num_bits_per_symbol: int,
        coderate: float,
        resource_grid: ResourceGrid,
    ) -> torch.Tensor:
        """``snr_db`` (Es/N0) → ``ebno_db`` 后调用 notebook 公式 ``ebnodb2no``。"""
        ebno_db = Channel.snr_db_to_ebno_db(snr_db, num_bits_per_symbol, coderate)
        return ebnodb2no(
            ebno_db,
            num_bits_per_symbol,
            coderate,
            resource_grid,
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        domain: str = "time",
        *,
        snr_db: Optional[float] = None,
        num_bits_per_symbol: int = 2,
        coderate: float = 1.0,
    ) -> torch.Tensor:
        """经信道并可选叠加 AWGN（与 ``MIMO_OFDM_Transmissions_over_CDL`` notebook 一致）。

        配置 ``snr_db`` (Es/N0 dB) 换算为 ``ebno_db`` 后
        ``no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, self.rg)``。
        未传 ``snr_db`` 时不加噪。
        """
        no: Optional[Union[float, torch.Tensor]] = None
        if snr_db is not None:
            no = self.noise_power_from_snr_db(
                snr_db, num_bits_per_symbol, coderate, self.rg
            )

        paths = self.paths()
        a, tau = paths.cir(
            num_time_steps=self.rg.num_time_samples + self.l_tot - 1,
            sampling_frequency=self.rg.bandwidth,
            normalize_delays=False,
            out_type="torch",
        )

        if domain == "time":
            return self.channel_time(inputs, self.h_time, 0)
        if domain == "frequency":
            return self.channel_freq(inputs, self.h_freq, 0)
        raise ValueError(f"不支持的域: {domain}。支持的值: 'time', 'frequency'")
