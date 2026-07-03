"""Sionna RT 射线追踪信道：多径 CIR/CFR 与时/频域施加。"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

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
    """RT 信道：CIR/CFR 计算与 Sionna 时/频域卷积施加。

    可通过 ``cir`` / ``cfr`` 属性注入预计算响应；未注入时由 ``get_cir`` / ``get_cfr``
    从 ``paths`` 实时求解。
    """

    def __init__(
        self,
        rg: ResourceGrid,
        paths: Callable[[], Paths],
        *,
        rx_names: Callable[[], list[str]] | None = None,
        tx_names: Callable[[], list[str]] | None = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """初始化 RT 信道

        参数:
        -------
        - rg: ResourceGrid
            资源网格
        - paths: Callable[[], Paths]
            路径生成器
        - rx_names: Callable[[], list[str]] | None
            接收机名称列表提供器，供 ``cfr_split`` 构造 ``{rx}-{tx}`` 键
        - tx_names: Callable[[], list[str]] | None
            发射机名称列表提供器，供 ``cfr_split`` 构造 ``{rx}-{tx}`` 键
        - precision: Optional[Precision]
            精度，可选 ``"float32"`` 或 ``"float64"``，默认 ``None``
        - device: Optional[str]
            设备类型，可选 ``"cpu"`` 或 ``"cuda"``，默认 ``None``

        CIR/CFR 注入请使用 ``channel.cir = ...`` / ``channel.cfr = ...``（初始均为 ``None``）。
        """
        super().__init__(precision=precision, device=device)
        self.rg = rg

        assert callable(paths), "paths 必须是可调用对象：Callable[[], Paths]"
        self.paths = paths
        self._rx_names = rx_names
        self._tx_names = tx_names
        self._cir: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cfr: torch.Tensor | None = None

        self._init_properties()
        self._init_components()

    # ==================== 初始化方法 ====================
    def _init_properties(self) -> None:
        self.l_min, self.l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.l_tot = self.l_max - self.l_min + 1
        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size,
            self.rg.subcarrier_spacing,
        )

    def _init_components(self) -> None:
        self.channel_freq = ApplyOFDMChannel(add_awgn=False)
        self.channel_time = ApplyTimeChannel(
            self.rg.num_time_samples, l_tot=self.l_tot, add_awgn=False
        )

    # ==================== CIR / CFR 注入属性 ====================
    @property
    def cir(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """注入的 CIR ``(a, tau)``；``None`` 表示每次从 ``paths`` 计算。"""
        return self._cir

    @cir.setter
    def cir(self, value: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        self._cir = value

    @property
    def cfr(self) -> torch.Tensor | None:
        """注入的 CFR；``None`` 表示每次从 ``paths`` 计算。"""
        return self._cfr

    @cfr.setter
    def cfr(self, value: torch.Tensor | None) -> None:
        self._cfr = value

    # ==================== 计算方法 ====================
    def get_cir(
        self, num_time_steps: int, sampling_frequency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 CIR；已注入 ``cir`` 时直接返回注入值。"""
        if self._cir is not None:
            return self._cir
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

    def get_cfr(
        self, num_time_steps: int, sampling_frequency: float, out_type: str = "torch"
    ) -> Any:
        """获取信道频率响应；已注入 ``cfr`` 时直接返回注入值（仅 ``out_type='torch'``）。

        ``out_type='torch'`` 时通常为 7D：
        ``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_frequencies]``；
        单天线合成阵列下也可能为 6D：
        ``[batch, num_rx, num_tx, num_rx_ant, num_time_steps, num_frequencies]``。
        """
        if self._cfr is not None:
            if out_type != "torch":
                raise ValueError(
                    "注入 cfr 仅支持 out_type='torch'，"
                    f"收到 {out_type!r}；请先清除 channel.cfr 或改用 torch。"
                )
            return self._cfr
        paths = self.paths()
        return paths.cfr(
            frequencies=self.frequencies,
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency,
            normalize_delays=False,
            normalize=True,
            out_type=out_type,
        )

    @staticmethod
    def cfr_pair_key(rx_name: str, tx_name: str) -> str:
        """收发机对字典键：``{rx_name}-{tx_name}``。"""
        return f"{rx_name}-{tx_name}"

    def cfr_split(
        self,
        num_time_steps: int,
        sampling_frequency: float,
        out_type: str = "torch",
    ) -> dict[str, torch.Tensor]:
        """计算 CFR 并按 (RX, TX) 对切片。

        参数与 ``get_cfr()`` 一致；返回键为 ``cfr_pair_key(rx, tx)``，
        值为 ``(num_time_steps, num_frequencies)`` 张量。
        """
        if self._rx_names is None or self._tx_names is None:
            raise ValueError("cfr_split 需要构造 RTChannel 时提供 rx_names 与 tx_names")
        rx_name_list = self._rx_names()
        tx_name_list = self._tx_names()

        cfr = self.get_cfr(
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency,
            out_type=out_type,
        )
        if not isinstance(cfr, torch.Tensor):
            raise TypeError(
                f"cfr_split out_type='torch' 须返回 Tensor，收到 {type(cfr)!r}"
            )

        if cfr.ndim == 7:
            num_rx = int(cfr.shape[1])
            num_tx = int(cfr.shape[3])
            tx_axis = 3
        elif cfr.ndim == 6:
            num_rx = int(cfr.shape[1])
            num_tx = int(cfr.shape[2])
            tx_axis = 2
        else:
            raise ValueError(
                "cfr 须为 6D (batch, num_rx, num_tx, num_rx_ant, S, F) 或 "
                "7D (batch, num_rx, num_rx_ant, num_tx, num_tx_ant, S, F)，"
                f"收到 ndim={cfr.ndim}, shape={tuple(cfr.shape)}"
            )
        if num_rx != len(rx_name_list):
            raise ValueError(
                f"CFR 的 rx 维 ({num_rx}) 与 rx_names 数量 ({len(rx_name_list)}) 不一致"
            )
        if num_tx != len(tx_name_list):
            raise ValueError(
                f"CFR 的 tx 维 ({num_tx}) 与 tx_names 数量 ({len(tx_name_list)}) 不一致"
            )

        out: dict[str, torch.Tensor] = {}
        for rx_i, rx_name in enumerate(rx_name_list):
            for tx_i, tx_name in enumerate(tx_name_list):
                if tx_axis == 3:
                    slab = cfr[0, rx_i, 0, tx_i, 0]
                else:
                    slab = cfr[0, rx_i, tx_i, 0]
                out[self.cfr_pair_key(rx_name, tx_name)] = slab
        return out

    # ==================== 属性 ====================
    @property
    def h_time(self) -> torch.Tensor:
        a, tau = self.get_cir(
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
        if self._cfr is not None:
            return self._cfr
        a, tau = self.get_cir(
            num_time_steps=self.rg.num_ofdm_symbols,
            sampling_frequency=1 / self.rg.ofdm_symbol_duration,
        )
        return cir_to_ofdm_channel(self.frequencies, a, tau, normalize=False)

    def _apply_channel(self, inputs: torch.Tensor, domain: str) -> torch.Tensor:
        """施加信道

        参数:
        ----------
        - inputs: torch.Tensor
            输入信号
        - domain: str
            域，可选 ``"time"`` 或 ``"frequency"``

        返回:
        -------
        - torch.Tensor
            输出信号
        """
        if domain == "time":
            return self.channel_time(inputs, self.h_time)
        elif domain == "frequency":
            return self.channel_freq(inputs, self.h_freq)
        else:
            raise ValueError(f"不支持的域: {domain}。支持的值: 'time', 'frequency'")
