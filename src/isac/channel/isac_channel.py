"""统一信道组件：RT / RCS 分发 + AWGN。"""

from typing import Literal, Optional

import torch

from .awgn import AWGN
from .channel import Channel
from .static_target_simulator import StaticTargetSimulator

_SNR_DEFAULT = object()


class IsacChannel:
    """ISAC 信道：``__call__`` 集成 RT 射线追踪或 RCS 点目标仿真，并按 TOML ``snr_db`` 加 AWGN。"""

    def __init__(
        self,
        channel_type: Literal["rt", "rcs"],
        default_snr_db: float,
        *,
        rt: Optional[Channel] = None,
        static_target_sim: Optional[StaticTargetSimulator] = None,
        awgn: Optional[AWGN] = None,
    ) -> None:
        self.channel_type = channel_type
        self.default_snr_db = float(default_snr_db)
        self._rt = rt
        self._static_target_sim = static_target_sim
        self._awgn = awgn if awgn is not None else AWGN()

    @property
    def static_target_sim(self) -> Optional[StaticTargetSimulator]:
        return self._static_target_sim

    def __call__(
        self,
        inputs: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] | object = _SNR_DEFAULT,
    ) -> torch.Tensor:
        """经信道并加 AWGN；未传 ``snr_db`` 时使用构造时的 ``default_snr_db``；显式 ``snr_db=None`` 则不加噪。"""
        if self.channel_type == "rcs":
            if domain != "time":
                raise ValueError("channel.type='rcs' 仅支持 domain='time'")
            y_clean = self._static_target_sim(inputs)
        else:
            if self._rt is None:
                raise ValueError("channel.type='rt' 要求已构建 RT 后端")
            y_clean = self._rt(inputs, domain=domain, snr_db=None)

        effective_snr = (
            self.default_snr_db if snr_db is _SNR_DEFAULT else snr_db
        )
        if effective_snr is None:
            return y_clean
        return self._awgn(y_clean, effective_snr)

    def cfr_per_tx(
        self,
        rt_scene: object,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离的 OFDM 频域信道（仅 ``channel_type='rt'``）。"""
        if self.channel_type != "rt" or self._rt is None:
            raise ValueError("cfr_per_tx 仅适用于 channel.type='rt'")
        return self._rt.cfr_per_tx(
            rt_scene,
            device=device,
            dtype=dtype,
        )
