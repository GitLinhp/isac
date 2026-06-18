"""
信道运行时组件构建（RT 或 RCS 点目标）
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from sionna.phy.ofdm import ResourceGrid

from ...params.channel import ChannelParams
from ....channel.awgn import AWGN
from ....channel.channel import Channel
from ....channel.rt.rt_scene import RTScene
from ....channel.static_target_simulator import StaticTargetSimulator


@dataclass
class ChannelComponents:
    """信道施加组件"""

    channel_type: Literal["rt", "rcs"]
    rt_channel: Optional[Channel] = None
    static_sim: Optional[StaticTargetSimulator] = None
    _awgn: AWGN = field(default_factory=AWGN)

    def __call__(
        self,
        inputs: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        if self.channel_type == "rcs":
            if domain != "time":
                raise ValueError("channel.type='rcs' 仅支持 domain='time'")
            y_clean = self.static_sim(inputs)
        else:
            y_clean = self.rt_channel(inputs, domain=domain, snr_db=None)
        if snr_db is None:
            return y_clean
        return self._awgn(y_clean, snr_db)

    @classmethod
    def build_from_params(
        cls,
        channel_params: ChannelParams,
        rg: ResourceGrid,
        rt_scene: Optional[RTScene],
        static_target: Optional[StaticTargetSimulator],
    ) -> "ChannelComponents":
        if channel_params.type == "rt":
            if rt_scene is None:
                raise ValueError("channel.type='rt' 要求已构建 rt_scene")
            return cls(
                channel_type="rt",
                rt_channel=Channel(rg=rg, paths=lambda: rt_scene.paths),
            )
        if static_target is None:
            raise ValueError("channel.type='rcs' 要求已构建 static_target")
        return cls(
            channel_type="rcs",
            static_sim=static_target,
        )

    def cfr_per_tx(
        self,
        rt_scene: RTScene,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离的 OFDM 频域信道（仅 ``channel.type='rt'``）。"""
        if self.channel_type != "rt" or self.rt_channel is None:
            raise ValueError("cfr_per_tx 仅适用于 channel.type='rt'")
        return self.rt_channel.cfr_per_tx(
            rt_scene,
            device=device,
            dtype=dtype,
        )
