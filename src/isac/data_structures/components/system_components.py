from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..params import SystemParams

from .channel import ChannelComponents
from .ofdm import OFDMComponents
from .rt_scene import RTSceneComponents
from .sensing import SensingComponents
from .static_target import StaticTargetComponents


@dataclass
class SystemComponents:
    """系统组件（五字段嵌套）"""

    ofdm: OFDMComponents
    channel: ChannelComponents
    sensing: SensingComponents
    rt_scene: Optional[object] = None
    static_target: Optional[object] = None

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str = "cuda:0",
    ) -> "SystemComponents":
        ofdm = OFDMComponents.build_from_params(system_params.ofdm, device=device)

        rt_comp = RTSceneComponents.build_from_params(
            system_params.rt_scene,
            carrier_frequency=system_params.ofdm.carrier_frequency,
            resource_grid=ofdm.rg,
        )
        st_comp = StaticTargetComponents.build_from_params(
            system_params.static_target,
            system_params.ofdm,
            ofdm.rg,
        )

        static_sim = st_comp.simulator if st_comp is not None else None
        channel = ChannelComponents.build_from_params(
            system_params.channel,
            ofdm.rg,
            rt_comp.rt_scene,
            static_sim,
        )
        sensing = SensingComponents.build_from_params(
            system_params.ofdm,
            system_params.sensing,
            ofdm.rg,
            device=device,
        )

        return cls(
            ofdm=ofdm,
            channel=channel,
            sensing=sensing,
            rt_scene=rt_comp.rt_scene,
            static_target=static_sim,
        )
