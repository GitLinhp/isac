"""
静态点目标散射运行时组件构建
"""

from dataclasses import dataclass
from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ...params.ofdm import OFDMParams
from ...params.static_target import StaticTargetConfig, StaticTargetParams
from ....channel.static_target_simulator import StaticTargetSimulator


@dataclass
class StaticTargetComponents:
    """静态点目标组件"""

    params: StaticTargetParams
    simulator: StaticTargetSimulator

    @classmethod
    def build_from_params(
        cls,
        static_target_config: Optional[StaticTargetConfig],
        ofdm_params: OFDMParams,
        rg: ResourceGrid,
    ) -> Optional["StaticTargetComponents"]:
        if static_target_config is None:
            return None
        params = static_target_config.to_params(
            samp_rate=int(rg.bandwidth),
            center_freq=ofdm_params.carrier_frequency,
        )
        return cls(params=params, simulator=StaticTargetSimulator(params))
