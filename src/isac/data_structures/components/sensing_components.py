"""
感知相关组件构建（与 ``sensing_params`` 对应）
"""

import math
from dataclasses import dataclass
from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ..params import SystemParams
from ...zc_source import ZCSource
from ...sensing.sensing_performance import SensingPerformance
from ...sensing.music_estimator import MUSICEstimator
from ...sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ...sensing.cfar import CFARDetector


@dataclass
class SensingComponents:
    """感知组件"""

    sensing_performance: SensingPerformance
    delay_doppler_spectrum: DelayDopplerSpectrum
    music_estimator: MUSICEstimator
    cfar: CFARDetector
    zc_source: Optional[ZCSource] = None


def build_sensing_components(
    system_params: SystemParams,
    rg: ResourceGrid,
    device: str,
) -> SensingComponents:
    """根据 ``SystemParams`` 构建感知性能、时延多普勒谱、MUSIC 与 CFAR 运行时实例。"""
    sensing_performance = SensingPerformance(
        resource_grid=rg,
        carrier_frequency=system_params.carrier_frequency,
    )
    delay_doppler_spectrum = DelayDopplerSpectrum(
        sensing_performance=sensing_performance,
        delay_window=system_params.sensing.windows.delay_window,
        doppler_window=system_params.sensing.windows.doppler_window,
    )
    music_estimator = MUSICEstimator(
        device=device, sensing_performance=sensing_performance
    )
    p = system_params.sensing.cfar
    cfar_inst = CFARDetector(
        cfar_type=p.cfar_type,
        guard=p.guard,
        trailing=p.trailing,
        pfa=p.pfa,
        detector=p.detector,
        offset=p.offset,
        k=p.k,
    )

    src = system_params.sensing.source
    zc_inst: Optional[ZCSource] = None
    if src.type == "zc":
        n_data = rg.num_data_symbols
        u = src.root_index
        if math.gcd(u, n_data) != 1:
            raise ValueError(
                "sensing.source: ZC requires gcd(root_index, rg.num_data_symbols)=1; "
                f"got root_index={u}, num_data_symbols={n_data}"
            )
        zc_inst = ZCSource(
            root_index=u,
            normalize=src.normalize,
            device=device,
        )

    return SensingComponents(
        sensing_performance=sensing_performance,
        delay_doppler_spectrum=delay_doppler_spectrum,
        music_estimator=music_estimator,
        cfar=cfar_inst,
        zc_source=zc_inst,
    )
