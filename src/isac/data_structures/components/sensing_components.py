"""
感知相关组件构建（与 ``sensing_params`` 对应）
"""

from dataclasses import dataclass

from sionna.phy.ofdm import ResourceGrid

from ..params import SystemParams
from ...sensing.sensing_performance import SensingPerformance
from ...sensing.music_estimator import MUSICEstimator
from ...sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ...sensing.cfar import CFARDetector
from ...sensing.clutter_suppression import MovingTargetIndication, MovingTargetDetection


@dataclass
class SensingComponents:
    """感知组件"""

    sensing_performance: SensingPerformance
    delay_doppler_spectrum: DelayDopplerSpectrum
    music_estimator: MUSICEstimator
    cfar: CFARDetector
    moving_target_indication: MovingTargetIndication
    moving_target_detection: MovingTargetDetection

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        rg: ResourceGrid,
        device: str,
    ) -> "SensingComponents":
        """根据 ``SystemParams`` 构建感知性能、时延多普勒谱、MUSIC、CFAR 与 MTI/MTD 运行时实例。"""
        sensing_performance = SensingPerformance(
            resource_grid=rg,
            carrier_frequency=system_params.carrier_frequency,
        )
        moving_target_indication = MovingTargetIndication(
            sensing_performance,
            filter_order=1,
        )
        moving_target_detection = MovingTargetDetection(
            sensing_performance,
            system_params.carrier_frequency,
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

        return cls(
            sensing_performance=sensing_performance,
            delay_doppler_spectrum=delay_doppler_spectrum,
            music_estimator=music_estimator,
            cfar=cfar_inst,
            moving_target_indication=moving_target_indication,
            moving_target_detection=moving_target_detection,
        )
