"""
感知相关组件构建
"""

from dataclasses import dataclass, asdict

from sionna.phy.ofdm import ResourceGrid

from ...params.ofdm import OFDMParams
from ...params.sensing import SensingParams
from ....sensing.sensing_performance import SensingPerformance
from ....sensing.music_estimator import MUSICEstimator
from ....sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ....sensing.cfar import CFARDetector
from ....sensing.clutter_suppression import MovingTargetIndication, MovingTargetDetection


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
        ofdm_params: OFDMParams,
        sensing_params: SensingParams,
        rg: ResourceGrid,
        device: str,
    ) -> "SensingComponents":
        sensing_performance = SensingPerformance(
            resource_grid=rg,
            carrier_frequency=ofdm_params.carrier_frequency,
        )
        moving_target_indication = MovingTargetIndication(
            sensing_performance,
            filter_order=1,
        )
        moving_target_detection = MovingTargetDetection(
            sensing_performance,
            ofdm_params.carrier_frequency,
        )
        delay_doppler_spectrum = DelayDopplerSpectrum(
            sensing_performance=sensing_performance,
            delay_window=sensing_params.windows.delay_window,
            doppler_window=sensing_params.windows.doppler_window,
        )
        music_estimator = MUSICEstimator(
            device=device, sensing_performance=sensing_performance
        )
        cfar_inst = CFARDetector(**asdict(sensing_params.cfar))

        return cls(
            sensing_performance=sensing_performance,
            delay_doppler_spectrum=delay_doppler_spectrum,
            music_estimator=music_estimator,
            cfar=cfar_inst,
            moving_target_indication=moving_target_indication,
            moving_target_detection=moving_target_detection,
        )
