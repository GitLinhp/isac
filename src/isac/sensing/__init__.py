"""感知 DSP 子包：DD 谱、CFAR、MUSIC、杂波抑制与定位。

典型导入::

    from isac.sensing import DelayDopplerSpectrum, MUSICEstimator, SensingEstimator, match_peaks_and_compute_radial_rmse, CFARDetector
    from isac.sensing.geometry import delay_to_range, doppler_to_velocity
"""

from ..data_structures.types import MetricMode, MusicPeaks, RoiSlices, SensingEstimate, SensMode
from .spectrum import (
    DelayDopplerRoi,
    DelayDopplerSpectrum,
    LSChannelEstimator,
    SensingPerformance,
)
from .detection import CFARDetector
from .detection.music_estimator import MUSICEstimator
from .evaluation import (
    SensingEstimator,
    match_peaks_and_compute_radial_rmse,
)
from .clutter import MovingTargetIndication, MovingTargetDetection
from .localization import (
    ground_circle_radius_sq,
    intersect_circles_xy,
    localize_xy_z0_colocated_tx_mono_bistatic,
    position_rmse_xy,
    select_xy_solution,
)
from .geometry import (
    MONOSTATIC_TX_RX_EPS_M,
    compute_path_type,
    compute_range,
    compute_vel,
    delay_to_range,
    doppler_to_velocity,
    monostatic_range_velocity,
    stack_state_field,
)

__all__ = [
    "SensingPerformance",
    "LSChannelEstimator",
    "DelayDopplerSpectrum",
    "DelayDopplerRoi",
    "MUSICEstimator",
    "SensingEstimator",
    "match_peaks_and_compute_radial_rmse",
    "MusicPeaks",
    "SensingEstimate",
    "MetricMode",
    "RoiSlices",
    "SensMode",
    "CFARDetector",
    "MovingTargetIndication",
    "MovingTargetDetection",
    "MONOSTATIC_TX_RX_EPS_M",
    "compute_path_type",
    "compute_range",
    "compute_vel",
    "delay_to_range",
    "doppler_to_velocity",
    "monostatic_range_velocity",
    "stack_state_field",
    "ground_circle_radius_sq",
    "intersect_circles_xy",
    "localize_xy_z0_colocated_tx_mono_bistatic",
    "position_rmse_xy",
    "select_xy_solution",
]
