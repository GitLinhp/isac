"""感知 DSP 子包：DD 谱、CFAR、MUSIC、杂波抑制与定位。

典型导入::

    from isac.sensing import DelayDopplerSpectrum, MUSICEstimator, MusicSensingEvaluator, CFARDetector
    from isac.sensing.geometry import delay_to_range, doppler_to_velocity
"""

from .spectrum import (
    DelayDopplerSpectrum,
    LSChannelEstimator,
    SensingPerformance,
)
from .detection import CFARDetector
from .detection.music_estimator import MUSICEstimator
from .detection.music_sensing import (
    MusicEvaluationResult,
    MusicSensingEvaluator,
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
    stack_state_field,
)

__all__ = [
    "SensingPerformance",
    "LSChannelEstimator",
    "DelayDopplerSpectrum",
    "MUSICEstimator",
    "MusicSensingEvaluator",
    "MusicEvaluationResult",
    "match_peaks_and_compute_radial_rmse",
    "CFARDetector",
    "MovingTargetIndication",
    "MovingTargetDetection",
    "MONOSTATIC_TX_RX_EPS_M",
    "compute_path_type",
    "compute_range",
    "compute_vel",
    "delay_to_range",
    "doppler_to_velocity",
    "stack_state_field",
    "ground_circle_radius_sq",
    "intersect_circles_xy",
    "localize_xy_z0_colocated_tx_mono_bistatic",
    "position_rmse_xy",
    "select_xy_solution",
]
