"""深度学习感知模型。"""

from .preprocess import (
    dd_spectrum_to_features,
    divide_cpi_dual_to_roi_range_profiles_np,
    divide_cpi_to_roi_range_profile_np,
    dual_range_profile_to_features,
    dual_range_profiles_to_features,
    kinematics_to_range_velocity,
    kinematics_to_target_bins,
    normalize_spectrum_batch,
    range_profile_to_features,
    spectrum_tensor_to_features,
)
from .loss import (
    MonostaticSensingLoss,
    MonostaticSensingLossConfig,
    TargetPositionRmseLoss,
)
from .model_design import (
    Conv1dResidualBlock,
    ConvResidualBlock,
    CooperativeMonostaticCNN,
    SensingCNN,
    load_cooperative_monostatic_cnn_checkpoint,
    load_sensing_cnn_checkpoint,
)

__all__ = [
    "Conv1dResidualBlock",
    "ConvResidualBlock",
    "CooperativeMonostaticCNN",
    "SensingCNN",
    "load_cooperative_monostatic_cnn_checkpoint",
    "load_sensing_cnn_checkpoint",
    "MonostaticSensingLoss",
    "MonostaticSensingLossConfig",
    "TargetPositionRmseLoss",
    "dd_spectrum_to_features",
    "divide_cpi_dual_to_roi_range_profiles_np",
    "divide_cpi_to_roi_range_profile_np",
    "dual_range_profile_to_features",
    "dual_range_profiles_to_features",
    "kinematics_to_range_velocity",
    "kinematics_to_target_bins",
    "normalize_spectrum_batch",
    "range_profile_to_features",
    "spectrum_tensor_to_features",
]
