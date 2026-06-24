"""深度学习感知模块：时延–多普勒谱 → 距离/速度估计。"""

from .dd_spectrum import (
    crop_dd_roi,
    dd_spectrum_to_features,
    monostatic_labels_from_kinematics,
    squeeze_cfr_to_sf,
)
from .monostatic_cnn import MonostaticDelayDopplerCNN
from .torch_dataset import MonostaticSensingTorchDataset

__all__ = [
    "MonostaticDelayDopplerCNN",
    "MonostaticSensingTorchDataset",
    "crop_dd_roi",
    "dd_spectrum_to_features",
    "monostatic_labels_from_kinematics",
    "squeeze_cfr_to_sf",
]
