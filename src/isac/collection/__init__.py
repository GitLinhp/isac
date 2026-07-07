"""蒙特卡洛采集流水线：HDF5 数据集、ROI 采样与 episode 过滤。"""

from .dataset import (
    CollectionMetadata,
    RTDataset,
    RTDatasetWriter,
    save_collection_artifacts,
)
from .h5_layout import (
    collection_dataset_description,
    collection_h5_path,
    collection_scene_png_path,
)
from .roi_sampling import (
    RoiKinematicsSampler,
    SamplingMode,
    parse_roi_xy,
    parse_speed_range,
)
from .utils import scene_slug_from_rt_simulator

__all__ = [
    "CollectionMetadata",
    "RTDataset",
    "RTDatasetWriter",
    "collection_dataset_description",
    "collection_h5_path",
    "collection_scene_png_path",
    "save_collection_artifacts",
    "RoiKinematicsSampler",
    "SamplingMode",
    "parse_roi_xy",
    "parse_speed_range",
    "scene_slug_from_rt_simulator",
]
