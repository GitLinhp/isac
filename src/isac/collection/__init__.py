"""蒙特卡洛采集流水线：HDF5 数据集、ROI 采样与 episode 过滤。"""

from .dataset import (
    CollectionMetadata,
    EpisodeBuffers,
    RTDataset,
    SensingMetadata,
    collection_dataset_description,
    collection_h5_path,
    collection_scene_png_path,
    save_collection_artifacts,
)
from .roi_sampling import RoiKinematicsSampler, SamplingMode, parse_roi_xy, parse_speed_range

__all__ = [
    "CollectionMetadata",
    "EpisodeBuffers",
    "RTDataset",
    "SensingMetadata",
    "collection_dataset_description",
    "collection_h5_path",
    "collection_scene_png_path",
    "save_collection_artifacts",
    "RoiKinematicsSampler",
    "SamplingMode",
    "parse_roi_xy",
    "parse_speed_range",
    "accept_episode_kinematics",
    "los_truth_from_kinematics",
    "scene_slug_from_rt_simulator",
]


def __getattr__(name: str):
    if name == "accept_episode_kinematics":
        from .episode_filter import accept_episode_kinematics

        return accept_episode_kinematics
    if name in ("los_truth_from_kinematics", "scene_slug_from_rt_simulator"):
        from . import channel_export

        return getattr(channel_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
