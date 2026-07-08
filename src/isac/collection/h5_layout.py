"""HDF5 采集布局：路径后缀、数据集键名与元数据前缀。"""

from __future__ import annotations

from pathlib import Path

EPISODE_CSV_SUFFIX = "_mc_dataset_episodes.csv"
H5_SUFFIX = "_mc_sionna_dataset.h5"
SCENE_PNG_SUFFIX = "_scene.png"

EPISODE_CSV_COLUMNS = (
    "sample_idx",
    "position",
    "velocity",
    "true_range_m",
    "true_radial_velocity_mps",
)

DATASET_KEY_H_DD = "delay_doppler_spectrum"
DATASET_KEY_TARGET_POSITION = "target_position"
DATASET_KEY_TARGET_VELOCITY = "target_velocity"
DATASET_KEY_BS_POS = "bs_pos"

META_KEY_DESCRIPTION = "description"

COLLECTION_TUPLE_FIELDS = frozenset({"roi", "speed_range"})

ARRAY_DATASET_SPECS: tuple[tuple[str, str], ...] = (
    (DATASET_KEY_BS_POS, "bs_pos"),
    (DATASET_KEY_TARGET_POSITION, "target_position"),
    (DATASET_KEY_TARGET_VELOCITY, "target_velocity"),
    (DATASET_KEY_H_DD, "h_dd"),
)


def collection_h5_path(scene_slug: str, out_dir: Path) -> Path:
    """HDF5 输出路径 ``{out_dir}/{scene_slug}_mc_sionna_dataset.h5``。"""
    return out_dir / f"{scene_slug}{H5_SUFFIX}"


def collection_scene_png_path(scene_slug: str, out_dir: Path) -> Path:
    """场景渲染图路径 ``{out_dir}/{scene_slug}_scene.png``。"""
    return out_dir / f"{scene_slug}{SCENE_PNG_SUFFIX}"


def collection_dataset_description(scene_slug: str, n_episodes: int) -> str:
    """生成写入根属性 ``description`` 的英文描述。"""
    return (
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) "
        f"in {scene_slug}"
    )
