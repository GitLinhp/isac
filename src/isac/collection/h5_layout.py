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
META_KEY_APPLY_MTI = "apply_mti"

COLLECTION_TUPLE_FIELDS = frozenset({"roi", "speed_range"})

ARRAY_DATASET_SPECS: tuple[tuple[str, str], ...] = (
    (DATASET_KEY_BS_POS, "bs_pos"),
    (DATASET_KEY_TARGET_POSITION, "target_position"),
    (DATASET_KEY_TARGET_VELOCITY, "target_velocity"),
    (DATASET_KEY_H_DD, "h_dd"),
)


def format_subcarrier_spacing_slug(subcarrier_spacing_hz: float) -> str:
    """将子载波间隔 (Hz) 格式化为目录名片段，例如 30000 -> ``30kHz``。"""
    khz = subcarrier_spacing_hz / 1e3
    if abs(khz - round(khz)) < 1e-6:
        return f"{int(round(khz))}kHz"
    text = f"{khz:.2f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}kHz"


def collection_dataset_dir(
    scene_slug: str,
    subcarrier_spacing_hz: float,
    base_dir: Path,
) -> Path:
    """返回采集产物子目录 ``{base_dir}/{scene_slug}_{scs_slug}``。"""
    scs_slug = format_subcarrier_spacing_slug(subcarrier_spacing_hz)
    return base_dir / f"{scene_slug}_{scs_slug}"


def collection_h5_path(scene_slug: str, out_dir: Path) -> Path:
    """HDF5 输出路径 ``{out_dir}/{scene_slug}_mc_sionna_dataset.h5``。"""
    return out_dir / f"{scene_slug}{H5_SUFFIX}"


def collection_dataset_description(scene_slug: str, n_episodes: int) -> str:
    """生成写入根属性 ``description`` 的英文描述。"""
    return (
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) "
        f"in {scene_slug}"
    )
