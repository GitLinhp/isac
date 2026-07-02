"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

由 ``run_data_collection.py`` 写入；训练侧通过 ``Dataset.load`` 消费。

HDF5 文件布局
--------------
根 datasets:

- ``channel_frequency_response``：复数 CFR，典型 shape ``(N, ..., S, F)``
  （``N``=episode 数，``S``=OFDM 符号，``F``=子载波）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）

根 attrs:

- ``carrier_frequency``, ``subcarrier_spacing``, ``num_subcarriers``,
  ``num_slots``, ``description``
- ``collection_*``：由 ``CollectionMetadata`` 写入，共 8 个采集可复现字段

Episode CSV（独立文件，非 HDF5 内）:

- ``{scene_slug}_mc_dataset_episodes.csv``

``Dataset.load`` 对目标运动学兼容旧键名 ``uav_position`` / ``uav_velocity``。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from isac import PROJECT_ROOT

DEFAULT_COLLECTION_OUT_DIR = PROJECT_ROOT / "out" / "dataset_collection"
"""``run_data_collection.py`` 默认输出目录。"""

_DATASET_KEY_CFR = "channel_frequency_response"
"""CFR，与 OFDM 频率网格一致。"""

_DATASET_KEY_TARGET_POSITION = "target_position"
"""目标位置 (m)，shape ``(num_slots, 3)``。"""

_DATASET_KEY_TARGET_VELOCITY = "target_velocity"
"""目标速度 (m/s)，shape ``(num_slots, 3)``。"""

# 新文件写入上述键名；Dataset.load 仍兼容旧键 uav_position / uav_velocity。
_DATASET_KEY_BS_POS = "bs_pos"
"""参考发射机/基站位置 (m)，shape ``(3,)``。"""

_META_KEY_CARRIER_FREQUENCY = "carrier_frequency"
"""载频 (Hz)，与 RT / 感知真值一致。"""

_META_KEY_SUBCARRIER_SPACING = "subcarrier_spacing"
"""子载波间隔 (Hz)。"""

_META_KEY_NUM_SUBCARRIERS = "num_subcarriers"
"""OFDM 有效子载波数（与 CFR 频域维对齐）。"""

_META_KEY_NUM_SLOTS = "num_slots"
"""有效 episode 数（与 CFR 第一维一致）。"""

_META_KEY_DESCRIPTION = "description"
"""数据集英文描述字符串。"""

# 采集元数据根属性前缀（``collection_*``），由 run_data_collection 写入。
_META_PREFIX_COLLECTION = "collection_"


@dataclass
class EpisodeBuffers:
    """主循环共享的 episode 级写出缓冲。

    采集循环中由 ``process_episode`` 逐条追加，循环结束后经
    ``save_episodes_csv`` / ``save_episode_buffers_h5`` 落盘。

    Attributes
    ----------
    h_freq_list : list[np.ndarray]
        逐 episode 的 CFR numpy 数组。
    target_pos_list : list[np.ndarray]
        逐 episode 目标位置，每条 shape ``(3,)`` (m)。
    target_vel_list : list[np.ndarray]
        逐 episode 目标速度，每条 shape ``(3,)`` (m/s)。
    csv_rows : list[dict[str, str | int]]
        逐 episode CSV 行（运动学 + 几何真值列）。
    """

    h_freq_list: list[np.ndarray] = field(default_factory=list)
    target_pos_list: list[np.ndarray] = field(default_factory=list)
    target_vel_list: list[np.ndarray] = field(default_factory=list)
    csv_rows: list[dict[str, str | int]] = field(default_factory=list)


@dataclass(frozen=True)
class CollectionMetadata:
    """一次采集运行的可复现配置摘要，序列化到 HDF5 根属性 ``collection_<field>``。

    共 8 个字段，对应 ``run_data_collection.py`` 蒙特卡洛平面 ROI 采集 CLI。

    Attributes
    ----------
    - seed : int
        随机种子（``--seed``）。
    - config_file : str
        仿真配置路径（``--config_file``）。
    - scene_slug : str
        RT 场景文件名片段，用于输出命名。
    - num_samples : int
        有效 episode 数（``--num_samples``）。
    - roi : tuple[float, float, float, float]
        平面 ROI 边界 ``(xmin, xmax, ymin, ymax)`` (m)，来自 ``--roi``。
    - position_sampling_mode : str
        位置采样分布（``--position_sampling_mode``）。
    - speed_range : tuple[float, float]
        速度模值范围 ``(min, max)`` (m/s)，来自 ``--speed_range``。
    - speed_sampling_mode : str
        速度模值采样分布（``--speed_sampling_mode``）。
    """

    seed: int
    config_file: str
    scene_slug: str
    num_samples: int
    roi: tuple[float, float, float, float]
    position_sampling_mode: str = "uniform"
    speed_range: tuple[float, float] = (0.0, 0.0)
    speed_sampling_mode: str = "uniform"

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """将全部字段写入 HDF5 根属性 ``collection_<field>``。"""
        for key, val in asdict(self).items():
            if isinstance(val, tuple):
                val = list(val)
            f.attrs[f"{_META_PREFIX_COLLECTION}{key}"] = val

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata | None:
        """从 HDF5 根属性读取采集元数据。

        无 ``collection_seed`` 时返回 ``None``（旧版文件）。
        兼容旧版分散字段（``roi_xmin`` 等、``speed_min`` 等、
        ``sampling_mode`` / ``velocity_sampling``）。
        文件中多余的 ``collection_*`` 属性静默忽略。
        """
        prefix = _META_PREFIX_COLLECTION
        if f"{prefix}seed" not in f.attrs:
            return None

        def _opt(name: str) -> Any:
            key = f"{prefix}{name}"
            return f.attrs[key] if key in f.attrs else None

        roi_raw = _opt("roi")
        if roi_raw is not None:
            roi = tuple(float(x) for x in roi_raw)
        elif _opt("roi_xmin") is not None:
            roi = (
                float(_opt("roi_xmin")),
                float(_opt("roi_xmax")),
                float(_opt("roi_ymin")),
                float(_opt("roi_ymax")),
            )
        else:
            roi = (0.0, 0.0, 0.0, 0.0)

        speed_range_raw = _opt("speed_range")
        if speed_range_raw is not None:
            speed_range = tuple(float(x) for x in speed_range_raw)
        elif _opt("speed_min") is not None:
            speed_range = (float(_opt("speed_min")), float(_opt("speed_max")))
        else:
            speed_range = (0.0, 0.0)

        position_sampling_mode = str(
            _opt("position_sampling_mode") or _opt("sampling_mode") or "uniform"
        )
        speed_sampling_mode = str(
            _opt("speed_sampling_mode") or _opt("velocity_sampling") or "uniform"
        )

        kwargs: dict[str, Any] = {
            "seed": int(_opt("seed")),
            "config_file": str(_opt("config_file")),
            "scene_slug": str(_opt("scene_slug")),
            "num_samples": int(_opt("num_samples")),
            "roi": roi,
            "position_sampling_mode": position_sampling_mode,
            "speed_range": speed_range,
            "speed_sampling_mode": speed_sampling_mode,
        }
        return cls(**kwargs)

    def format_roi(self) -> str:
        """返回人类可读的 ROI 范围字符串。"""
        x_lo, x_hi, y_lo, y_hi = self.roi
        return f"x=[{x_lo:.2f}, {x_hi:.2f}], y=[{y_lo:.2f}, {y_hi:.2f}]"

    @classmethod
    def from_collection_args(
        cls, args: argparse.Namespace, scene_slug: str
    ) -> CollectionMetadata:
        """从 ``run_data_collection.py`` CLI 参数构建元数据。

        依赖字段：``--seed``, ``--config_file``, ``--num_samples``,
        ``--roi``, ``--position_sampling_mode``, ``--speed_sampling_mode``,
        ``--speed_range``。
        """
        r = args.roi
        return cls(
            seed=int(args.seed),
            config_file=str(args.config_file),
            scene_slug=scene_slug,
            num_samples=int(args.num_samples),
            roi=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
            position_sampling_mode=str(args.position_sampling_mode),
            speed_range=(float(args.speed_range[0]), float(args.speed_range[1])),
            speed_sampling_mode=str(args.speed_sampling_mode),
        )


def save_episodes_csv(
    *,
    scene_slug: str,
    rows: list[dict[str, str | int]],
    output_root: Path | None = None,
) -> None:
    """写入 Episode CSV（动态列并集）。

    输出路径：``{output_root}/{scene_slug}_mc_dataset_episodes.csv``。
    ``output_root`` 默认为 ``PROJECT_ROOT / "out"``。
    各行字典的键取并集作为 CSV 列，缺失列填空字符串。
    """
    if not rows:
        print("无 CSV 行，跳过写入")
        return
    out_dir = output_root if output_root is not None else PROJECT_ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{scene_slug}_mc_dataset_episodes.csv"
    keys_set: set[str] = set()
    for r in rows:
        keys_set.update(r.keys())
    keys = sorted(keys_set)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=keys, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Episode CSV 已写入: {path}")


def _resolve_h5_path(
    scene_slug: str,
    n_episodes: int,
    out_dir: Path,
) -> tuple[Path, str, str]:
    """解析 HDF5 输出路径、描述文本与场景名。

    返回
    ----
    - h5_path : ``{out_dir}/{scene_slug}_mc_sionna_dataset.h5``
    - description : 写入根属性 ``description`` 的英文描述
    - scene_name : 内部场景标识 ``{scene_slug}_mc``
    """
    mc_slug = f"{scene_slug}_mc"
    return (
        out_dir / f"{scene_slug}_mc_sionna_dataset.h5",
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) in {scene_slug}",
        mc_slug,
    )


def save_episode_buffers_h5(
    buffers: EpisodeBuffers,
    *,
    scene_slug: str,
    n_episodes: int,
    bs_pos: np.ndarray,
    carrier_frequency: float,
    subcarrier_spacing: float,
    num_subcarriers: int,
    collection_meta: CollectionMetadata,
    out_dir: Path,
) -> None:
    """将 episode 缓冲的 CFR/kinematics 封装为 ``Dataset`` 并落盘 HDF5。

    ``buffers.h_freq_list`` 为空时打印提示并跳过。
    OFDM 网格参数（载频、子载波间隔、子载波数）写入 HDF5 根属性。
    """
    if not buffers.h_freq_list:
        print("未采集 CFR，跳过 HDF5")
        return

    h5_path, desc_h5, scene_name = _resolve_h5_path(scene_slug, n_episodes, out_dir)
    Dataset.from_export_arrays(
        np.array(buffers.h_freq_list),
        np.array(buffers.target_pos_list),
        np.array(buffers.target_vel_list),
        bs_pos,
        carrier_frequency,
        subcarrier_spacing,
        num_subcarriers,
        len(buffers.h_freq_list),
        scene_name,
        description=desc_h5,
        collection_meta=collection_meta,
    ).save(h5_path)


def _require_cfr_dataset(f: h5py.File) -> h5py.Dataset:
    """返回必选 CFR 数据集；缺失时抛出 ``KeyError``。"""
    if _DATASET_KEY_CFR not in f:
        raise KeyError(
            f"HDF5 缺少必选数据集 {_DATASET_KEY_CFR!r}（channel_frequency_response）。"
        )
    return f[_DATASET_KEY_CFR]


def _require_dataset_any(f: h5py.File, keys: tuple[str, ...]) -> h5py.Dataset:
    """返回 ``keys`` 中首个存在于文件内的数据集。"""
    for k in keys:
        if k in f:
            return f[k]
    raise KeyError(f"HDF5 缺少以下任一数据集: {keys!r}")


def _read_meta(f: h5py.File, key: str) -> Any:
    """读取元数据：优先根 ``attrs``，兼容旧版单元素 dataset 存储。"""
    if key in f.attrs:
        return f.attrs[key]
    if key in f:
        return f[key][()]
    raise KeyError(f"HDF5 中未找到元数据字段: {key}")


@dataclass
class Dataset:
    """ISAC HDF5 数据集的内存表示（CFR + kinematics）。

    典型数组形状：

    - ``cfr``：``(num_slots, ..., num_ofdm_symbols, num_subcarriers)``，复数
    - ``target_position`` / ``target_velocity``：``(num_slots, 3)``
    - ``bs_pos``：``(3,)``

    ``num_slots`` 与 ``cfr`` 第一维（episode 数）一致。

    Attributes
    ----------
    cfr : np.ndarray
        信道频率响应。
    target_position : np.ndarray
        目标位置 (m)。
    target_velocity : np.ndarray
        目标速度 (m/s)。
    bs_pos : np.ndarray
        参考发射机位置 (m)。
    carrier_frequency : float
        载频 (Hz)。
    subcarrier_spacing : float
        子载波间隔 (Hz)。
    num_subcarriers : int
        子载波数（与 CFR 频域维对齐）。
    num_slots : int
        有效 episode 数。
    description : str
        数据集描述。
    collection_meta : CollectionMetadata | None
        采集可复现配置；旧文件可能为 ``None``。
    """

    cfr: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    bs_pos: np.ndarray
    carrier_frequency: float
    subcarrier_spacing: float
    num_subcarriers: int
    num_slots: int
    description: str = ""
    collection_meta: CollectionMetadata | None = None

    def __repr__(self) -> str:
        return (
            f"Dataset(num_slots={self.num_slots}, cfr_shape={self.cfr.shape}, "
            f"fc={self.carrier_frequency:.3e} Hz)"
        )

    @classmethod
    def from_export_arrays(
        cls,
        dataset_cfr: np.ndarray,
        dataset_pos: np.ndarray,
        dataset_vel: np.ndarray,
        bs_pos: np.ndarray,
        carrier_frequency: float,
        subcarrier_spacing: float,
        num_subcarriers: int,
        num_valid: int,
        scene_name: str,
        *,
        description: str | None = None,
        collection_meta: CollectionMetadata | None = None,
    ) -> Dataset:
        """由 ndarray 组装 ``Dataset``（与采集脚本导出字段对齐）。

        ``num_valid`` 映射为 ``num_slots``（有效 episode 数）。
        ``scene_name`` 仅用于生成默认 ``description``，不单独写入 HDF5。
        """
        desc = description or (
            f"Sionna generated ISAC dataset with car trajectory in {scene_name}"
        )
        return cls(
            cfr=dataset_cfr,
            target_position=dataset_pos,
            target_velocity=dataset_vel,
            bs_pos=bs_pos,
            carrier_frequency=carrier_frequency,
            subcarrier_spacing=subcarrier_spacing,
            num_subcarriers=num_subcarriers,
            num_slots=num_valid,
            description=desc,
            collection_meta=collection_meta,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> Dataset:
        """从 HDF5 加载数据集。

        目标位置/速度优先读取 ``target_position`` / ``target_velocity``，
        否则回退旧键 ``uav_position`` / ``uav_velocity``。
        采集元数据经 ``CollectionMetadata.read_hdf5_attrs`` 读取。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            return cls(
                cfr=_require_cfr_dataset(f)[:],
                target_position=_require_dataset_any(
                    f, (_DATASET_KEY_TARGET_POSITION, "uav_position")
                )[:],
                target_velocity=_require_dataset_any(
                    f, (_DATASET_KEY_TARGET_VELOCITY, "uav_velocity")
                )[:],
                bs_pos=f[_DATASET_KEY_BS_POS][:],
                carrier_frequency=float(_read_meta(f, _META_KEY_CARRIER_FREQUENCY)),
                subcarrier_spacing=float(_read_meta(f, _META_KEY_SUBCARRIER_SPACING)),
                num_subcarriers=int(_read_meta(f, _META_KEY_NUM_SUBCARRIERS)),
                num_slots=int(_read_meta(f, _META_KEY_NUM_SLOTS)),
                description=(
                    str(_read_meta(f, _META_KEY_DESCRIPTION))
                    if (_META_KEY_DESCRIPTION in f.attrs or _META_KEY_DESCRIPTION in f)
                    else ""
                ),
                collection_meta=CollectionMetadata.read_hdf5_attrs(f),
            )

    def save(self, filepath: str | Path) -> None:
        """写入 HDF5：CFR 使用 gzip 压缩；根属性含 OFDM 网格与 ``collection_*``。"""
        path = Path(filepath)
        with h5py.File(path, "w") as f:
            f.create_dataset(_DATASET_KEY_CFR, data=self.cfr, compression="gzip")
            f.create_dataset(_DATASET_KEY_TARGET_POSITION, data=self.target_position)
            f.create_dataset(_DATASET_KEY_TARGET_VELOCITY, data=self.target_velocity)
            f.create_dataset(_DATASET_KEY_BS_POS, data=self.bs_pos)

            f.attrs[_META_KEY_CARRIER_FREQUENCY] = self.carrier_frequency
            f.attrs[_META_KEY_SUBCARRIER_SPACING] = self.subcarrier_spacing
            f.attrs[_META_KEY_NUM_SUBCARRIERS] = self.num_subcarriers
            f.attrs[_META_KEY_NUM_SLOTS] = self.num_slots
            f.attrs[_META_KEY_DESCRIPTION] = self.description
            if self.collection_meta is not None:
                self.collection_meta.write_hdf5_attrs(f)

        print(f"所有数据已成功保存至: {path}")
