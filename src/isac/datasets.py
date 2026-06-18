"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

由 ``run_dataset_collection.py`` 写入；``learning/torch_dataset.py`` 等通过 ``Dataset.load`` 消费。
文件布局：必选数据集 ``channel_frequency_response`` + 目标运动学 + OFDM 网格元属性；
可选 CIR（``channel_impulse_response_a/tau``）与 ``collection_*`` 根属性（采集可复现配置）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


_DATASET_KEY_CFR = "channel_frequency_response"
"""CFR，与 OFDM 频率网格一致。"""

_DATASET_KEY_CIR_A = "channel_impulse_response_a"
"""路径增益：最后一维为 ``[Re, Im]``（float64）。"""

_DATASET_KEY_CIR_TAU = "channel_impulse_response_tau"
"""路径时延 τ（秒，float64）。"""

_DATASET_KEY_TARGET_POSITION = "target_position"
"""目标位置 (m)，shape ``(num_slots, 3)``。"""
_DATASET_KEY_TARGET_VELOCITY = "target_velocity"
"""目标速度 (m/s)，shape ``(num_slots, 3)``。"""
# 新文件写入上述键名；Dataset.load 仍兼容旧键 uav_position / uav_velocity。
_DATASET_KEY_BS_POS = "bs_pos"
"""参考发射机/基站位置 (m)，shape ``(3,)``（采集脚本取 ``bs1``）。"""

_META_KEY_CARRIER_FREQUENCY = "carrier_frequency"
"""载频 (Hz)，与 RT / 感知真值一致。"""
_META_KEY_SUBCARRIER_SPACING = "subcarrier_spacing"
"""子载波间隔 (Hz)。"""
_META_KEY_NUM_SUBCARRIERS = "num_subcarriers"
"""OFDM 有效子载波数（与 CFR 频域维对齐）。"""
_META_KEY_NUM_SLOTS = "num_slots"
"""有效 episode / 轨迹步数（与 CFR 第一维一致）。"""
_META_KEY_DESCRIPTION = "description"
_META_KEY_HAS_CIR = "has_cir"
"""是否写入 CIR 数据集（与 cir_a/cir_tau 是否存在一致）。"""

# 采集元数据根属性前缀（``collection_*``），由 ``run_dataset_collection.py`` 写入
_META_PREFIX_COLLECTION = "collection_"


@dataclass(frozen=True)
class CollectionMetadata:
    """``run_dataset_collection.py`` 一次采集运行的可复现配置摘要。

    序列化到 HDF5 根属性 ``collection_<field>``（见 ``write_hdf5_attrs``）。
    蒙特卡洛采集含 ``roi_*``、``quality_*`` 等字段。
    """

    seed: int
    config_file: str
    scene_slug: str
    source: str = "monte_carlo"
    num_samples: int | None = None
    run_sensing: bool = False
    save_cir: bool = False
    roi_xmin: float | None = None
    roi_xmax: float | None = None
    roi_ymin: float | None = None
    roi_ymax: float | None = None
    roi_z: float = 0.0
    sampling_mode: str | None = None
    velocity_sampling: str | None = None
    safe_margin: float | None = None
    max_trials_factor: int | None = None
    speed_min: float | None = None
    speed_max: float | None = None
    quality_filter: bool = False
    quality_accepted: int | None = None
    quality_rejected: int | None = None
    quality_reject_no_valid_paths: int | None = None
    quality_reject_weak_los: int | None = None
    quality_reject_low_peak_prominence: int | None = None
    quality_reject_peak_misaligned: int | None = None
    require_los: bool | None = None
    min_los_ratio: float | None = None
    min_peak_prominence_db: float | None = None
    max_bin_offset: int | None = None

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """写入 HDF5 根属性（标量/字符串；None 字段跳过）。"""
        fields: dict[str, str | int | float | bool] = {
            "source": self.source,
            "seed": self.seed,
            "config_file": self.config_file,
            "scene_slug": self.scene_slug,
            "run_sensing": self.run_sensing,
            "save_cir": self.save_cir,
            "roi_z": self.roi_z,
        }
        optional: dict[str, str | int | float | None] = {
            "num_samples": self.num_samples,
            "roi_xmin": self.roi_xmin,
            "roi_xmax": self.roi_xmax,
            "roi_ymin": self.roi_ymin,
            "roi_ymax": self.roi_ymax,
            "sampling_mode": self.sampling_mode,
            "velocity_sampling": self.velocity_sampling,
            "safe_margin": self.safe_margin,
            "max_trials_factor": self.max_trials_factor,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
            "quality_filter": self.quality_filter,
            "quality_accepted": self.quality_accepted,
            "quality_rejected": self.quality_rejected,
            "quality_reject_no_valid_paths": self.quality_reject_no_valid_paths,
            "quality_reject_weak_los": self.quality_reject_weak_los,
            "quality_reject_low_peak_prominence": self.quality_reject_low_peak_prominence,
            "quality_reject_peak_misaligned": self.quality_reject_peak_misaligned,
            "require_los": self.require_los,
            "min_los_ratio": self.min_los_ratio,
            "min_peak_prominence_db": self.min_peak_prominence_db,
            "max_bin_offset": self.max_bin_offset,
        }
        for key, val in fields.items():
            f.attrs[f"{_META_PREFIX_COLLECTION}{key}"] = val
        for key, val in optional.items():
            if val is not None:
                f.attrs[f"{_META_PREFIX_COLLECTION}{key}"] = val

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata | None:
        """从 HDF5 根属性读取；旧文件无 ``collection_source`` 时返回 ``None``。"""
        src_key = f"{_META_PREFIX_COLLECTION}source"
        if src_key not in f.attrs:
            return None

        def _opt(name: str) -> Any:
            key = f"{_META_PREFIX_COLLECTION}{name}"
            return f.attrs[key] if key in f.attrs else None

        return cls(
            source=str(f.attrs[src_key]),
            seed=int(f.attrs[f"{_META_PREFIX_COLLECTION}seed"]),
            config_file=str(f.attrs[f"{_META_PREFIX_COLLECTION}config_file"]),
            scene_slug=str(f.attrs[f"{_META_PREFIX_COLLECTION}scene_slug"]),
            num_samples=(
                int(_opt("num_samples")) if _opt("num_samples") is not None else None
            ),
            run_sensing=bool(
                f.attrs.get(f"{_META_PREFIX_COLLECTION}run_sensing", False)
            ),
            save_cir=bool(f.attrs.get(f"{_META_PREFIX_COLLECTION}save_cir", False)),
            roi_xmin=float(_opt("roi_xmin")) if _opt("roi_xmin") is not None else None,
            roi_xmax=float(_opt("roi_xmax")) if _opt("roi_xmax") is not None else None,
            roi_ymin=float(_opt("roi_ymin")) if _opt("roi_ymin") is not None else None,
            roi_ymax=float(_opt("roi_ymax")) if _opt("roi_ymax") is not None else None,
            roi_z=float(f.attrs.get(f"{_META_PREFIX_COLLECTION}roi_z", 0.0)),
            sampling_mode=(
                str(_opt("sampling_mode"))
                if _opt("sampling_mode") is not None
                else None
            ),
            velocity_sampling=(
                str(_opt("velocity_sampling"))
                if _opt("velocity_sampling") is not None
                else None
            ),
            safe_margin=(
                float(_opt("safe_margin")) if _opt("safe_margin") is not None else None
            ),
            max_trials_factor=(
                int(_opt("max_trials_factor"))
                if _opt("max_trials_factor") is not None
                else None
            ),
            speed_min=(
                float(_opt("speed_min")) if _opt("speed_min") is not None else None
            ),
            speed_max=(
                float(_opt("speed_max")) if _opt("speed_max") is not None else None
            ),
            time_delta=(
                float(_opt("time_delta")) if _opt("time_delta") is not None else None
            ),
            steps=int(_opt("steps")) if _opt("steps") is not None else None,
            quality_filter=bool(
                f.attrs.get(f"{_META_PREFIX_COLLECTION}quality_filter", False)
            ),
            quality_accepted=(
                int(_opt("quality_accepted"))
                if _opt("quality_accepted") is not None
                else None
            ),
            quality_rejected=(
                int(_opt("quality_rejected"))
                if _opt("quality_rejected") is not None
                else None
            ),
            quality_reject_no_valid_paths=(
                int(_opt("quality_reject_no_valid_paths"))
                if _opt("quality_reject_no_valid_paths") is not None
                else None
            ),
            quality_reject_weak_los=(
                int(_opt("quality_reject_weak_los"))
                if _opt("quality_reject_weak_los") is not None
                else None
            ),
            quality_reject_low_peak_prominence=(
                int(_opt("quality_reject_low_peak_prominence"))
                if _opt("quality_reject_low_peak_prominence") is not None
                else None
            ),
            quality_reject_peak_misaligned=(
                int(_opt("quality_reject_peak_misaligned"))
                if _opt("quality_reject_peak_misaligned") is not None
                else None
            ),
            require_los=(
                bool(_opt("require_los")) if _opt("require_los") is not None else None
            ),
            min_los_ratio=(
                float(_opt("min_los_ratio"))
                if _opt("min_los_ratio") is not None
                else None
            ),
            min_peak_prominence_db=(
                float(_opt("min_peak_prominence_db"))
                if _opt("min_peak_prominence_db") is not None
                else None
            ),
            max_bin_offset=(
                int(_opt("max_bin_offset"))
                if _opt("max_bin_offset") is not None
                else None
            ),
        )

    def format_roi(self) -> str:
        if self.roi_xmin is None:
            return "(not set)"
        return (
            f"x=[{self.roi_xmin:.2f}, {self.roi_xmax:.2f}], "
            f"y=[{self.roi_ymin:.2f}, {self.roi_ymax:.2f}], z={self.roi_z:.2f}"
        )


def _require_cfr_dataset(f: h5py.File) -> h5py.Dataset:
    if _DATASET_KEY_CFR not in f:
        raise KeyError(
            f"HDF5 缺少必选数据集 {_DATASET_KEY_CFR!r}（channel_frequency_response）。"
        )
    return f[_DATASET_KEY_CFR]


def _optional_array_dataset(f: h5py.File, key: str) -> np.ndarray | None:
    if key not in f:
        return None
    return f[key][:]


def _require_dataset_any(f: h5py.File, keys: tuple[str, ...]) -> h5py.Dataset:
    """返回 ``keys`` 中首个存在于文件内的数据集，否则报错。"""
    for k in keys:
        if k in f:
            return f[k]
    raise KeyError(f"HDF5 缺少以下任一数据集: {keys!r}")


def _read_meta(f: h5py.File, key: str) -> Any:
    """读取元数据：优先根 ``attrs``，兼容旧版将标量存为单元素 dataset 的文件。"""
    if key in f.attrs:
        return f.attrs[key]
    if key in f:
        return f[key][()]
    raise KeyError(f"HDF5 中未找到元数据字段: {key}")


def _array_info(name: str, arr: np.ndarray) -> str:
    return f"  {name}: shape={arr.shape}, dtype={arr.dtype}"


@dataclass
class Dataset:
    """ISAC HDF5 数据集的内存表示（必含 CFR；CIR 可选）。

    典型数组形状：

    - ``cfr``：``(num_slots, S, F)`` 或 ``(num_slots, F)``，复数；``S``=OFDM 符号，``F``=子载波
    - ``cir_a`` / ``cir_tau``：路径数可变时在路径维零填充；``cir_a`` 末维为 ``[Re, Im]``
    - ``target_position`` / ``target_velocity``：``(num_slots, 3)``
    - ``bs_pos``：``(3,)``

    常用 API：

    - 构造：``Dataset.from_export_arrays(...)``
    - 落盘 / 读取：``save`` / ``load``
    - 校验：``show()``；CFR 预览：``plot_cfr(slot=...)``
    """

    cfr: np.ndarray
    cir_a: np.ndarray | None
    cir_tau: np.ndarray | None
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
        dataset_cir_a: np.ndarray | None = None,
        dataset_cir_tau: np.ndarray | None = None,
        description: str | None = None,
        collection_meta: CollectionMetadata | None = None,
    ) -> Dataset:
        """由 ndarray 组装数据集（与采集脚本导出字段对齐）。

        ``dataset_cir_a`` / ``dataset_cir_tau`` 可省略（仅写 CFR + kinematics）。
        ``scene_name`` 仅用于在未显式给出 ``description`` 时生成默认英文描述；不单独写入 HDF5。
        ``num_valid`` 映射为属性 ``num_slots``（有效 episode / 步数）。
        """
        if (dataset_cir_a is None) != (dataset_cir_tau is None):
            raise ValueError("dataset_cir_a 与 dataset_cir_tau 须同时提供或同时为 None")
        desc = description or (
            f"Sionna generated ISAC Monte Carlo dataset in {scene_name}"
        )
        return cls(
            cfr=dataset_cfr,
            cir_a=dataset_cir_a,
            cir_tau=dataset_cir_tau,
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
        """从 HDF5 加载（必含 CFR；CIR 可选）。

        目标位置/速度优先读取 ``target_position`` / ``target_velocity``，否则回退 ``uav_*`` 旧键。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            cir_a = _optional_array_dataset(f, _DATASET_KEY_CIR_A)
            cir_tau = _optional_array_dataset(f, _DATASET_KEY_CIR_TAU)
            if (cir_a is None) != (cir_tau is None):
                raise ValueError(
                    "HDF5 中 CIR 数据集不完整："
                    f"{_DATASET_KEY_CIR_A!r} 与 {_DATASET_KEY_CIR_TAU!r} 须同时存在或同时缺失"
                )
            return cls(
                cfr=_require_cfr_dataset(f)[:],
                cir_a=cir_a,
                cir_tau=cir_tau,
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
        """写入 HDF5（gzip 压缩 CFR/CIR）；根属性含 OFDM 网格与 ``has_cir``。"""
        path = Path(filepath)
        has_cir = self.cir_a is not None and self.cir_tau is not None
        with h5py.File(path, "w") as f:
            f.create_dataset(_DATASET_KEY_CFR, data=self.cfr, compression="gzip")
            if has_cir:
                f.create_dataset(
                    _DATASET_KEY_CIR_A, data=self.cir_a, compression="gzip"
                )
                f.create_dataset(
                    _DATASET_KEY_CIR_TAU, data=self.cir_tau, compression="gzip"
                )
            f.create_dataset(_DATASET_KEY_TARGET_POSITION, data=self.target_position)
            f.create_dataset(_DATASET_KEY_TARGET_VELOCITY, data=self.target_velocity)
            f.create_dataset(_DATASET_KEY_BS_POS, data=self.bs_pos)

            f.attrs[_META_KEY_CARRIER_FREQUENCY] = self.carrier_frequency
            f.attrs[_META_KEY_SUBCARRIER_SPACING] = self.subcarrier_spacing
            f.attrs[_META_KEY_NUM_SUBCARRIERS] = self.num_subcarriers
            f.attrs[_META_KEY_NUM_SLOTS] = self.num_slots
            f.attrs[_META_KEY_DESCRIPTION] = self.description
            f.attrs[_META_KEY_HAS_CIR] = has_cir
            if self.collection_meta is not None:
                self.collection_meta.write_hdf5_attrs(f)

        print(f"所有数据已成功保存至: {path}")

    def show(self, *, description_max_len: int = 300) -> None:
        """在终端打印元数据与各数组 shape/dtype（用于快速校验）。"""
        desc = self.description
        if len(desc) > description_max_len:
            desc = desc[:description_max_len] + "..."
        print(self)
        print("description:", desc or "(empty)")
        print(
            f"  carrier_frequency={self.carrier_frequency} Hz, "
            f"subcarrier_spacing={self.subcarrier_spacing} Hz, "
            f"num_subcarriers={self.num_subcarriers}"
        )
        print(_array_info("cfr", self.cfr))
        if self.cir_a is not None and self.cir_tau is not None:
            print(_array_info("cir_a", self.cir_a))
            print(_array_info("cir_tau", self.cir_tau))
        else:
            print("  cir: (not stored)")
        print(_array_info("target_position", self.target_position))
        print(_array_info("target_velocity", self.target_velocity))
        print(_array_info("bs_pos", self.bs_pos))
        if self.collection_meta is not None:
            m = self.collection_meta
            print(
                f"  collection: source={m.source}, seed={m.seed}, "
                f"config={m.config_file}, scene={m.scene_slug}"
            )
            print(f"  collection ROI: {m.format_roi()}")
            if m.num_samples is not None:
                print(
                    f"  collection num_samples={m.num_samples}, sampling_mode={m.sampling_mode}"
                )
            if m.time_delta is not None:
                print(f"  collection time_delta={m.time_delta}, steps={m.steps}")

    def plot_cfr(self, slot: int = 0, *, magnitude: bool = True) -> None:
        """绘制某一 slot 的 CFR；默认显示幅度二维图（需 matplotlib、适宜交互后端）。

        ``cfr[slot]`` 若为复数则取 ``abs``；若为二维 ``(H, W)`` 则 ``imshow``，若为一维则 ``plot``。
        """
        import matplotlib.pyplot as plt

        c = np.asarray(self.cfr)
        if c.ndim < 1:
            raise ValueError("cfr 数组维度不足，无法按 slot 索引")
        n = c.shape[0]
        if slot < 0 or slot >= n:
            raise IndexError(f"slot {slot} 越界，有效范围 [0, {n - 1}]")
        plane = np.asarray(c[slot])
        if magnitude and np.iscomplexobj(plane):
            plane = np.abs(plane)

        fig, ax = plt.subplots(figsize=(8, 5))
        if plane.ndim == 1:
            ax.plot(plane)
            ax.set_title(f"CFR slot={slot} (1D)")
            ax.set_xlabel("subcarrier index")
        elif plane.ndim == 2:
            im = ax.imshow(plane, aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"CFR slot={slot} ({'|·|' if magnitude else 'real'} 2D)")
        else:
            raise ValueError(
                f"cfr[slot] 维度 {plane.ndim} 不支持快速绘图，请先 reshape"
            )

        plt.tight_layout()
        plt.show()
