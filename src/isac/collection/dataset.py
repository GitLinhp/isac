"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

采集期：``RTDatasetWriter.open`` 按 episode 流式写入 HDF5。
消费侧：``RTDataset.load`` 加载内存；``__getitem__`` 返回 CNN 训练 dict。
采集期完成 CFR→h_dd 感知链；HDF5 仅存 ROI 裁剪后的复数 DD 谱与运动学。
``RTDataset`` 继承 ``torch.utils.data.Dataset``（只读）；写入由 ``RTDatasetWriter`` 负责。

HDF5 文件布局
--------------
根 datasets:

- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``delay_doppler_spectrum``：复数 h_dd，shape ``(N, H, W)``（ROI 裁剪后）

根 attrs:

- ``description``
- ``seed``, ``roi``, ``position_sampling_mode``, ``speed_range``, ``speed_sampling_mode``、
  ``num_samples``, ``sampler_pool_factor``：由 ``CollectionMetadata`` 写入

感知 ROI 与分辨率见同目录 TOML 副本（``[dd_spectrum_roi]``、``[ofdm]``），不写入 HDF5。

采集落盘产物（``data/``）:

- TOML：采集配置副本（保留原文件名）
- ``{scene_slug}_mc_dataset_episodes.csv``
- ``{scene_slug}_mc_sionna_dataset.h5``
- ``{scene_slug}_scene.png``
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from isac.models.preprocess import dd_spectrum_to_features
from isac.sensing.geometry import monostatic_range_velocity
from isac.sensing.metric import SpectrumMetric
from isac.data_structures.types import MusicPeaks
from isac.utils import set_random_seed
from isac.utils.config_loader import resolve_config_path

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator

from .h5_layout import (
    ARRAY_DATASET_SPECS,
    COLLECTION_TUPLE_FIELDS,
    DATASET_KEY_BS_POS,
    DATASET_KEY_H_DD,
    DATASET_KEY_TARGET_POSITION,
    DATASET_KEY_TARGET_VELOCITY,
    EPISODE_CSV_COLUMNS,
    EPISODE_CSV_SUFFIX,
    META_KEY_DESCRIPTION,
    SCENE_PNG_SUFFIX,
    collection_dataset_description,
)
from .roi_sampling import RoiKinematicsSampler, SamplingMode

_VALID_SAMPLING_MODES = frozenset({"uniform", "gaussian"})


def _parse_sampling_mode(raw: Any, *, field: str) -> SamplingMode:
    mode = str(raw).strip().lower()
    if mode not in _VALID_SAMPLING_MODES:
        raise ValueError(
            f"{field} 仅支持 'uniform' 或 'gaussian'，收到 {raw!r}"
        )
    return mode  # type: ignore[return-value]


# --- HDF5 读写辅助 ---
def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """返回指定名称的 HDF5 dataset；缺失时 ``KeyError``。"""
    if key not in f:
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


def _read_array_datasets(f: h5py.File) -> dict[str, np.ndarray]:
    """读取 h_dd、运动学与 ``bs_pos`` ndarray。"""
    return {
        attr: _require_dataset(f, h5_key)[:] for h5_key, attr in ARRAY_DATASET_SPECS
    }


def _write_hdf5_root_metadata(
    f: h5py.File,
    *,
    n_episodes: int,
    collection_meta: CollectionMetadata,
    scene_slug: str,
) -> None:
    """写入 ``description`` 与采集元数据根属性。"""
    f.attrs[META_KEY_DESCRIPTION] = collection_dataset_description(
        scene_slug, n_episodes
    )
    collection_meta.write_hdf5_attrs(f)


def _create_vec3_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """创建可扩展的 ``(N, 3)`` float64 运动学数据集。"""
    return f.create_dataset(
        key,
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float64,
        chunks=(1024, 3),
    )


# --- 数据类 ---
def _hdf5_serialize(val: Any) -> Any:
    """将 tuple 转为 list，供 h5py 根属性写入。"""
    return list(val) if isinstance(val, tuple) else val


def _hdf5_deserialize_collection(name: str, val: Any) -> Any:
    """读采集元数据 attrs 时，将 list 还原为 tuple（见 ``COLLECTION_TUPLE_FIELDS``）。"""
    if name in COLLECTION_TUPLE_FIELDS:
        return tuple(float(x) for x in val)
    return val


@dataclass(frozen=True)
class CollectionMetadata:
    """一次采集运行的可复现配置摘要，序列化到 HDF5 根属性。

    由 ``run_data_collection.py`` CLI 经 ``from_args`` 解析；``build_sampler`` 构建预采样池。

    字段
    ----
    - ``seed``：随机种子
    - ``roi``：平面 ROI ``[xmin, xmax, ymin, ymax]``（m），z 固定为 0
    - ``position_sampling_mode``：``uniform`` / ``gaussian``
    - ``speed_range``：速度模值 ``[vmin, vmax]``（m/s）
    - ``speed_sampling_mode``：``uniform`` / ``gaussian``
    - ``num_samples``：目标采纳 episode 数
    - ``sampler_pool_factor``：预采样池倍数（池大小 = ``num_samples × sampler_pool_factor``）
    """

    seed: int
    roi: tuple[float, float, float, float]
    position_sampling_mode: SamplingMode = "uniform"
    speed_range: tuple[float, float] = (0.0, 0.0)
    speed_sampling_mode: SamplingMode = "uniform"
    num_samples: int = 20000
    sampler_pool_factor: int = 5

    @property
    def pool_size(self) -> int:
        """预采样池大小：``num_samples × sampler_pool_factor``。"""
        return self.num_samples * self.sampler_pool_factor

    def build_sampler(self) -> RoiKinematicsSampler:
        """按本元数据构建 ``RoiKinematicsSampler``（池大小为 ``pool_size``）。"""
        return RoiKinematicsSampler(
            roi=self.roi,
            position_sampling_mode=self.position_sampling_mode,
            speed_range=self.speed_range,
            speed_sampling_mode=self.speed_sampling_mode,
            num_samples=self.pool_size,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> CollectionMetadata:
        """从 ``run_data_collection.py`` CLI 参数解析并设置随机种子。"""
        seed = int(args.seed)
        num_samples = int(args.num_samples)
        sampler_pool_factor = int(args.sampler_pool_factor)
        if num_samples < 1:
            raise ValueError("num_samples 须 >= 1")
        if sampler_pool_factor < 1:
            raise ValueError("sampler_pool_factor 须 >= 1")
        set_random_seed(seed)
        return cls(
            seed=seed,
            roi=RoiKinematicsSampler.parse_roi_xy(args.roi),
            position_sampling_mode=_parse_sampling_mode(
                args.position_sampling_mode,
                field="position_sampling_mode",
            ),
            speed_range=RoiKinematicsSampler.parse_speed_range(args.speed_range),
            speed_sampling_mode=_parse_sampling_mode(
                args.speed_sampling_mode,
                field="speed_sampling_mode",
            ),
            num_samples=num_samples,
            sampler_pool_factor=sampler_pool_factor,
        )

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """写入采集元数据根属性（``seed``、``roi`` 等）。"""
        for key, val in asdict(self).items():
            f.attrs[key] = _hdf5_serialize(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata:
        """从根属性读取采集元数据；缺字段时 ``ValueError``。"""
        missing = [fld.name for fld in fields(cls) if fld.name not in f.attrs]
        if missing:
            raise ValueError(
                f"HDF5 缺少采集元数据根属性: {', '.join(missing)}"
            )
        kwargs: dict[str, Any] = {}
        for fld in fields(cls):
            kwargs[fld.name] = _hdf5_deserialize_collection(
                fld.name, f.attrs[fld.name]
            )
        return cls(**kwargs)


@dataclass
class RTDatasetWriter:
    """HDF5 采集流式写入器：``open`` → ``append_episode`` → ``finalize``。"""

    path: Path
    bs_pos: np.ndarray
    compression: str | None = "lzf"
    _file: h5py.File | None = field(default=None, repr=False)
    _h_dd_ds: h5py.Dataset | None = field(default=None, repr=False)
    _pos_ds: h5py.Dataset | None = field(default=None, repr=False)
    _vel_ds: h5py.Dataset | None = field(default=None, repr=False)
    _writer_count: int = 0
    _finalized: bool = False

    def __post_init__(self) -> None:
        self.bs_pos = np.asarray(self.bs_pos, dtype=np.float64).reshape(-1)
        self.path = Path(self.path)
        if self.compression in (None, "none"):
            self.compression = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        bs_pos: np.ndarray,
        *,
        compression: str | None = "lzf",
    ) -> RTDatasetWriter:
        """打开 HDF5 文件，准备按 episode 流式写入。

        ``compression`` 可选 ``lzf``（默认）、``gzip`` 或 ``none``。
        """
        return cls(path=path, bs_pos=bs_pos, compression=compression)

    def __len__(self) -> int:
        return self._writer_count

    def __repr__(self) -> str:
        return f"RTDatasetWriter(count={self._writer_count}, path={self.path})"

    def __enter__(self) -> RTDatasetWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        """未 ``finalize`` 时异常退出仍关闭 HDF5 文件。"""
        if self._file is not None and not self._finalized:
            self._file.close()
            self._file = None

    def append_episode(
        self,
        h_dd: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> None:
        """追加单条 episode 的 h_dd 与运动学。

        ``h_dd`` 为 ``(H, W)`` complex64；``pos`` / ``vel`` 为 ``(3,)`` float64。
        首次调用时创建 HDF5 可扩展 datasets。
        """
        h_dd_arr = np.asarray(h_dd, dtype=np.complex64)
        pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
        if self._file is None:
            self._open_writer(h_dd_arr)
        idx = self._writer_count
        self._resize_writer(idx + 1)
        assert self._h_dd_ds is not None
        assert self._pos_ds is not None
        assert self._vel_ds is not None
        self._h_dd_ds[idx] = h_dd_arr
        self._pos_ds[idx] = pos_row
        self._vel_ds[idx] = vel_row
        self._writer_count += 1

    def _open_writer(self, h_dd: np.ndarray) -> None:
        """首次 ``append_episode`` 时创建 HDF5 文件与可扩展 datasets。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.create_dataset(DATASET_KEY_BS_POS, data=self.bs_pos)

        h_dd_kwargs: dict[str, Any] = {
            "maxshape": (None,) + tuple(h_dd.shape),
            "chunks": (1,) + tuple(h_dd.shape),
        }
        if self.compression:
            h_dd_kwargs["compression"] = self.compression

        self._h_dd_ds = self._file.create_dataset(
            DATASET_KEY_H_DD,
            shape=(0,) + tuple(h_dd.shape),
            dtype=h_dd.dtype,
            **h_dd_kwargs,
        )
        self._pos_ds = _create_vec3_dataset(self._file, DATASET_KEY_TARGET_POSITION)
        self._vel_ds = _create_vec3_dataset(self._file, DATASET_KEY_TARGET_VELOCITY)

    def _resize_writer(self, new_count: int) -> None:
        """将 h_dd / 运动学 datasets 第一维扩展至 ``new_count``。"""
        assert self._h_dd_ds is not None
        assert self._pos_ds is not None
        assert self._vel_ds is not None
        self._h_dd_ds.resize((new_count,) + self._h_dd_ds.shape[1:])
        self._pos_ds.resize((new_count, 3))
        self._vel_ds.resize((new_count, 3))

    def finalize(
        self,
        *,
        collection_meta: CollectionMetadata,
        scene_slug: str,
    ) -> None:
        """写入根属性并关闭 HDF5 文件；无 episode 时 ``ValueError``。"""
        if self._file is None:
            raise ValueError("RTDatasetWriter 无 episode 数据")
        if self._finalized:
            return
        _write_hdf5_root_metadata(
            self._file,
            n_episodes=self._writer_count,
            collection_meta=collection_meta,
            scene_slug=scene_slug,
        )
        self._file.close()
        self._file = None
        self._finalized = True


@dataclass
class RTDataset(Dataset):
    """ISAC HDF5 只读数据集（内存驻留），供训练与评估。

    典型形状：``h_dd`` 为 ``(N, H, W)`` 复数；运动学为 ``(N, 3)``。

    读取 / 训练
    -----------
    - ``RTDataset.load(path)``
    - ``len(dataset)`` → episode 条数
    - ``dataset[i]`` → ``dict`` 含 ``features``、``range_m``、``velocity_mps``、``slot``
    - ``spectrum_tensor(i)`` → 单条复数 h_dd（评估用）

    采集写入请使用 ``RTDatasetWriter.open``。
    """

    bs_pos: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    h_dd: np.ndarray
    collection_meta: CollectionMetadata
    sensing_performance: Any | None = field(default=None, repr=False, compare=False)

    def bind_sensing_performance(self, sensing_performance: Any) -> None:
        """绑定感知性能对象，供 ``__getitem__`` 生成 MusicPeaks 局部 bin 标签。"""
        self.sensing_performance = sensing_performance

    def __post_init__(self) -> None:
        self.bs_pos = np.asarray(self.bs_pos, dtype=np.float64).reshape(-1)

    def _validate_slot_index(self, idx: int) -> None:
        n = len(self)
        if idx < 0 or idx >= n:
            raise IndexError(f"index {idx} out of range for {n} slots")

    def __len__(self) -> int:
        return int(self.h_dd.shape[0])

    def spectrum_tensor(
        self, idx: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """返回单条 episode 的复数 h_dd 张量。"""
        self._validate_slot_index(idx)
        t = torch.from_numpy(self.h_dd[idx])
        if device is not None:
            t = t.to(device)
        return t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """CNN 训练样本 dict。

        返回键
        ------
        - ``features``：``(2, H, W)`` float32 幅相双通道（幅度 dB + 相位）
        - ``peaks_delay`` / ``peaks_doppler``：ROI 局部 bin 监督 (float32 标量)
        - ``range_m`` / ``velocity_mps``：单基地几何标签 (float32 标量)
        - ``slot``：episode 索引 (int64)
        """
        self._validate_slot_index(idx)
        if self.sensing_performance is None:
            raise ValueError(
                "RTDataset.__getitem__ 须先调用 bind_sensing_performance(sp)"
            )
        h_dd = torch.from_numpy(self.h_dd[idx])
        features = dd_spectrum_to_features(h_dd)
        range_m, vel_mps = monostatic_range_velocity(
            self.target_position[idx],
            self.target_velocity[idx],
            self.bs_pos,
        )
        num_doppler_bins = h_dd.shape[0]
        metric = SpectrumMetric(self.sensing_performance)
        delay_bin, doppler_bin = metric.physical_to_local_bins(
            range_m,
            vel_mps,
            num_doppler_bins=num_doppler_bins,
            sens_mode="monostatic",
        )
        peaks = MusicPeaks.from_local_bins(
            delay_bin,
            doppler_bin,
            device="cpu",
        )
        return {
            "features": features.to(dtype=torch.float32),
            "peaks_delay": peaks.peaks_delay[0].to(dtype=torch.float32),
            "peaks_doppler": peaks.peaks_doppler[0].to(dtype=torch.float32),
            "range_m": torch.tensor(range_m, dtype=torch.float32),
            "velocity_mps": torch.tensor(vel_mps, dtype=torch.float32),
            "slot": torch.tensor(idx, dtype=torch.int64),
        }

    def __repr__(self) -> str:
        return f"RTDataset(n={len(self)}, h_dd_shape={self.h_dd.shape})"

    @classmethod
    def load(cls, filepath: str | Path) -> RTDataset:
        """从 HDF5 加载至内存。"""
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            arrays = _read_array_datasets(f)
            return cls(
                bs_pos=arrays["bs_pos"],
                target_position=arrays["target_position"],
                target_velocity=arrays["target_velocity"],
                h_dd=arrays["h_dd"],
                collection_meta=CollectionMetadata.read_hdf5_attrs(f),
            )

# 
def save_collection_artifacts(
    *,
    scene_slug: str,
    config_file: str | Path,
    csv_rows: list[dict[str, str | int]],
    rt_simulator: RTSimulator,
    out_dir: Path,
) -> None:
    """写出采集产物：TOML、CSV、场景 PNG。HDF5 由采集循环内 ``RTDatasetWriter.finalize`` 完成。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置文件
    src = resolve_config_path(config_file)
    shutil.copy2(src, out_dir / src.name)

    # 保存 CSV 文件
    if csv_rows:
        csv_path = out_dir / f"{scene_slug}{EPISODE_CSV_SUFFIX}"
        keys = list(EPISODE_CSV_COLUMNS)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=keys, restval="")
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({k: row.get(k, "") for k in keys})

    # 保存场景 PNG 文件
    rt_simulator.render_to_file(
        f"{scene_slug}{SCENE_PNG_SUFFIX}",
        output_dir=out_dir,
    )

    print(f"采集产物已保存至: {out_dir}")
