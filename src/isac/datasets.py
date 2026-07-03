"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

由 ``run_data_collection.py`` 写入；训练/评估侧通过 ``RTDataset.load`` 消费。
采集期完成 CFR→h_dd 感知链；HDF5 仅存 ROI 裁剪后的复数 DD 谱与运动学。
``RTDataset`` 继承 ``torch.utils.data.Dataset``，``__getitem__`` 返回 CNN 训练 dict。

HDF5 文件布局
--------------
根 datasets:

- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``delay_doppler_spectrum``：复数 h_dd，shape ``(N, H, W)``（ROI 裁剪后）

根 attrs:

- ``num_slots``, ``description``
- ``collection_*``：由 ``CollectionMetadata`` 写入
- ``sensing_*``：由 ``SensingMetadata`` 写入（ROI、分辨率、SNR 等）

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

from isac import DEFAULT_COLLECTION_OUT_DIR
from isac.models.dd_spectrum import (
    dd_spectrum_to_features,
    monostatic_labels_from_kinematics,
)
from isac.utils.config_loader import resolve_config_path

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator
    from isac.system import System

# --- 路径与键名常量 ---

_EPISODE_CSV_SUFFIX = "_mc_dataset_episodes.csv"
_H5_SUFFIX = "_mc_sionna_dataset.h5"
_SCENE_PNG_SUFFIX = "_scene.png"
_LEGACY_DATASET_KEY_CFR = "channel_frequency_response"

_EPISODE_CSV_COLUMNS = (
    "sample_idx",
    "position",
    "velocity",
    "true_range_m",
    "true_radial_velocity_mps",
)

_DATASET_KEY_H_DD = "delay_doppler_spectrum"
_DATASET_KEY_TARGET_POSITION = "target_position"
_DATASET_KEY_TARGET_VELOCITY = "target_velocity"
_DATASET_KEY_BS_POS = "bs_pos"

_META_KEY_NUM_SLOTS = "num_slots"
_META_KEY_DESCRIPTION = "description"
_META_PREFIX_COLLECTION = "collection_"
_META_PREFIX_SENSING = "sensing_"

_COLLECTION_TUPLE_FIELDS = frozenset({"roi", "speed_range"})

_ARRAY_DATASET_SPECS: tuple[tuple[str, str, bool], ...] = (
    (_DATASET_KEY_BS_POS, "bs_pos", False),
    (_DATASET_KEY_TARGET_POSITION, "target_position", False),
    (_DATASET_KEY_TARGET_VELOCITY, "target_velocity", False),
    (_DATASET_KEY_H_DD, "h_dd", True),
)


def collection_h5_path(scene_slug: str, out_dir: Path) -> Path:
    """HDF5 输出路径 ``{out_dir}/{scene_slug}_mc_sionna_dataset.h5``."""
    return out_dir / f"{scene_slug}{_H5_SUFFIX}"


def collection_scene_png_path(scene_slug: str, out_dir: Path) -> Path:
    """场景渲染 PNG 路径 ``{out_dir}/{scene_slug}_scene.png``。"""
    return out_dir / f"{scene_slug}{_SCENE_PNG_SUFFIX}"


def collection_dataset_description(scene_slug: str, n_episodes: int) -> str:
    """生成写入根属性 ``description`` 的英文描述。"""
    return (
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) "
        f"in {scene_slug}"
    )


def _resolve_out_dir(output_root: Path | None) -> Path:
    """解析采集输出目录并确保存在。"""
    out = output_root or DEFAULT_COLLECTION_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- HDF5 读写辅助 ---


def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """返回指定名称的数据集；缺失时抛出 ``KeyError`` 或格式错误提示。"""
    if key not in f:
        if key == _DATASET_KEY_H_DD and _LEGACY_DATASET_KEY_CFR in f:
            raise ValueError(
                "HDF5 为旧 CFR 格式，请重新运行 run_data_collection.py 采集 h_dd 数据集"
            )
        if key == _DATASET_KEY_H_DD:
            raise KeyError(f"HDF5 缺少必选数据集 {key!r}（delay_doppler_spectrum）。")
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


def _read_array_datasets(f: h5py.File) -> dict[str, np.ndarray]:
    """读取 h_dd、运动学与 ``bs_pos`` ndarray。"""
    return {
        attr: _require_dataset(f, h5_key)[:] for h5_key, attr, _ in _ARRAY_DATASET_SPECS
    }


def _write_array_datasets(f: h5py.File, ds: RTDataset) -> None:
    """写入四个 ndarray 数据集（h_dd 使用 gzip）。"""
    for h5_key, attr, gzip in _ARRAY_DATASET_SPECS:
        kwargs = {"compression": "gzip"} if gzip else {}
        f.create_dataset(h5_key, data=getattr(ds, attr), **kwargs)


def _write_root_attrs(
    f: h5py.File, ds: RTDataset, *, scene_slug: str | None = None
) -> None:
    """写入 ``num_slots``、``description``、``collection_*`` 与 ``sensing_*`` 根属性。"""
    f.attrs[_META_KEY_NUM_SLOTS] = ds.num_slots
    if ds.collection_meta is not None:
        if scene_slug is not None:
            f.attrs[_META_KEY_DESCRIPTION] = collection_dataset_description(
                scene_slug, ds.num_slots
            )
        ds.collection_meta.write_hdf5_attrs(f)
    if ds.sensing_meta is not None:
        ds.sensing_meta.write_hdf5_attrs(f)


# --- 数据类 ---


@dataclass
class EpisodeBuffers:
    """主循环共享的 episode 级写出缓冲。"""

    h_dd_list: list[np.ndarray] = field(default_factory=list)
    target_pos_list: list[np.ndarray] = field(default_factory=list)
    target_vel_list: list[np.ndarray] = field(default_factory=list)
    csv_rows: list[dict[str, str | int]] = field(default_factory=list)


def _collection_attr_key(name: str) -> str:
    return f"{_META_PREFIX_COLLECTION}{name}"


def _sensing_attr_key(name: str) -> str:
    return f"{_META_PREFIX_SENSING}{name}"


def _hdf5_serialize(val: Any) -> Any:
    return list(val) if isinstance(val, tuple) else val


def _hdf5_deserialize_collection(name: str, val: Any) -> Any:
    if name in _COLLECTION_TUPLE_FIELDS:
        return tuple(float(x) for x in val)
    return val


@dataclass(frozen=True)
class CollectionMetadata:
    """一次采集运行的可复现配置摘要。"""

    seed: int
    roi: tuple[float, float, float, float]
    position_sampling_mode: str = "uniform"
    speed_range: tuple[float, float] = (0.0, 0.0)
    speed_sampling_mode: str = "uniform"

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        for key, val in asdict(self).items():
            f.attrs[_collection_attr_key(key)] = _hdf5_serialize(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata | None:
        if _collection_attr_key("seed") not in f.attrs:
            return None
        kwargs: dict[str, Any] = {}
        for fld in fields(cls):
            attr_key = _collection_attr_key(fld.name)
            if attr_key not in f.attrs:
                continue
            kwargs[fld.name] = _hdf5_deserialize_collection(fld.name, f.attrs[attr_key])
        return cls(**kwargs)

    @classmethod
    def from_collection_args(cls, args: argparse.Namespace) -> CollectionMetadata:
        return cls(
            seed=int(args.seed),
            roi=tuple(map(float, args.roi)),
            position_sampling_mode=str(args.position_sampling_mode),
            speed_range=tuple(map(float, args.speed_range)),
            speed_sampling_mode=str(args.speed_sampling_mode),
        )


@dataclass(frozen=True)
class SensingMetadata:
    """感知链配置摘要，序列化到 HDF5 根属性 ``sensing_<field>``。"""

    max_range_m: float
    max_velocity_mps: float
    range_resolution: float
    velocity_resolution: float
    snr_db: float

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        for key, val in asdict(self).items():
            f.attrs[_sensing_attr_key(key)] = float(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> SensingMetadata | None:
        if _sensing_attr_key("max_range_m") not in f.attrs:
            return None
        kwargs: dict[str, Any] = {}
        for fld in fields(cls):
            attr_key = _sensing_attr_key(fld.name)
            if attr_key not in f.attrs:
                continue
            kwargs[fld.name] = float(f.attrs[attr_key])
        return cls(**kwargs)

    @classmethod
    def from_system(cls, system: System) -> SensingMetadata:
        sp = system.components.sensing_performance
        if sp is None:
            raise ValueError("采集要求已构建 sensing_performance 组件")
        dd = system.components.delay_doppler_spectrum
        if dd is None or not dd.has_roi:
            raise ValueError(
                "采集要求配置 [dd_spectrum_roi]（max_range_m / max_velocity_mps）"
            )
        return cls(
            max_range_m=float(dd.max_range_m),
            max_velocity_mps=float(dd.max_velocity_mps),
            range_resolution=float(sp.range_resolution),
            velocity_resolution=float(sp.velocity_resolution),
            snr_db=float(system.params.channel.snr_db),
        )


@dataclass
class RTDataset(Dataset):
    """ISAC HDF5 数据集的内存表示（h_dd + kinematics）。"""

    bs_pos: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    h_dd: np.ndarray
    collection_meta: CollectionMetadata | None = None
    sensing_meta: SensingMetadata | None = None
    use_phase: bool = True

    @property
    def num_slots(self) -> int:
        return int(self.h_dd.shape[0])

    @property
    def range_resolution(self) -> float:
        if self.sensing_meta is None:
            raise ValueError("RTDataset 缺少 sensing_meta")
        return self.sensing_meta.range_resolution

    @property
    def velocity_resolution(self) -> float:
        if self.sensing_meta is None:
            raise ValueError("RTDataset 缺少 sensing_meta")
        return self.sensing_meta.velocity_resolution

    @property
    def max_range_m(self) -> float:
        if self.sensing_meta is None:
            raise ValueError("RTDataset 缺少 sensing_meta")
        return self.sensing_meta.max_range_m

    @property
    def max_velocity_mps(self) -> float:
        if self.sensing_meta is None:
            raise ValueError("RTDataset 缺少 sensing_meta")
        return self.sensing_meta.max_velocity_mps

    def __len__(self) -> int:
        return self.num_slots

    def spectrum_tensor(
        self, idx: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        if idx < 0 or idx >= self.num_slots:
            raise IndexError(f"index {idx} out of range for {self.num_slots} slots")
        t = torch.from_numpy(self.h_dd[idx])
        if device is not None:
            t = t.to(device)
        return t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_slots:
            raise IndexError(f"index {idx} out of range for {self.num_slots} slots")
        h_dd = torch.from_numpy(self.h_dd[idx])
        features = dd_spectrum_to_features(h_dd, use_phase=self.use_phase)
        range_m, vel_mps = monostatic_labels_from_kinematics(
            self.target_position[idx],
            self.target_velocity[idx],
            self.bs_pos,
        )
        return {
            "features": features.to(dtype=torch.float32),
            "range_m": torch.tensor(range_m, dtype=torch.float32),
            "velocity_mps": torch.tensor(vel_mps, dtype=torch.float32),
            "slot": torch.tensor(idx, dtype=torch.int64),
        }

    def __repr__(self) -> str:
        return f"RTDataset(num_slots={self.num_slots}, h_dd_shape={self.h_dd.shape})"

    @classmethod
    def from_buffers(
        cls,
        buffers: EpisodeBuffers,
        bs_pos: np.ndarray,
        *,
        collection_meta: CollectionMetadata | None = None,
        sensing_meta: SensingMetadata | None = None,
    ) -> RTDataset:
        if not buffers.h_dd_list:
            raise ValueError("EpisodeBuffers 无 h_dd 数据")
        return cls(
            bs_pos=bs_pos,
            target_position=np.array(buffers.target_pos_list),
            target_velocity=np.array(buffers.target_vel_list),
            h_dd=np.array(buffers.h_dd_list),
            collection_meta=collection_meta,
            sensing_meta=sensing_meta,
        )

    @classmethod
    def load(cls, filepath: str | Path, *, use_phase: bool = True) -> RTDataset:
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            arrays = _read_array_datasets(f)
            return cls(
                **arrays,
                collection_meta=CollectionMetadata.read_hdf5_attrs(f),
                sensing_meta=SensingMetadata.read_hdf5_attrs(f),
                use_phase=use_phase,
            )

    def save(self, filepath: str | Path, *, scene_slug: str | None = None) -> None:
        path = Path(filepath)
        with h5py.File(path, "w") as f:
            _write_array_datasets(f, self)
            _write_root_attrs(f, self, scene_slug=scene_slug)


class Hdf5CollectionWriter:
    """采集期按 episode 流式写入 HDF5。"""

    def __init__(
        self,
        path: str | Path,
        bs_pos: np.ndarray,
        *,
        compression: str | None = "lzf",
    ) -> None:
        self._path = Path(path)
        self._bs_pos = np.asarray(bs_pos, dtype=np.float64).reshape(-1)
        self._compression = None if compression in (None, "none") else compression
        self._file: h5py.File | None = None
        self._h_dd_ds: h5py.Dataset | None = None
        self._pos_ds: h5py.Dataset | None = None
        self._vel_ds: h5py.Dataset | None = None
        self._count = 0
        self._finalized = False

    def __enter__(self) -> Hdf5CollectionWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None and not self._finalized:
            self._file.close()
            self._file = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> Path:
        return self._path

    def append_episode(
        self,
        h_dd: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> None:
        h_dd_arr = np.asarray(h_dd, dtype=np.complex64)
        pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
        if self._file is None:
            self._open(h_dd_arr)
        idx = self._count
        self._resize(idx + 1)
        assert self._h_dd_ds is not None
        assert self._pos_ds is not None
        assert self._vel_ds is not None
        self._h_dd_ds[idx] = h_dd_arr
        self._pos_ds[idx] = pos_row
        self._vel_ds[idx] = vel_row
        self._count += 1

    def _open(self, h_dd: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset(_DATASET_KEY_BS_POS, data=self._bs_pos)

        h_dd_chunks = (1,) + tuple(h_dd.shape)
        h_dd_kwargs: dict[str, Any] = {
            "maxshape": (None,) + tuple(h_dd.shape),
            "chunks": h_dd_chunks,
        }
        if self._compression:
            h_dd_kwargs["compression"] = self._compression

        self._h_dd_ds = self._file.create_dataset(
            _DATASET_KEY_H_DD,
            shape=(0,) + tuple(h_dd.shape),
            dtype=h_dd.dtype,
            **h_dd_kwargs,
        )
        self._pos_ds = self._file.create_dataset(
            _DATASET_KEY_TARGET_POSITION,
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(1024, 3),
        )
        self._vel_ds = self._file.create_dataset(
            _DATASET_KEY_TARGET_VELOCITY,
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(1024, 3),
        )

    def _resize(self, new_count: int) -> None:
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
        sensing_meta: SensingMetadata,
    ) -> None:
        if self._file is None:
            raise ValueError("Hdf5CollectionWriter 无 episode 数据")
        if self._finalized:
            return
        self._file.attrs[_META_KEY_NUM_SLOTS] = self._count
        self._file.attrs[_META_KEY_DESCRIPTION] = collection_dataset_description(
            scene_slug, self._count
        )
        collection_meta.write_hdf5_attrs(self._file)
        sensing_meta.write_hdf5_attrs(self._file)
        self._file.close()
        self._file = None
        self._finalized = True


def _save_collection_config(
    *,
    config_file: str | Path,
    output_root: Path,
) -> Path:
    src = resolve_config_path(config_file)
    dst = output_root / src.name
    shutil.copy2(src, dst)
    return dst


def _save_episodes_csv(
    *,
    scene_slug: str,
    rows: list[dict[str, str | int]],
    output_root: Path,
) -> None:
    if not rows:
        return
    path = output_root / f"{scene_slug}{_EPISODE_CSV_SUFFIX}"
    keys = list(_EPISODE_CSV_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=keys, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _save_scene_render(
    rt_simulator: RTSimulator,
    scene_slug: str,
    output_root: Path,
) -> Path:
    filename = f"{scene_slug}{_SCENE_PNG_SUFFIX}"
    return rt_simulator.render_to_file(filename, output_dir=output_root)


def save_collection_artifacts(
    *,
    scene_slug: str,
    config_file: str | Path,
    buffers: EpisodeBuffers,
    bs_pos: np.ndarray,
    args: argparse.Namespace,
    rt_simulator: RTSimulator,
    sensing_meta: SensingMetadata,
    out_dir: Path | None = None,
    h5_already_written: bool = False,
) -> None:
    collection_meta = CollectionMetadata.from_collection_args(args)
    target_dir = _resolve_out_dir(out_dir)
    _save_collection_config(config_file=config_file, output_root=target_dir)
    _save_episodes_csv(
        scene_slug=scene_slug,
        rows=buffers.csv_rows,
        output_root=target_dir,
    )
    if not h5_already_written and buffers.h_dd_list:
        RTDataset.from_buffers(
            buffers,
            bs_pos,
            collection_meta=collection_meta,
            sensing_meta=sensing_meta,
        ).save(collection_h5_path(scene_slug, target_dir), scene_slug=scene_slug)
    _save_scene_render(rt_simulator, scene_slug, target_dir)
    print(f"采集产物已保存至: {target_dir}")
