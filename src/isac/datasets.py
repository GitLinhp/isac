"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

采集期：``RTDataset.open_for_collection`` 按 episode 流式写入 HDF5。
消费侧：``RTDataset.load`` 加载内存；``__getitem__`` 返回 CNN 训练 dict。
采集期完成 CFR→h_dd 感知链；HDF5 仅存 ROI 裁剪后的复数 DD 谱与运动学。
``RTDataset`` 继承 ``torch.utils.data.Dataset``。

RTDataset 双模式
----------------
- **写入模式**：``open_for_collection`` → ``append_episode`` → ``finalize``；
  内存数组为占位，未完成 ``finalize`` 前不可 ``__getitem__``。
- **读取模式**：``load`` 后 ``h_dd`` 等数组驻内存，供 DataLoader / 评估脚本使用。

HDF5 文件布局
--------------
根 datasets:

- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``delay_doppler_spectrum``：复数 h_dd，shape ``(N, H, W)``（ROI 裁剪后）

根 attrs:

- ``num_slots``, ``description``
- ``collection_*``：由 ``CollectionMetadata`` 写入
- ``sensing_*``：由 ``SensingMetadata`` 写入（ROI、分辨率等）

采集落盘产物（``data/``）:

- TOML：采集配置副本（保留原文件名）
- ``{scene_slug}_mc_dataset_episodes.csv``
- ``{scene_slug}_mc_sionna_dataset.h5``
- ``{scene_slug}_scene.png``
"""

from __future__ import annotations

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
    from isac.data_structures import CollectionSamplingParams

# --- 路径与键名常量 ---

_EPISODE_CSV_SUFFIX = "_mc_dataset_episodes.csv"
_H5_SUFFIX = "_mc_sionna_dataset.h5"
_SCENE_PNG_SUFFIX = "_scene.png"
# 旧 HDF5 格式键名；``load`` 时若存在则提示重新采集 h_dd 数据集
_LEGACY_DATASET_KEY_CFR = "channel_frequency_response"

# 与 ``EpisodeBuffers.csv_rows`` 字典键顺序一致
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

# 读 HDF5 ``collection_*`` attrs 时需还原为 tuple 的字段名
_COLLECTION_TUPLE_FIELDS = frozenset({"roi", "speed_range"})

# HDF5 dataset 键名 → ``RTDataset`` 属性名（供 ``_read_array_datasets`` 使用）
_ARRAY_DATASET_SPECS: tuple[tuple[str, str], ...] = (
    (_DATASET_KEY_BS_POS, "bs_pos"),
    (_DATASET_KEY_TARGET_POSITION, "target_position"),
    (_DATASET_KEY_TARGET_VELOCITY, "target_velocity"),
    (_DATASET_KEY_H_DD, "h_dd"),
)


def collection_h5_path(scene_slug: str, out_dir: Path) -> Path:
    """HDF5 输出路径 ``{out_dir}/{scene_slug}_mc_sionna_dataset.h5``。"""
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


def _meta_attr_key(prefix: str, name: str) -> str:
    """HDF5 根属性键名：``{prefix}{field}``（如 ``collection_seed``）。"""
    return f"{prefix}{name}"


def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """返回指定名称的 HDF5 dataset；缺失或旧 CFR 格式时抛出明确错误。"""
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
        attr: _require_dataset(f, h5_key)[:] for h5_key, attr in _ARRAY_DATASET_SPECS
    }


def _write_hdf5_root_metadata(
    f: h5py.File,
    *,
    num_slots: int,
    collection_meta: CollectionMetadata | None,
    sensing_meta: SensingMetadata | None,
    scene_slug: str | None = None,
) -> None:
    """写入 ``num_slots``、``description``、``collection_*`` 与 ``sensing_*`` 根属性。"""
    f.attrs[_META_KEY_NUM_SLOTS] = num_slots
    if collection_meta is not None:
        if scene_slug is not None:
            f.attrs[_META_KEY_DESCRIPTION] = collection_dataset_description(
                scene_slug, num_slots
            )
        collection_meta.write_hdf5_attrs(f)
    if sensing_meta is not None:
        sensing_meta.write_hdf5_attrs(f)


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


@dataclass
class EpisodeBuffers:
    """主循环共享的 episode 级 CSV 缓冲。

    ``run_data_collection`` 采集循环逐条追加 kinematics 行；循环结束后由
    ``save_collection_artifacts`` 写出 CSV。h_dd 由 ``RTDataset.open_for_collection``
    流式写入 HDF5。

    每行 dict 的键与 ``_EPISODE_CSV_COLUMNS`` 一致：
    ``sample_idx``, ``position``, ``velocity``, ``true_range_m``,
    ``true_radial_velocity_mps``。
    """

    csv_rows: list[dict[str, str | int]] = field(default_factory=list)


def _hdf5_serialize(val: Any) -> Any:
    """将 tuple 转为 list，供 h5py 根属性写入。"""
    return list(val) if isinstance(val, tuple) else val


def _hdf5_deserialize_collection(name: str, val: Any) -> Any:
    """读 ``collection_*`` attrs 时，将 list 还原为 tuple（见 ``_COLLECTION_TUPLE_FIELDS``）。"""
    if name in _COLLECTION_TUPLE_FIELDS:
        return tuple(float(x) for x in val)
    return val


@dataclass(frozen=True)
class CollectionMetadata:
    """一次采集运行的可复现配置摘要，序列化到 HDF5 根属性 ``collection_<field>``。

    采样字段来自 TOML ``[monte_carlo_sampling]``；``seed`` 来自 CLI。

    字段
    ----
    - ``seed``：随机种子
    - ``roi``：平面 ROI ``[xmin, xmax, ymin, ymax]``（m），z 固定为 0
    - ``position_sampling_mode``：``uniform`` / ``gaussian``
    - ``speed_range``：速度模值 ``[vmin, vmax]``（m/s）
    - ``speed_sampling_mode``：``uniform`` / ``gaussian``
    """

    seed: int
    roi: tuple[float, float, float, float]
    position_sampling_mode: str = "uniform"
    speed_range: tuple[float, float] = (0.0, 0.0)
    speed_sampling_mode: str = "uniform"

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """写入 ``collection_<field>`` 根属性。"""
        for key, val in asdict(self).items():
            f.attrs[_meta_attr_key(_META_PREFIX_COLLECTION, key)] = _hdf5_serialize(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata | None:
        """从根属性读取；缺少 ``collection_seed`` 时返回 ``None``。"""
        if _meta_attr_key(_META_PREFIX_COLLECTION, "seed") not in f.attrs:
            return None
        kwargs: dict[str, Any] = {}
        for fld in fields(cls):
            attr_key = _meta_attr_key(_META_PREFIX_COLLECTION, fld.name)
            if attr_key not in f.attrs:
                continue
            kwargs[fld.name] = _hdf5_deserialize_collection(fld.name, f.attrs[attr_key])
        return cls(**kwargs)

    @classmethod
    def from_sampling_params(
        cls,
        seed: int,
        params: CollectionSamplingParams,
    ) -> CollectionMetadata:
        """从 CLI ``seed`` 与 ``[monte_carlo_sampling]`` 解析结果构造。"""
        return cls(
            seed=int(seed),
            roi=params.roi,
            position_sampling_mode=str(params.position_sampling_mode),
            speed_range=params.speed_range,
            speed_sampling_mode=str(params.speed_sampling_mode),
        )


@dataclass(frozen=True)
class SensingMetadata:
    """感知链配置摘要，序列化到 HDF5 根属性 ``sensing_<field>``。

    由 ``from_system`` 在采集启动时从 ``System`` 组件提取。

    字段
    ----
    - ``max_range_m``：DD 谱 ROI 最大距离（m）
    - ``max_velocity_mps``：DD 谱 ROI 最大多普勒速度（m/s）
    - ``range_resolution``：距离分辨率（m）
    - ``velocity_resolution``：速度分辨率（m/s）
    """

    max_range_m: float
    max_velocity_mps: float
    range_resolution: float
    velocity_resolution: float

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """写入 ``sensing_<field>`` 根属性。"""
        for key, val in asdict(self).items():
            f.attrs[_meta_attr_key(_META_PREFIX_SENSING, key)] = float(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> SensingMetadata | None:
        """从根属性读取；缺少 ``sensing_max_range_m`` 时返回 ``None``。"""
        if _meta_attr_key(_META_PREFIX_SENSING, "max_range_m") not in f.attrs:
            return None
        kwargs: dict[str, Any] = {}
        for fld in fields(cls):
            attr_key = _meta_attr_key(_META_PREFIX_SENSING, fld.name)
            if attr_key not in f.attrs:
                continue
            kwargs[fld.name] = float(f.attrs[attr_key])
        return cls(**kwargs)

    @classmethod
    def from_system(cls, system: System) -> SensingMetadata:
        """从已构建 ``System`` 提取 ROI 与分辨率（需 ``[dd_spectrum_roi]``）。"""
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
        )


class RTDataset(Dataset):
    """ISAC HDF5 数据集：采集流式写入与内存读取。

    典型形状：``h_dd`` 为 ``(num_slots, H, W)`` 复数；运动学为 ``(num_slots, 3)``。

    采集写入
    --------
    ``with RTDataset.open_for_collection(path, bs_pos) as ds:``
    ``ds.append_episode(...)`` → ``ds.finalize(...)``

    读取 / 训练
    -----------
    - ``RTDataset.load(path)``
    - ``len(dataset)`` → ``num_slots``
    - ``dataset[i]`` → ``dict`` 含 ``features``、``range_m``、``velocity_mps``、``slot``
    - ``spectrum_tensor(i)`` → 单条复数 h_dd（评估用）
    """

    def __init__(
        self,
        *,
        bs_pos: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        h_dd: np.ndarray,
        collection_meta: CollectionMetadata | None = None,
        sensing_meta: SensingMetadata | None = None,
        use_phase: bool = True,
    ) -> None:
        """构造数据集实例（读取模式或 ``open_for_collection`` 占位）。

        公开字段约定
        ------------
        - ``bs_pos``：``(3,)`` float64
        - ``target_position`` / ``target_velocity``：``(N, 3)`` float64
        - ``h_dd``：``(N, H, W)`` complex64（写入模式下为占位空数组）
        - ``use_phase``：``__getitem__`` 是否在 ``features`` 中包含相位通道

        写入模式内部状态（``_path`` … ``_finalized``）由 ``open_for_collection`` 设置，
        用户代码不应直接构造写入实例（请用 ``open_for_collection``）。
        """
        self.bs_pos = np.asarray(bs_pos, dtype=np.float64).reshape(-1)
        self.target_position = target_position
        self.target_velocity = target_velocity
        self.h_dd = h_dd
        self.collection_meta = collection_meta
        self.sensing_meta = sensing_meta
        self.use_phase = use_phase
        self._path: Path | None = None
        self._compression: str | None = None
        self._file: h5py.File | None = None
        self._h_dd_ds: h5py.Dataset | None = None
        self._pos_ds: h5py.Dataset | None = None
        self._vel_ds: h5py.Dataset | None = None
        self._writer_count = 0
        self._finalized = False

    @property
    def num_slots(self) -> int:
        """episode 条数：写入模式为 ``count``，读取模式为 ``h_dd.shape[0]``。"""
        if self._path is not None:
            return self._writer_count
        return int(self.h_dd.shape[0])

    @property
    def count(self) -> int:
        """采集写入模式下已追加的 episode 数。"""
        return self._writer_count

    @property
    def path(self) -> Path | None:
        """采集写入模式下的 HDF5 输出路径。"""
        return self._path

    def _ensure_readable(self) -> None:
        """写入模式且未 ``finalize`` 时禁止 ``__getitem__`` / ``spectrum_tensor``。"""
        if self._path is not None and not self._finalized:
            raise RuntimeError(
                "RTDataset 处于采集写入模式，请先 finalize 再 load 或索引"
            )

    def _require_sensing_meta(self) -> SensingMetadata:
        """返回 ``sensing_meta``；缺失时 ``ValueError``。"""
        if self.sensing_meta is None:
            raise ValueError("RTDataset 缺少 sensing_meta")
        return self.sensing_meta

    @property
    def range_resolution(self) -> float:
        """距离分辨率 (m)，来自 ``sensing_meta``。"""
        return self._require_sensing_meta().range_resolution

    @property
    def velocity_resolution(self) -> float:
        """速度分辨率 (m/s)，来自 ``sensing_meta``。"""
        return self._require_sensing_meta().velocity_resolution

    @property
    def max_range_m(self) -> float:
        """DD 谱 ROI 最大距离 (m)，来自 ``sensing_meta``。"""
        return self._require_sensing_meta().max_range_m

    @property
    def max_velocity_mps(self) -> float:
        """DD 谱 ROI 最大多普勒速度 (m/s)，来自 ``sensing_meta``。"""
        return self._require_sensing_meta().max_velocity_mps

    def _validate_slot_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_slots:
            raise IndexError(f"index {idx} out of range for {self.num_slots} slots")

    def __len__(self) -> int:
        return self.num_slots

    def spectrum_tensor(
        self, idx: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """返回单条 episode 的复数 h_dd 张量。"""
        self._ensure_readable()
        self._validate_slot_index(idx)
        t = torch.from_numpy(self.h_dd[idx])
        if device is not None:
            t = t.to(device)
        return t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """CNN 训练样本 dict。

        返回键
        ------
        - ``features``：``(C, H, W)`` float32；C=1（幅度 dB）或 2（+相位），由 ``use_phase`` 决定
        - ``range_m`` / ``velocity_mps``：单基地几何标签 (float32 标量)
        - ``slot``：episode 索引 (int64)
        """
        self._ensure_readable()
        self._validate_slot_index(idx)
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
        if self._path is not None:
            return f"RTDataset(writer, count={self._writer_count}, path={self._path})"
        return f"RTDataset(num_slots={self.num_slots}, h_dd_shape={self.h_dd.shape})"

    @classmethod
    def open_for_collection(
        cls,
        path: str | Path,
        bs_pos: np.ndarray,
        *,
        compression: str | None = "lzf",
    ) -> RTDataset:
        """打开 HDF5 文件，准备按 episode 流式写入。

        ``compression`` 可选 ``lzf``（默认）、``gzip`` 或 ``none``。
        """
        comp = None if compression in (None, "none") else compression
        ds = cls(
            bs_pos=np.asarray(bs_pos, dtype=np.float64).reshape(-1),
            target_position=np.empty((0, 3), dtype=np.float64),
            target_velocity=np.empty((0, 3), dtype=np.float64),
            h_dd=np.empty((0,), dtype=np.complex64),
        )
        ds._path = Path(path)
        ds._compression = comp
        return ds

    def __enter__(self) -> RTDataset:
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
        if self._path is None:
            raise RuntimeError("append_episode 仅用于 open_for_collection 返回的实例")
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
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset(_DATASET_KEY_BS_POS, data=self.bs_pos)

        h_dd_kwargs: dict[str, Any] = {
            "maxshape": (None,) + tuple(h_dd.shape),
            "chunks": (1,) + tuple(h_dd.shape),
        }
        if self._compression:
            h_dd_kwargs["compression"] = self._compression

        self._h_dd_ds = self._file.create_dataset(
            _DATASET_KEY_H_DD,
            shape=(0,) + tuple(h_dd.shape),
            dtype=h_dd.dtype,
            **h_dd_kwargs,
        )
        self._pos_ds = _create_vec3_dataset(self._file, _DATASET_KEY_TARGET_POSITION)
        self._vel_ds = _create_vec3_dataset(self._file, _DATASET_KEY_TARGET_VELOCITY)

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
        sensing_meta: SensingMetadata,
    ) -> None:
        """写入根属性并关闭 HDF5 文件；无 episode 时 ``ValueError``。"""
        if self._file is None:
            raise ValueError("RTDataset 无 episode 数据")
        if self._finalized:
            return
        _write_hdf5_root_metadata(
            self._file,
            num_slots=self._writer_count,
            collection_meta=collection_meta,
            sensing_meta=sensing_meta,
            scene_slug=scene_slug,
        )
        self._file.close()
        self._file = None
        self._finalized = True

    @classmethod
    def load(cls, filepath: str | Path, *, use_phase: bool = True) -> RTDataset:
        """从 HDF5 加载至内存。

        旧 CFR 格式（含 ``channel_frequency_response``）会抛出 ``ValueError`` 并提示重新采集。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            arrays = _read_array_datasets(f)
            return cls(
                bs_pos=arrays["bs_pos"],
                target_position=arrays["target_position"],
                target_velocity=arrays["target_velocity"],
                h_dd=arrays["h_dd"],
                collection_meta=CollectionMetadata.read_hdf5_attrs(f),
                sensing_meta=SensingMetadata.read_hdf5_attrs(f),
                use_phase=use_phase,
            )


# --- 采集产物写出 ---


def _save_collection_config(
    *,
    config_file: str | Path,
    output_root: Path,
) -> Path:
    """复制采集 TOML 配置到输出目录（保留原文件名）。"""
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
    """写出 ``{scene_slug}_mc_dataset_episodes.csv``。"""
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
    """渲染 RT 场景并保存 ``{scene_slug}_scene.png``。"""
    filename = f"{scene_slug}{_SCENE_PNG_SUFFIX}"
    return rt_simulator.render_to_file(filename, output_dir=output_root)


def save_collection_artifacts(
    *,
    scene_slug: str,
    config_file: str | Path,
    buffers: EpisodeBuffers,
    rt_simulator: RTSimulator,
    out_dir: Path | None = None,
) -> None:
    """写出采集产物：TOML、CSV、场景 PNG。HDF5 由采集循环内 ``RTDataset.finalize`` 完成。"""
    target_dir = _resolve_out_dir(out_dir)
    _save_collection_config(config_file=config_file, output_root=target_dir)
    _save_episodes_csv(
        scene_slug=scene_slug,
        rows=buffers.csv_rows,
        output_root=target_dir,
    )
    _save_scene_render(rt_simulator, scene_slug, target_dir)
    print(f"采集产物已保存至: {target_dir}")
