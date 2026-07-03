"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

由 ``run_data_collection.py`` 写入；训练侧通过 ``Dataset.load`` 消费。
``Dataset`` 支持 CIFAR10 风格序列访问：``cfr, label = dataset[i]``，
或 ``for cfr, label in dataset: ...``（可再包装为 ``torch.utils.data.Dataset``）。

HDF5 文件布局
--------------
根 datasets:

- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``channel_frequency_response``：复数 CFR，典型 shape ``(N, ..., S, F)``
  （``N``=episode 数，``S``=OFDM 符号，``F``=子载波）

根 attrs:

- ``num_slots``, ``description``
- ``collection_*``：由 ``CollectionMetadata`` 写入，共 5 个字段
  （``seed``, ``roi``, ``position_sampling_mode``, ``speed_range``,
  ``speed_sampling_mode``）

采集落盘产物（``out/dataset_collection/``）:

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

from isac import PROJECT_ROOT
from isac.utils.config_loader import resolve_config_path

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator

DatasetLabel = list[tuple[float, float, float]]
"""单样本标签：``[position_xyz, velocity_xyz]``。"""
DatasetSample = tuple[np.ndarray, DatasetLabel]

# --- 路径与键名常量 ---

DEFAULT_COLLECTION_OUT_DIR = PROJECT_ROOT / "out" / "dataset_collection"
"""``run_data_collection.py`` 默认输出目录。"""

_EPISODE_CSV_SUFFIX = "_mc_dataset_episodes.csv"
_H5_SUFFIX = "_mc_sionna_dataset.h5"
_SCENE_PNG_SUFFIX = "_scene.png"

_EPISODE_CSV_COLUMNS = (
    "sample_idx",
    "position",
    "velocity",
    "true_range_m",
    "true_radial_velocity_mps",
)

_DATASET_KEY_CFR = "channel_frequency_response"
_DATASET_KEY_TARGET_POSITION = "target_position"
_DATASET_KEY_TARGET_VELOCITY = "target_velocity"
_DATASET_KEY_BS_POS = "bs_pos"

_META_KEY_NUM_SLOTS = "num_slots"
_META_KEY_DESCRIPTION = "description"
_META_PREFIX_COLLECTION = "collection_"

_COLLECTION_TUPLE_FIELDS = frozenset({"roi", "speed_range"})

_ARRAY_DATASET_SPECS: tuple[tuple[str, str, bool], ...] = (
    (_DATASET_KEY_BS_POS, "bs_pos", False),
    (_DATASET_KEY_TARGET_POSITION, "target_position", False),
    (_DATASET_KEY_TARGET_VELOCITY, "target_velocity", False),
    (_DATASET_KEY_CFR, "cfr", True),
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


def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """返回指定名称的数据集；缺失时抛出 ``KeyError``。"""
    if key not in f:
        if key == _DATASET_KEY_CFR:
            raise KeyError(
                f"HDF5 缺少必选数据集 {key!r}（channel_frequency_response）。"
            )
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


def _read_array_datasets(f: h5py.File) -> dict[str, np.ndarray]:
    """读取 CFR、运动学与 ``bs_pos`` ndarray。"""
    return {
        attr: _require_dataset(f, h5_key)[:]
        for h5_key, attr, _ in _ARRAY_DATASET_SPECS
    }


def _write_array_datasets(f: h5py.File, ds: Dataset) -> None:
    """写入四个 ndarray 数据集（CFR 使用 gzip）。"""
    for h5_key, attr, gzip in _ARRAY_DATASET_SPECS:
        kwargs = {"compression": "gzip"} if gzip else {}
        f.create_dataset(h5_key, data=getattr(ds, attr), **kwargs)


def _write_root_attrs(
    f: h5py.File, ds: Dataset, *, scene_slug: str | None = None
) -> None:
    """写入 ``num_slots``、``description`` 与 ``collection_*`` 根属性。"""
    f.attrs[_META_KEY_NUM_SLOTS] = ds.num_slots
    if ds.collection_meta is not None:
        if scene_slug is not None:
            f.attrs[_META_KEY_DESCRIPTION] = collection_dataset_description(
                scene_slug, ds.num_slots
            )
        ds.collection_meta.write_hdf5_attrs(f)


# --- 数据类 ---


@dataclass
class EpisodeBuffers:
    """主循环共享的 episode 级写出缓冲。

    采集循环中由 ``process_episode`` 逐条追加，循环结束后经
    ``save_collection_artifacts`` 落盘。

    Attributes
    ----------
    - h_freq_list : list[np.ndarray]
        逐 episode 的 CFR numpy 数组。
    - target_pos_list : list[np.ndarray]
        逐 episode 目标位置，每条 shape ``(3,)`` (m)。
    - target_vel_list : list[np.ndarray]
        逐 episode 目标速度，每条 shape ``(3,)`` (m/s)。
    - csv_rows : list[dict[str, str | int]]
        逐 episode CSV 行（运动学 + 几何真值列）。
    """

    h_freq_list: list[np.ndarray] = field(default_factory=list)
    target_pos_list: list[np.ndarray] = field(default_factory=list)
    target_vel_list: list[np.ndarray] = field(default_factory=list)
    csv_rows: list[dict[str, str | int]] = field(default_factory=list)


def _collection_attr_key(name: str) -> str:
    """``CollectionMetadata`` 字段对应的 HDF5 根属性键名。"""
    return f"{_META_PREFIX_COLLECTION}{name}"


def _hdf5_serialize(val: Any) -> Any:
    """写入 HDF5 attrs 前将 tuple 转为 list。"""
    return list(val) if isinstance(val, tuple) else val


def _hdf5_deserialize_collection(name: str, val: Any) -> Any:
    """从 HDF5 attrs 还原 ``CollectionMetadata`` 字段值。"""
    if name in _COLLECTION_TUPLE_FIELDS:
        return tuple(float(x) for x in val)
    return val


@dataclass(frozen=True)
class CollectionMetadata:
    """一次采集运行的可复现配置摘要，序列化到 HDF5 根属性 ``collection_<field>``。

    共 5 个字段，对应 ``run_data_collection.py`` 蒙特卡洛平面 ROI 采集 CLI。
    episode 数由 ``Dataset.cfr`` 第一维推断；仿真配置以 TOML 副本单独落盘。

    Attributes
    ----------
    - seed : int
        随机种子（``--seed``）。
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
    roi: tuple[float, float, float, float]
    position_sampling_mode: str = "uniform"
    speed_range: tuple[float, float] = (0.0, 0.0)
    speed_sampling_mode: str = "uniform"

    def write_hdf5_attrs(self, f: h5py.File) -> None:
        """将全部字段写入 HDF5 根属性 ``collection_<field>``。"""
        for key, val in asdict(self).items():
            f.attrs[_collection_attr_key(key)] = _hdf5_serialize(val)

    @classmethod
    def read_hdf5_attrs(cls, f: h5py.File) -> CollectionMetadata | None:
        """从 HDF5 根属性读取采集元数据。

        无 ``collection_seed`` 时返回 ``None``。
        多余 ``collection_*`` 属性静默忽略；缺失字段使用 dataclass 默认值。
        """
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
        """从 ``run_data_collection.py`` CLI 参数构建元数据。

        依赖字段：``--seed``, ``--roi``, ``--position_sampling_mode``,
        ``--speed_sampling_mode``, ``--speed_range``。
        """
        return cls(
            seed=int(args.seed),
            roi=tuple(map(float, args.roi)),
            position_sampling_mode=str(args.position_sampling_mode),
            speed_range=tuple(map(float, args.speed_range)),
            speed_sampling_mode=str(args.speed_sampling_mode),
        )


@dataclass
class Dataset:
    """ISAC HDF5 数据集的内存表示（CFR + kinematics）。

    典型数组形状：

    - ``bs_pos``：``(3,)``
    - ``target_position`` / ``target_velocity``：``(num_slots, 3)``
    - ``cfr``：``(num_slots, ..., num_ofdm_symbols, num_subcarriers)``，复数

    ``num_slots`` 由 ``cfr`` 第一维（episode 数）推断。

    序列协议
    --------
    - ``len(dataset)`` → ``num_slots``
    - ``dataset[i]`` → ``(cfr_i, [(px, py, pz), (vx, vy, vz)])``
    - 仅支持非负整数索引；越界抛出 ``IndexError``

    Attributes
    ----------
    - bs_pos : np.ndarray
        参考发射机位置 (m)。
    - target_position : np.ndarray
        目标位置 (m)。
    - target_velocity : np.ndarray
        目标速度 (m/s)。
    - cfr : np.ndarray
        信道频率响应。
    - collection_meta : CollectionMetadata | None
        采集可复现配置；旧文件可能为 ``None``。
    """

    bs_pos: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    cfr: np.ndarray
    collection_meta: CollectionMetadata | None = None

    @property
    def num_slots(self) -> int:
        """有效 episode 数（与 ``cfr`` 第一维一致）。"""
        return int(self.cfr.shape[0])

    def __len__(self) -> int:
        return self.num_slots

    def __getitem__(self, idx: int) -> DatasetSample:
        if idx < 0 or idx >= self.num_slots:
            raise IndexError(f"index {idx} out of range for {self.num_slots} slots")
        pos = self.target_position[idx]
        vel = self.target_velocity[idx]
        label: DatasetLabel = [
            (float(pos[0]), float(pos[1]), float(pos[2])),
            (float(vel[0]), float(vel[1]), float(vel[2])),
        ]
        return self.cfr[idx], label

    def __repr__(self) -> str:
        return f"Dataset(num_slots={self.num_slots}, cfr_shape={self.cfr.shape})"

    @classmethod
    def from_buffers(
        cls,
        buffers: EpisodeBuffers,
        bs_pos: np.ndarray,
        *,
        collection_meta: CollectionMetadata | None = None,
    ) -> Dataset:
        """由 ``EpisodeBuffers`` 组装 ``Dataset``。

        ``buffers.h_freq_list`` 为空时抛出 ``ValueError``。
        """
        if not buffers.h_freq_list:
            raise ValueError("EpisodeBuffers 无 CFR 数据")
        return cls(
            bs_pos=bs_pos,
            target_position=np.array(buffers.target_pos_list),
            target_velocity=np.array(buffers.target_vel_list),
            cfr=np.array(buffers.h_freq_list),
            collection_meta=collection_meta,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> Dataset:
        """从 HDF5 加载数据集。

        采集元数据经 ``CollectionMetadata.read_hdf5_attrs`` 读取。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            arrays = _read_array_datasets(f)
            return cls(
                **arrays,
                collection_meta=CollectionMetadata.read_hdf5_attrs(f),
            )

    def save(
        self, filepath: str | Path, *, scene_slug: str | None = None
    ) -> None:
        """写入 HDF5：CFR 使用 gzip 压缩；根属性含 ``collection_*``。

        ``scene_slug`` 用于生成根属性 ``description``；未提供时不写入该字段。
        """
        path = Path(filepath)
        with h5py.File(path, "w") as f:
            _write_array_datasets(f, self)
            _write_root_attrs(f, self, scene_slug=scene_slug)


# --- 流式 HDF5 写入 ---


class Hdf5CollectionWriter:
    """采集期按 episode 流式写入 HDF5，避免内存堆叠与整表压缩。

    CFR 使用 ``chunks=(1, *episode_shape)`` 与可配置压缩（默认 ``lzf``）；
    采集结束后调用 ``finalize`` 写入根属性并关闭文件。
    """

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
        self._cfr_ds: h5py.Dataset | None = None
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
        cfr: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> None:
        """追加单条 episode 的 CFR 与运动学。"""
        cfr_arr = np.asarray(cfr)
        pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
        if self._file is None:
            self._open(cfr_arr)
        idx = self._count
        self._resize(idx + 1)
        assert self._cfr_ds is not None
        assert self._pos_ds is not None
        assert self._vel_ds is not None
        self._cfr_ds[idx] = cfr_arr
        self._pos_ds[idx] = pos_row
        self._vel_ds[idx] = vel_row
        self._count += 1

    def _open(self, cfr: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset(_DATASET_KEY_BS_POS, data=self._bs_pos)

        cfr_chunks = (1,) + tuple(cfr.shape)
        cfr_kwargs: dict[str, Any] = {
            "maxshape": (None,) + tuple(cfr.shape),
            "chunks": cfr_chunks,
        }
        if self._compression:
            cfr_kwargs["compression"] = self._compression

        self._cfr_ds = self._file.create_dataset(
            _DATASET_KEY_CFR,
            shape=(0,) + tuple(cfr.shape),
            dtype=cfr.dtype,
            **cfr_kwargs,
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
        assert self._cfr_ds is not None
        assert self._pos_ds is not None
        assert self._vel_ds is not None
        self._cfr_ds.resize((new_count,) + self._cfr_ds.shape[1:])
        self._pos_ds.resize((new_count, 3))
        self._vel_ds.resize((new_count, 3))

    def finalize(
        self,
        *,
        collection_meta: CollectionMetadata,
        scene_slug: str,
    ) -> None:
        """写入根属性并关闭 HDF5 文件。"""
        if self._file is None:
            raise ValueError("Hdf5CollectionWriter 无 episode 数据")
        if self._finalized:
            return
        self._file.attrs[_META_KEY_NUM_SLOTS] = self._count
        self._file.attrs[_META_KEY_DESCRIPTION] = collection_dataset_description(
            scene_slug, self._count
        )
        collection_meta.write_hdf5_attrs(self._file)
        self._file.close()
        self._file = None
        self._finalized = True


# --- 采集落盘 API ---


def _save_collection_config(
    *,
    config_file: str | Path,
    output_root: Path,
) -> Path:
    """将采集用 TOML 复制到输出目录，目标文件名为源文件 basename。"""
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
    """写入 Episode CSV（固定 9 列）。"""
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
    """渲染场景 PNG 至采集产物目录（``clip_at`` 由 TOML ``[rt_simulator.render]`` 提供）。"""
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
    out_dir: Path | None = None,
    h5_already_written: bool = False,
) -> None:
    """一次写出采集产物：TOML 配置副本、Episode CSV、HDF5 数据集与场景 PNG。"""
    collection_meta = CollectionMetadata.from_collection_args(args)
    target_dir = _resolve_out_dir(out_dir)
    _save_collection_config(config_file=config_file, output_root=target_dir)
    _save_episodes_csv(
        scene_slug=scene_slug,
        rows=buffers.csv_rows,
        output_root=target_dir,
    )
    if not h5_already_written and buffers.h_freq_list:
        Dataset.from_buffers(
            buffers, bs_pos, collection_meta=collection_meta
        ).save(collection_h5_path(scene_slug, target_dir), scene_slug=scene_slug)
    _save_scene_render(rt_simulator, scene_slug, target_dir)
    print(f"采集产物已保存至: {target_dir}")
