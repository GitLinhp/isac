"""ISAC 仿真采集结果的 HDF5 数据集读写与元数据封装。

采集期：``RTDatasetWriter.open`` 按 episode 流式写入 HDF5。
消费侧：``RTDataset.load`` 加载内存；``__getitem__`` 返回原始谱与运动学 dict。
采集期完成 CFR→h_dd 感知链；HDF5 仅存 ROI 裁剪后的复数 DD 谱与运动学。
``RTDataset`` 继承 ``torch.utils.data.Dataset``（只读）；写入由 ``RTDatasetWriter`` 负责。

数据流
------
::

    run_data_collection
        → RTDatasetWriter.append_episode → HDF5
        → RTDatasetWriter.finalize（根 attrs）
        → save_collection_artifacts（TOML / CSV / PNG）
    RTDataset.load
        → DataLoader + 训练脚本预处理（特征 / bin 标签）
        → spectrum_tensor（逐条评估）

职责边界
--------
- ``CollectionMetadata``：定义于 ``collection_metadata.py``，序列化采集可复现配置
- 布局常量（dataset 键名、CSV 列、文件后缀）：``h5_layout.py``
- 特征提取与标签生成：``isac.models``，在训练脚本中调用，不在本模块完成

设计说明
--------
``RTDataset.__getitem__``  deliberately 返回原始复数 ``spectrum_tensor`` 与运动学，
避免 Dataset 层绑定 ``SensingPerformance``；与 ``tests/test_dataset_indexing.py`` 契约一致。

HDF5 文件布局
--------------
根 datasets:

- ``bs_pos``：参考发射机位置，shape ``(3,)``（采集脚本取 ``bs1``）
- ``target_position`` / ``target_velocity``：目标运动学，shape ``(N, 3)``（m / m/s）
- ``delay_doppler_spectrum``：复数 h_dd，shape ``(N, H, W)``（ROI 裁剪后）

根 attrs:

- ``description``
- ``seed``, ``roi``, ``roi_z``, ``position_sampling_mode``, ``speed_range``, ``speed_sampling_mode``、
  ``num_samples``, ``sampler_pool_factor``：由 ``CollectionMetadata`` 写入

感知 ROI 与分辨率见同目录 TOML 副本（``[dd_spectrum_roi]``、``[ofdm]``），不写入 HDF5。

采集落盘产物（``data/``）:

- TOML：采集配置副本（保留原文件名）
- ``{scene_slug}_mc_dataset_episodes.csv``
- ``{scene_slug}_mc_sionna_dataset.h5``
- ``{scene_slug}_scene.png``
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from isac.utils.config_loader import resolve_config_path

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator

from .collection_metadata import CollectionMetadata
from .h5_layout import (
    ARRAY_DATASET_SPECS,
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

# --- HDF5 读写辅助 ---
def _require_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """返回指定名称的 HDF5 dataset。

    Raises
    ------
    KeyError
        文件中不存在 ``key`` 对应的数据集。
    """
    if key not in f:
        raise KeyError(f"HDF5 缺少数据集 {key!r}。")
    return cast(h5py.Dataset, f[key])


def _read_array_datasets(f: h5py.File) -> dict[str, np.ndarray]:
    """按 ``ARRAY_DATASET_SPECS`` 映射读取全部数组型 datasets。

    返回 dict 键名为 Python 属性名（``bs_pos``、``h_dd`` 等），
    与 HDF5 内键名（``delay_doppler_spectrum`` 等）的对应关系见 ``h5_layout.py``。
    """
    return {
        attr: _require_dataset(f, h5_key)[:] for h5_key, attr in ARRAY_DATASET_SPECS
    }


def _create_vec3_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """创建可扩展的 ``(N, 3)`` float64 运动学数据集。

    ``maxshape=(None, 3)`` 支持按 episode 追加行；``chunks=(1024, 3)``
    减少频繁 ``resize`` 时的 I/O 开销。
    """
    return f.create_dataset(
        key,
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float64,
        chunks=(1024, 3),
    )


@dataclass
class RTDatasetWriter:
    """HDF5 采集流式写入器。

    生命周期
    --------
    ``open(path, bs_pos)``
        → ``append_episode(h_dd, pos, vel)``（可重复）
        → ``finalize(collection_meta, scene_slug)``

    支持 ``with`` 上下文管理；异常退出且未 ``finalize`` 时，``__exit__`` 仍会关闭 HDF5 文件。
    ``finalize`` 幂等：已 finalize 后直接返回。无 episode 时 ``finalize`` 抛出 ``ValueError``。

    字段
    ----
    - ``path``：HDF5 输出路径
    - ``bs_pos``：参考基站位置 ``(3,)``，首次打开时写入静态 dataset
    - ``compression``：``h_dd`` 压缩算法（``lzf`` / ``gzip`` / ``None``）
    - ``_file`` / ``_h_dd_ds`` / ``_kinematics_datasets``：延迟创建的 HDF5 句柄与 datasets
    - ``_writer_count``：已追加 episode 数
    - ``_finalized``：是否已写入根属性并关闭文件
    """

    path: Path
    bs_pos: np.ndarray
    compression: str | None = "lzf"
    _file: h5py.File | None = field(default=None, repr=False)
    _h_dd_ds: h5py.Dataset | None = field(default=None, repr=False)
    _kinematics_datasets: tuple[h5py.Dataset, h5py.Dataset] | None = field(
        default=None, repr=False
    )
    _writer_count: int = 0
    _finalized: bool = False

    def __post_init__(self) -> None:
        """规范化 ``bs_pos`` 为一维 float64；``compression="none"`` 视为无压缩。"""
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

        Parameters
        ----------
        path
            HDF5 输出路径。
        bs_pos
            参考基站位置 ``(3,)``。
        compression
            ``h_dd`` 压缩算法：``lzf``（默认）、``gzip`` 或 ``none``（无压缩）。
        """
        return cls(path=path, bs_pos=bs_pos, compression=compression)

    def __len__(self) -> int:
        """已写入的 episode 条数。"""
        return self._writer_count

    def __repr__(self) -> str:
        """调试表示：已写入条数与输出路径。"""
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
        首次调用时创建 HDF5 可扩展 datasets；后续 ``h_dd`` 的 ``H, W`` 须与首条一致。
        """
        h_dd_arr = np.asarray(h_dd, dtype=np.complex64)
        pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
        self._ensure_open(h_dd_arr)
        idx = self._writer_count
        self._resize_writer(idx + 1)
        assert self._h_dd_ds is not None
        assert self._kinematics_datasets is not None
        self._h_dd_ds[idx] = h_dd_arr
        # 运动学 datasets 与 h_dd 共享行索引 idx
        for ds, row in zip(self._kinematics_datasets, (pos_row, vel_row)):
            ds[idx] = row
        self._writer_count += 1

    def _ensure_open(self, h_dd: np.ndarray) -> None:
        """首次 ``append_episode`` 时创建 HDF5 文件与可扩展 datasets。

        写入静态 ``bs_pos`` dataset，并创建 ``h_dd``（chunks ``(1, H, W)``）
        与两条 ``(N, 3)`` 运动学 datasets。
        """
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.create_dataset(DATASET_KEY_BS_POS, data=self.bs_pos)

        # 按 episode 分块，利于流式追加与压缩
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
        self._kinematics_datasets = (
            _create_vec3_dataset(self._file, DATASET_KEY_TARGET_POSITION),
            _create_vec3_dataset(self._file, DATASET_KEY_TARGET_VELOCITY),
        )

    def _resize_writer(self, new_count: int) -> None:
        """将 h_dd / 运动学 datasets 第一维扩展至 ``new_count``，不改变 ``H, W``。"""
        assert self._h_dd_ds is not None
        assert self._kinematics_datasets is not None
        self._h_dd_ds.resize((new_count,) + self._h_dd_ds.shape[1:])
        for ds in self._kinematics_datasets:
            ds.resize((new_count, 3))

    def finalize(
        self,
        *,
        collection_meta: CollectionMetadata,
        scene_slug: str,
    ) -> None:
        """写入根属性并关闭 HDF5 文件。

        Parameters
        ----------
        collection_meta
            采集可复现配置，序列化至根 attrs。
        scene_slug
            场景标识，用于生成 ``description`` 文本。

        Raises
        ------
        ValueError
            从未调用 ``append_episode``（无 episode 数据）。
        """
        if self._file is None:
            raise ValueError("RTDatasetWriter 无 episode 数据")
        if self._finalized:
            return
        self._file.attrs[META_KEY_DESCRIPTION] = collection_dataset_description(
            scene_slug, self._writer_count
        )
        collection_meta.write_hdf5_attrs(self._file)
        self._file.close()
        self._file = None
        self._finalized = True


@dataclass
class RTDataset(Dataset):
    """ISAC HDF5 只读数据集（内存驻留），供训练与评估。

    字段
    ----
    - ``bs_pos``：``(3,)`` float64，全数据集共享的参考基站位置
    - ``target_position`` / ``target_velocity``：``(N, 3)`` float64 目标运动学（m / m/s）
    - ``h_dd``：``(N, H, W)`` complex64，ROI 裁剪后的时延–多普勒谱
    - ``collection_meta``：从 HDF5 根 attrs 反序列化的采集配置

    读取 / 训练
    -----------
    - ``RTDataset.load(path)``
    - ``len(dataset)`` → ``N = h_dd.shape[0]``
    - ``dataset[i]`` → 原始 ``spectrum_tensor``、运动学与 ``bs_pos``
    - ``spectrum_tensor(i)`` → 单条复数 h_dd（评估用）

    采集写入请使用 ``RTDatasetWriter.open``。
    """

    bs_pos: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    h_dd: np.ndarray
    collection_meta: CollectionMetadata

    def __post_init__(self) -> None:
        """将 ``bs_pos`` 规范化为 ``(3,)`` float64。"""
        self.bs_pos = np.asarray(self.bs_pos, dtype=np.float64).reshape(-1)

    def _validate_slot_index(self, idx: int) -> None:
        """校验 episode 索引；不支持负索引，越界时 ``IndexError``。"""
        n = len(self)
        if idx < 0 or idx >= n:
            raise IndexError(f"index {idx} out of range for {n} slots")

    def __len__(self) -> int:
        """episode 条数 ``N = h_dd.shape[0]``。"""
        return int(self.h_dd.shape[0])

    def spectrum_tensor(
        self, idx: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """返回单条 episode 的复数 h_dd 张量。

        Parameters
        ----------
        idx
            episode 索引，须满足 ``0 <= idx < len(self)``。
        device
            可选，将张量移至指定设备。

        Returns
        -------
        torch.Tensor
            shape ``(H, W)``，dtype complex64。
        """
        self._validate_slot_index(idx)
        t = torch.from_numpy(self.h_dd[idx])
        if device is not None:
            t = t.to(device)
        return t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """原始训练/评估样本 dict。

        返回键
        ------
        - ``spectrum_tensor``：``(H, W)`` complex64 ROI 裁切谱
        - ``target_position`` / ``target_velocity``：``(3,)`` float32，逐样本运动学
        - ``bs_pos``：``(3,)`` float32，全数据集共享（非 per-sample）
        - ``slot``：episode 索引 (int64)

        特征提取与标签生成由 ``isac.models`` 在训练脚本中完成。
        """
        self._validate_slot_index(idx)
        return {
            "spectrum_tensor": torch.from_numpy(self.h_dd[idx]),
            "target_position": torch.from_numpy(self.target_position[idx]).float(),
            "target_velocity": torch.from_numpy(self.target_velocity[idx]).float(),
            "bs_pos": torch.from_numpy(self.bs_pos).float(),
            "slot": torch.tensor(idx, dtype=torch.int64),
        }

    def __repr__(self) -> str:
        """调试表示：样本数与 ``h_dd`` 形状。"""
        return f"RTDataset(n={len(self)}, h_dd_shape={self.h_dd.shape})"

    @classmethod
    def load(cls, filepath: str | Path) -> RTDataset:
        """从 HDF5 一次性读入全部 arrays 至内存。

        Parameters
        ----------
        filepath
            HDF5 数据集路径。

        Returns
        -------
        RTDataset
            内存驻留的只读数据集实例。
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
            )


# --- 采集产物 ---
def save_collection_artifacts(
    *,
    scene_slug: str,
    config_file: str | Path,
    csv_rows: list[dict[str, str | int]],
    rt_simulator: RTSimulator,
    out_dir: Path,
) -> None:
    """写出采集辅助产物：TOML 配置副本、episode CSV、场景 PNG。

    HDF5 不在此函数内写出，由采集循环内 ``RTDatasetWriter.finalize`` 完成。
    须在 ``finalize`` 关闭 HDF5 之后调用（见 ``run_data_collection.py``）。

    Parameters
    ----------
    scene_slug
        场景标识，用于 CSV / PNG 文件名前缀。
    config_file
        采集 TOML 配置路径，复制至 ``out_dir`` 并保留原文件名。
    csv_rows
        episode 元数据行；列名来自 ``EPISODE_CSV_COLUMNS``。为空时跳过 CSV。
    rt_simulator
        RT 仿真器，用于渲染场景 PNG。
    out_dir
        输出目录（通常为 ``data/``）。
    """
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
