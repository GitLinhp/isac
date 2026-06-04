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
_DATASET_KEY_TARGET_VELOCITY = "target_velocity"
# 新文件写入上述键名；Dataset.load 仍兼容旧键 uav_position / uav_velocity。
_DATASET_KEY_BS_POS = "bs_pos"

_META_KEY_CARRIER_FREQUENCY = "carrier_frequency"
_META_KEY_SUBCARRIER_SPACING = "subcarrier_spacing"
_META_KEY_NUM_SUBCARRIERS = "num_subcarriers"
_META_KEY_NUM_SLOTS = "num_slots"
_META_KEY_DESCRIPTION = "description"


def _require_array_dataset(f: h5py.File, key: str) -> h5py.Dataset:
    if key not in f:
        raise KeyError(
            f"HDF5 缺少必选数据集 {key!r}。当前格式要求同时包含 CFR 与 CIR "
            "（channel_frequency_response、channel_impulse_response_a、"
            "channel_impulse_response_tau），请重新生成数据集。"
        )
    return f[key]


def _require_dataset_any(f: h5py.File, keys: tuple[str, ...]) -> h5py.Dataset:
    """返回 ``keys`` 中首个存在于文件内的数据集，否则报错。"""
    for k in keys:
        if k in f:
            return f[k]
    raise KeyError(f"HDF5 缺少以下任一数据集: {keys!r}")


def _read_meta(f: h5py.File, key: str) -> Any:
    """读取元数据：优先 attrs，兼容旧版 dataset 字段。"""
    if key in f.attrs:
        return f.attrs[key]
    if key in f:
        return f[key][()]
    raise KeyError(f"HDF5 中未找到元数据字段: {key}")


def _array_info(name: str, arr: np.ndarray) -> str:
    return f"  {name}: shape={arr.shape}, dtype={arr.dtype}"


@dataclass
class Dataset:
    """ISAC HDF5 数据集的内存表示（必含 CFR + CIR）。

    蒙特卡洛等场景下射线数目可能随样本变化；``cir_a`` / ``cir_tau`` 在路径维上可能对较短样本零填充。

    - 构造采集结果：``Dataset.from_export_arrays(...)``
    - 落盘：``dataset.save(path)``
    - 读取：``Dataset.load(path)``
    - 摘要：``dataset.show()``；CFR 预览：``dataset.plot_cfr(slot=...)``
    """

    cfr: np.ndarray
    cir_a: np.ndarray
    cir_tau: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    bs_pos: np.ndarray
    carrier_frequency: float
    subcarrier_spacing: float
    num_subcarriers: int
    num_slots: int
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"Dataset(num_slots={self.num_slots}, cfr_shape={self.cfr.shape}, "
            f"fc={self.carrier_frequency:.3e} Hz)"
        )

    @classmethod
    def from_export_arrays(
        cls,
        dataset_cfr: np.ndarray,
        dataset_cir_a: np.ndarray,
        dataset_cir_tau: np.ndarray,
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
    ) -> Dataset:
        """由 ndarray 组装数据集（与采集脚本导出字段对齐）。

        ``scene_name`` 仅用于在未显式给出 ``description`` 时生成默认英文描述；不单独写入 HDF5。
        ``num_valid`` 映射为属性 ``num_slots``（有效 episode / 步数）。
        """
        desc = description or (
            f"Sionna generated ISAC dataset with car trajectory in {scene_name}"
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
        )

    @classmethod
    def load(cls, filepath: str | Path) -> Dataset:
        """从 HDF5 加载（必须包含 CIR 与 CFR；不兼容旧版仅 CFR 文件）。

        目标位置/速度优先读取 ``target_position`` / ``target_velocity``，否则回退 ``uav_*`` 旧键。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, "r") as f:
            return cls(
                cfr=_require_array_dataset(f, _DATASET_KEY_CFR)[:],
                cir_a=_require_array_dataset(f, _DATASET_KEY_CIR_A)[:],
                cir_tau=_require_array_dataset(f, _DATASET_KEY_CIR_TAU)[:],
                target_position=_require_dataset_any(
                    f, (_DATASET_KEY_TARGET_POSITION, "uav_position")
                )[:],
                target_velocity=_require_dataset_any(
                    f, (_DATASET_KEY_TARGET_VELOCITY, "uav_velocity")
                )[:],
                bs_pos=_require_array_dataset(f, _DATASET_KEY_BS_POS)[:],
                carrier_frequency=float(_read_meta(f, _META_KEY_CARRIER_FREQUENCY)),
                subcarrier_spacing=float(_read_meta(f, _META_KEY_SUBCARRIER_SPACING)),
                num_subcarriers=int(_read_meta(f, _META_KEY_NUM_SUBCARRIERS)),
                num_slots=int(_read_meta(f, _META_KEY_NUM_SLOTS)),
                description=str(_read_meta(f, _META_KEY_DESCRIPTION))
                if (_META_KEY_DESCRIPTION in f.attrs or _META_KEY_DESCRIPTION in f)
                else "",
            )

    def save(self, filepath: str | Path) -> None:
        """写入 HDF5；成功后打印保存路径。"""
        path = Path(filepath)
        with h5py.File(path, "w") as f:
            f.create_dataset(_DATASET_KEY_CFR, data=self.cfr, compression="gzip")
            f.create_dataset(_DATASET_KEY_CIR_A, data=self.cir_a, compression="gzip")
            f.create_dataset(_DATASET_KEY_CIR_TAU, data=self.cir_tau, compression="gzip")
            f.create_dataset(_DATASET_KEY_TARGET_POSITION, data=self.target_position)
            f.create_dataset(_DATASET_KEY_TARGET_VELOCITY, data=self.target_velocity)
            f.create_dataset(_DATASET_KEY_BS_POS, data=self.bs_pos)

            f.attrs[_META_KEY_CARRIER_FREQUENCY] = self.carrier_frequency
            f.attrs[_META_KEY_SUBCARRIER_SPACING] = self.subcarrier_spacing
            f.attrs[_META_KEY_NUM_SUBCARRIERS] = self.num_subcarriers
            f.attrs[_META_KEY_NUM_SLOTS] = self.num_slots
            f.attrs[_META_KEY_DESCRIPTION] = self.description

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
        print(_array_info("cir_a", self.cir_a))
        print(_array_info("cir_tau", self.cir_tau))
        print(_array_info("target_position", self.target_position))
        print(_array_info("target_velocity", self.target_velocity))
        print(_array_info("bs_pos", self.bs_pos))

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
            raise ValueError(f"cfr[slot] 维度 {plane.ndim} 不支持快速绘图，请先 reshape")

        plt.tight_layout()
        plt.show()
