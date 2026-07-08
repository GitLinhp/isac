"""一次采集运行的可复现配置摘要，序列化到 HDF5 根属性。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import Any

import h5py

from isac.utils import set_random_seed

from .h5_layout import COLLECTION_TUPLE_FIELDS
from .roi_sampling import RoiKinematicsSampler, SamplingMode

_VALID_SAMPLING_MODES = frozenset({"uniform", "gaussian"})


def _parse_sampling_mode(raw: Any, *, field: str) -> SamplingMode:
    mode = str(raw).strip().lower()
    if mode not in _VALID_SAMPLING_MODES:
        raise ValueError(
            f"{field} 仅支持 'uniform' 或 'gaussian'，收到 {raw!r}"
        )
    return mode  # type: ignore[return-value]


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
        return cls(
            **{
                fld.name: _hdf5_deserialize_collection(fld.name, f.attrs[fld.name])
                for fld in fields(cls)
            }
        )
