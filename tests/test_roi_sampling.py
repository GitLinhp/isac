"""ROI 采样固定 z 与 CollectionMetadata roi_z 测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from isac.collection import CollectionMetadata, RoiKinematicsSampler
from isac.utils import set_random_seed


def _pop_all_z(sampler: RoiKinematicsSampler) -> list[float]:
    values: list[float] = []
    while len(sampler) > 0:
        pos, _, _ = sampler.pop()
        values.append(float(pos[2]))
    return values


def test_uniform_roi_z_fixed() -> None:
    set_random_seed(0)
    sampler = RoiKinematicsSampler(
        roi=[-2.5, 2.5, -4.5, 4.5],
        position_sampling_mode="uniform",
        speed_range=[0.1, 3.0],
        speed_sampling_mode="uniform",
        num_samples=20,
        roi_z=1.5,
    )
    z_values = _pop_all_z(sampler)
    assert len(z_values) == 20
    assert all(z == pytest.approx(1.5) for z in z_values)


def test_gaussian_roi_z_fixed() -> None:
    set_random_seed(1)
    sampler = RoiKinematicsSampler(
        roi=[-2.5, 2.5, -4.5, 4.5],
        position_sampling_mode="gaussian",
        speed_range=[0.1, 3.0],
        speed_sampling_mode="uniform",
        num_samples=20,
        roi_z=-0.5,
    )
    z_values = _pop_all_z(sampler)
    assert len(z_values) == 20
    assert all(z == pytest.approx(-0.5) for z in z_values)


def test_collection_metadata_hdf5_roundtrip_with_roi_z() -> None:
    meta = CollectionMetadata(
        seed=7,
        roi=(-2.5, 2.5, -4.5, 4.5),
        roi_z=1.2,
        position_sampling_mode="uniform",
        speed_range=(0.1, 3.0),
        speed_sampling_mode="uniform",
        num_samples=10,
        sampler_pool_factor=2,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "meta.h5"
        with h5py.File(path, "w") as f:
            meta.write_hdf5_attrs(f)
        with h5py.File(path, "r") as f:
            loaded = CollectionMetadata.read_hdf5_attrs(f)
    assert loaded.roi_z == pytest.approx(1.2)
    assert loaded.roi == meta.roi


def test_collection_metadata_read_hdf5_missing_roi_z_defaults_zero() -> None:
    attrs = {
        "seed": 7,
        "roi": [-2.5, 2.5, -4.5, 4.5],
        "position_sampling_mode": "uniform",
        "speed_range": [0.1, 3.0],
        "speed_sampling_mode": "uniform",
        "num_samples": 10,
        "sampler_pool_factor": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.h5"
        with h5py.File(path, "w") as f:
            for key, val in attrs.items():
                f.attrs[key] = val
        with h5py.File(path, "r") as f:
            loaded = CollectionMetadata.read_hdf5_attrs(f)
    assert loaded.roi_z == pytest.approx(0.0)
