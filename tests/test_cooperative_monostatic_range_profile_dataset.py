"""CooperativeMonostaticRangeProfileDataset 与 session 划分测试。"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    CooperativeMonostaticRangeProfileDataset,
    session_train_val_split,
)

_VLEN = 32768


def _write_synthetic_h5(path: Path, *, n_sessions: int = 4, frames_per_session: int = 3) -> None:
    total = n_sessions * frames_per_session
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV0,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV1,
            data=rng.standard_normal((total, _VLEN)).astype(np.complex64),
        )
        target = np.zeros((total, 3), dtype=np.float64)
        for s in range(n_sessions):
            start = s * frames_per_session
            end = start + frames_per_session
            target[start:end, 0] = s * 0.1
            target[start:end, 1] = s * 0.2
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        session_index = np.repeat(
            np.arange(n_sessions, dtype=np.int32),
            frames_per_session,
        )
        f.create_dataset(DATASET_KEY_SESSION_INDEX, data=session_index)
        frame_index = np.tile(
            np.arange(frames_per_session, dtype=np.int32),
            n_sessions,
        )
        f.create_dataset(DATASET_KEY_FRAME_INDEX, data=frame_index)
        f.attrs["num_sessions"] = n_sessions
        f.attrs["frames_per_session"] = frames_per_session


def test_session_train_val_split_no_leak(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=5, frames_per_session=4)
    with h5py.File(h5_path, "r") as f:
        session_indices = f[DATASET_KEY_SESSION_INDEX][:]

    train_idx, val_idx = session_train_val_split(session_indices, 0.4, seed=7)
    train_sessions = set(session_indices[train_idx].tolist())
    val_sessions = set(session_indices[val_idx].tolist())
    assert train_sessions.isdisjoint(val_sessions)
    assert train_idx.size + val_idx.size == session_indices.size


def test_lazy_dataset_roi_transform(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=2, frames_per_session=2)
    frame_indices = np.arange(4, dtype=np.int64)
    ds = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        frame_indices,
        proc_params=grc_cooperative_processing_params(),
        range_roi=DEFAULT_RANGE_ROI,
        transform_on_load=True,
        feature_mode="complex_roi",
    )
    item = ds[0]
    assert item["dual_profiles"].shape[0] == 2
    assert item["dual_profiles"].shape[1] == 27
    assert item["dual_profiles"].dtype == torch.complex64
    assert item["target_xy"].shape == (2,)
    assert item["session_index"].dtype == torch.int64
    ds.close()


def test_lazy_dataset_no_transform(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=1, frames_per_session=1)
    ds = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        np.array([0], dtype=np.int64),
        transform_on_load=False,
    )
    item = ds[0]
    assert item["dual_profiles"].shape == (2, _VLEN)
    ds.close()


def test_label_jitter_disabled_matches_h5(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=1, frames_per_session=1)
    with h5py.File(h5_path, "r") as f:
        true_xy = f[DATASET_KEY_TARGET_POSITION][0, :2]

    ds = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        np.array([0], dtype=np.int64),
        transform_on_load=False,
        label_jitter_m=0.0,
    )
    for _ in range(5):
        item = ds[0]
        np.testing.assert_allclose(item["target_xy"].numpy(), true_xy, rtol=0.0, atol=1e-6)
    ds.close()


def test_label_jitter_within_bounds(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=1, frames_per_session=1)
    with h5py.File(h5_path, "r") as f:
        true_xy = f[DATASET_KEY_TARGET_POSITION][0, :2]

    ds = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        np.array([0], dtype=np.int64),
        transform_on_load=False,
        label_jitter_m=0.02,
    )
    for _ in range(20):
        jittered = ds[0]["target_xy"].numpy()
        delta = jittered - true_xy
        assert np.all(np.abs(delta) <= 0.02 + 1e-6)
    ds.close()


def test_label_jitter_changes_between_reads(tmp_path: Path):
    h5_path = tmp_path / "coop.h5"
    _write_synthetic_h5(h5_path, n_sessions=1, frames_per_session=1)
    ds = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        np.array([0], dtype=np.int64),
        transform_on_load=False,
        label_jitter_m=0.02,
    )
    samples = {tuple(ds[0]["target_xy"].tolist()) for _ in range(10)}
    assert len(samples) > 1
    ds.close()
