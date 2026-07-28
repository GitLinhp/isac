"""cooperative monostatic features sidecar H5 构建与 Dataset 测试。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from isac_imp.cooperative_monostatic_pipeline import DEFAULT_RANGE_ROI, grc_cooperative_processing_params
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    CooperativeMonostaticFeaturesDataset,
    CooperativeMonostaticRangeProfileDataset,
    build_cooperative_monostatic_features_h5,
    default_features_h5_path,
    is_cooperative_monostatic_features_h5,
    open_cooperative_monostatic_training_dataset,
    summarize_cooperative_monostatic_h5,
)

_VLEN = 32768


def _write_synthetic_raw_h5(path: Path, *, n_frames: int = 4) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV0,
            data=rng.standard_normal((n_frames, _VLEN)).astype(np.complex64),
        )
        f.create_dataset(
            DATASET_KEY_PROFILES_DEV1,
            data=rng.standard_normal((n_frames, _VLEN)).astype(np.complex64),
        )
        target = np.zeros((n_frames, 3), dtype=np.float64)
        target[:, 0] = np.linspace(0.0, 0.3, n_frames)
        target[:, 1] = np.linspace(0.0, 0.6, n_frames)
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        f.create_dataset(
            DATASET_KEY_SESSION_INDEX,
            data=np.arange(n_frames, dtype=np.int32),
        )
        f.create_dataset(
            DATASET_KEY_FRAME_INDEX,
            data=np.zeros(n_frames, dtype=np.int32),
        )
        f.attrs["num_sessions"] = n_frames
        f.attrs["frames_per_session"] = 1
        f.attrs["data_kind"] = "divide_cpi"


def test_default_features_h5_path() -> None:
    raw = Path("data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5")
    assert default_features_h5_path(raw).name == "cooperative_monostatic_dataset_features.h5"


def test_build_features_h5_shape_and_kind(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=3)

    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=DEFAULT_RANGE_ROI,
        show_progress=False,
    )

    assert is_cooperative_monostatic_features_h5(feat_path)
    summary = summarize_cooperative_monostatic_h5(feat_path)
    assert summary["data_kind"] == "cooperative_monostatic_features"
    assert summary["features_shape"][0] == 3
    assert summary["features_shape"][1] == 4
    assert summary["features_shape"][2] == summary["roi_len"]


def test_features_dataset_matches_online_transform(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=2)

    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        show_progress=False,
    )

    online_ds = CooperativeMonostaticRangeProfileDataset(
        raw_path,
        np.array([0], dtype=np.int64),
        proc_params=grc_cooperative_processing_params(),
        range_roi=DEFAULT_RANGE_ROI,
        transform_on_load=True,
        feature_mode="legacy_4ch",
    )
    offline_ds = CooperativeMonostaticFeaturesDataset(
        feat_path,
        np.array([0], dtype=np.int64),
    )

    online_item = online_ds[0]
    offline_item = offline_ds[0]
    torch.testing.assert_close(
        online_item["dual_profiles"],
        offline_item["dual_profiles"],
        atol=1e-5,
        rtol=1e-5,
    )
    online_ds.close()
    offline_ds.close()


def test_open_training_dataset_auto_detects_features(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=1)
    build_cooperative_monostatic_features_h5(raw_path, feat_path, show_progress=False)

    ds = open_cooperative_monostatic_training_dataset(
        feat_path,
        np.array([0], dtype=np.int64),
        feature_mode="legacy_4ch",
    )
    assert isinstance(ds, CooperativeMonostaticFeaturesDataset)
    item = ds[0]
    assert item["dual_profiles"].dtype == torch.float32
    assert item["dual_profiles"].shape[0] == 4
    ds.close()


def test_build_rejects_features_source(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=1)
    build_cooperative_monostatic_features_h5(raw_path, feat_path, show_progress=False)

    with pytest.raises(ValueError, match="already features"):
        build_cooperative_monostatic_features_h5(feat_path, tmp_path / "dup.h5", show_progress=False)
