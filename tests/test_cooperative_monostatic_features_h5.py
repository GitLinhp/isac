"""cooperative monostatic features sidecar H5 构建与 Dataset 测试。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from isac_imp.cooperative_monostatic_pipeline import DEFAULT_RANGE_ROI, grc_cooperative_processing_params
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_ENERGY,
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    META_KEY_FEATURE_MODE,
    CooperativeMonostaticFeaturesDataset,
    CooperativeMonostaticRangeProfileDataset,
    build_cooperative_monostatic_features_h5,
    cooperative_frame_cpi_energy,
    default_features_h5_path,
    is_cooperative_monostatic_features_h5,
    load_cooperative_frame_energy,
    open_cooperative_monostatic_training_dataset,
    resolve_cooperative_features_h5,
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
    named = default_features_h5_path(
        raw, range_roi=(0.0, 4.5), feature_mode="real_imag"
    )
    assert named.name == "cooperative_monostatic_dataset_features_roi0_4.5_real_imag.h5"


def test_build_features_h5_shape_and_kind(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=3)

    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=DEFAULT_RANGE_ROI,
        feature_mode="legacy_4ch",
        show_progress=False,
    )

    assert is_cooperative_monostatic_features_h5(feat_path)
    summary = summarize_cooperative_monostatic_h5(feat_path)
    assert summary["data_kind"] == "cooperative_monostatic_features"
    assert summary["features_shape"][0] == 3
    assert summary["features_shape"][1] == 4
    assert summary["features_shape"][2] == summary["roi_len"]
    with h5py.File(feat_path, "r") as f:
        assert META_KEY_FEATURE_MODE in f.attrs
        assert str(f.attrs[META_KEY_FEATURE_MODE]) == "legacy_4ch"
        assert DATASET_KEY_FRAME_ENERGY in f
        assert f[DATASET_KEY_FRAME_ENERGY].shape == (3,)


def test_features_dataset_matches_online_transform(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=2)

    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        feature_mode="legacy_4ch",
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


def test_real_imag_features_match_online(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features_real_imag.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=2)
    range_roi = (0.0, 4.5)

    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=range_roi,
        feature_mode="real_imag",
        show_progress=False,
    )

    online_ds = CooperativeMonostaticRangeProfileDataset(
        raw_path,
        np.array([0], dtype=np.int64),
        proc_params=grc_cooperative_processing_params(),
        range_roi=range_roi,
        transform_on_load=True,
        feature_mode="real_imag",
    )
    offline_ds = open_cooperative_monostatic_training_dataset(
        feat_path,
        np.array([0], dtype=np.int64),
        range_roi=range_roi,
        feature_mode="real_imag",
    )
    assert isinstance(offline_ds, CooperativeMonostaticFeaturesDataset)
    torch.testing.assert_close(
        online_ds[0]["dual_profiles"],
        offline_ds[0]["dual_profiles"],
        atol=1e-5,
        rtol=1e-5,
    )
    online_ds.close()
    offline_ds.close()


def test_frame_energy_matches_raw_cpi(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "feat.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=3)
    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        feature_mode="real_imag",
        range_roi=(0.0, 4.0),
        show_progress=False,
    )
    with h5py.File(raw_path, "r") as f:
        expected = cooperative_frame_cpi_energy(
            f[DATASET_KEY_PROFILES_DEV0][:],
            f[DATASET_KEY_PROFILES_DEV1][:],
        )
    loaded = load_cooperative_frame_energy(feat_path)
    assert loaded is not None
    np.testing.assert_allclose(loaded, expected, rtol=1e-5, atol=1e-5)


def test_resolve_features_h5_auto_and_mismatch(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=1)
    range_roi = (0.0, 4.5)
    feat_path = default_features_h5_path(
        raw_path, range_roi=range_roi, feature_mode="real_imag"
    )
    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=range_roi,
        feature_mode="real_imag",
        show_progress=False,
    )

    resolved = resolve_cooperative_features_h5(
        raw_path,
        range_roi=range_roi,
        feature_mode="real_imag",
    )
    assert resolved == feat_path.resolve()

    with pytest.raises(ValueError, match="feature_mode"):
        resolve_cooperative_features_h5(
            feat_path,
            range_roi=range_roi,
            feature_mode="legacy_4ch",
        )

    with pytest.raises(ValueError, match="ROI"):
        resolve_cooperative_features_h5(
            feat_path,
            range_roi=(0.0, 3.5),
            feature_mode="real_imag",
        )

    missing_raw = tmp_path / "other_raw.h5"
    _write_synthetic_raw_h5(missing_raw, n_frames=1)
    with pytest.raises(FileNotFoundError, match="require"):
        resolve_cooperative_features_h5(
            missing_raw,
            range_roi=range_roi,
            feature_mode="real_imag",
            require=True,
        )


def test_features_dataset_augmentation_smoke(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "feat.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=1)
    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        feature_mode="real_imag",
        range_roi=(0.0, 4.0),
        show_progress=False,
    )
    ds = CooperativeMonostaticFeaturesDataset(
        feat_path,
        np.array([0], dtype=np.int64),
        feature_noise_std=0.1,
        spec_augment_prob=1.0,
        spec_augment_max_bins=2,
        augment=True,
    )
    item = ds[0]
    assert item["dual_profiles"].shape[0] == 4
    assert item["dual_profiles"].dtype == torch.float32
    ds.close()


def test_open_training_dataset_auto_detects_features(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "raw_features.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=1)
    build_cooperative_monostatic_features_h5(
        raw_path, feat_path, feature_mode="legacy_4ch", show_progress=False
    )

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
    build_cooperative_monostatic_features_h5(
        raw_path, feat_path, feature_mode="legacy_4ch", show_progress=False
    )

    with pytest.raises(ValueError, match="already features"):
        build_cooperative_monostatic_features_h5(
            feat_path, tmp_path / "dup.h5", show_progress=False
        )
