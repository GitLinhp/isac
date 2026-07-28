"""训练增强 RNG / DataLoader 严格可复现测试。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from isac.models.preprocess import apply_cooperative_feature_augmentation
from isac.utils import set_random_seed
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    CooperativeMonostaticFeaturesDataset,
    build_cooperative_monostatic_features_h5,
)

_VLEN = 32768


def _write_synthetic_raw_h5(path: Path, *, n_frames: int = 8) -> None:
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


def test_apply_augmentation_seeded_reproducible() -> None:
    feat = torch.randn(4, 16)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    rng_c = np.random.default_rng(8)
    out_a = apply_cooperative_feature_augmentation(
        feat, noise_std=0.1, spec_augment_prob=1.0, rng=rng_a
    )
    out_b = apply_cooperative_feature_augmentation(
        feat, noise_std=0.1, spec_augment_prob=1.0, rng=rng_b
    )
    out_c = apply_cooperative_feature_augmentation(
        feat, noise_std=0.1, spec_augment_prob=1.0, rng=rng_c
    )
    torch.testing.assert_close(out_a, out_b)
    assert not torch.allclose(out_a, out_c)


def test_features_dataset_same_seed_reproducible(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "feat.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=4)
    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=(0.0, 4.0),
        feature_mode="real_imag",
        show_progress=False,
    )

    def _item(seed: int):
        ds = CooperativeMonostaticFeaturesDataset(
            feat_path,
            np.array([0], dtype=np.int64),
            label_jitter_m=0.05,
            feature_noise_std=0.02,
            spec_augment_prob=1.0,
            spec_augment_max_bins=2,
            augment=True,
            seed=seed,
        )
        item = ds[0]
        ds.close()
        return item

    a = _item(42)
    b = _item(42)
    c = _item(99)
    torch.testing.assert_close(a["dual_profiles"], b["dual_profiles"])
    torch.testing.assert_close(a["target_xy"], b["target_xy"])
    assert not torch.allclose(a["dual_profiles"], c["dual_profiles"])


def test_features_dataloader_num_workers_zero_reproducible(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    feat_path = tmp_path / "feat.h5"
    _write_synthetic_raw_h5(raw_path, n_frames=8)
    build_cooperative_monostatic_features_h5(
        raw_path,
        feat_path,
        range_roi=(0.0, 4.0),
        feature_mode="real_imag",
        show_progress=False,
    )

    def _epoch_means(seed: int) -> torch.Tensor:
        set_random_seed(seed)
        ds = CooperativeMonostaticFeaturesDataset(
            feat_path,
            np.arange(8, dtype=np.int64),
            label_jitter_m=0.05,
            feature_noise_std=0.02,
            spec_augment_prob=0.5,
            augment=True,
            seed=seed,
        )
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=gen,
        )
        means = []
        for batch in loader:
            means.append(batch["dual_profiles"].mean())
            means.append(batch["target_xy"].mean())
        ds.close()
        return torch.stack(means)

    torch.testing.assert_close(_epoch_means(123), _epoch_means(123))
    assert not torch.allclose(_epoch_means(123), _epoch_means(456))
