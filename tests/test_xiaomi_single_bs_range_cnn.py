"""小米单站测距 CNN：预处理 / 损失 / 网络 / 帧级划分测试。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import random_split

from isac.xiaomi_models import (
    SingleBsRangeCNN,
    TargetRangeRmseLoss,
    default_range_bin_step,
    profile_to_features,
    profile_to_roi,
)
from isac.xiaomi_models.dataset import SingleBsRangeTorchDataset
from isac_imp.range_profile_roi_slice import compute_range_roi


def test_profile_to_roi_matches_compute_range_roi() -> None:
    vlen = 256
    step = default_range_bin_step()
    profile = (np.arange(vlen) + 1j * np.arange(vlen)[::-1]).astype(np.complex64)
    roi = (0.0, 2.0)
    start, num, _ = compute_range_roi(range_roi=roi, range_bin_step=step, vlen_in=vlen)
    out = profile_to_roi(profile, range_roi=roi, range_bin_step=step)
    assert out.shape == (num,)
    np.testing.assert_array_equal(out, profile[start : start + num])


def test_profile_to_features_shapes() -> None:
    profile = np.ones(32, dtype=np.complex64) * (1 + 0.5j)
    ri = profile_to_features(profile, mode="real_imag")
    mp = profile_to_features(profile, mode="mag_phase")
    assert ri.shape == (2, 32)
    assert mp.shape == (2, 32)
    assert ri.dtype == torch.float32


def test_target_range_rmse_zero() -> None:
    crit = TargetRangeRmseLoss()
    y = torch.tensor([1.0, 2.0, 3.0])
    loss = crit(y, y)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_single_bs_range_cnn_forward_shape() -> None:
    model = SingleBsRangeCNN(in_channels=2, base_channels=8, num_layers=2, dropout=0.0)
    x = torch.randn(4, 2, 64)
    y = model(x)
    assert y.shape == (4,)


def _write_tiny_h5(path: Path, *, n_frames: int = 20, vlen: int = 64) -> Path:
    rng = np.random.default_rng(0)
    profiles = (
        rng.standard_normal((n_frames, vlen)) + 1j * rng.standard_normal((n_frames, vlen))
    ).astype(np.complex64)
    target_range = np.linspace(0.5, 3.0, n_frames, dtype=np.float64)
    session_index = np.repeat(np.arange(n_frames // 5, dtype=np.int32), 5)[:n_frames]
    frame_index = np.tile(np.arange(5, dtype=np.int32), n_frames // 5 + 1)[:n_frames]
    with h5py.File(path, "w") as f:
        f.create_dataset("profiles", data=profiles)
        f.create_dataset("target_range", data=target_range)
        f.create_dataset("session_index", data=session_index)
        f.create_dataset("frame_index", data=frame_index)
        f.attrs["fft_len"] = 4096
        f.attrs["zeropadding_fac"] = 4
        f.attrs["vlen"] = vlen
    return path


def test_dataset_and_frame_random_split(tmp_path: Path) -> None:
    h5_path = _write_tiny_h5(tmp_path / "tiny.h5", n_frames=20, vlen=64)
    ds = SingleBsRangeTorchDataset(
        h5_path,
        range_roi=(0.0, 4.0),
        feature_mode="real_imag",
        cache_features=True,
    )
    assert len(ds) == 20
    sample = ds[0]
    assert sample["features"].ndim == 2
    assert sample["features"].shape[0] == 2
    assert sample["target_range"].ndim == 0

    val_ratio = 0.2
    n_val = max(1, int(round(len(ds) * val_ratio)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    assert len(train_ds) + len(val_ds) == len(ds)
    assert len(val_ds) == n_val
