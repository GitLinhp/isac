"""Cooperative monostatic CNN feature modes 与加权损失测试。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from isac.models import (
    TargetPositionRmseLoss,
    cooperative_feature_in_channels,
    dual_roi_to_model_input,
    save_cooperative_norm_stats,
    load_cooperative_norm_stats,
    apply_cooperative_cpi_augmentation,
    apply_cooperative_feature_augmentation,
    compute_logmag_norm_stats_from_dual_rois,
)


def test_cooperative_feature_in_channels():
    assert cooperative_feature_in_channels("real_imag") == 4
    assert cooperative_feature_in_channels("logmag_fixed_norm") == 2
    assert cooperative_feature_in_channels("legacy_4ch") == 4


def test_dual_roi_to_real_imag_features():
    dual = torch.randn(2, 34, dtype=torch.complex64)
    feat = dual_roi_to_model_input(dual, mode="real_imag")
    assert feat.shape == (4, 34)
    assert feat.dtype == torch.float32


def test_dual_roi_to_logmag_fixed_norm():
    dual = torch.randn(2, 34, dtype=torch.complex64)
    means = np.array([0.0, 1.0], dtype=np.float64)
    stds = np.array([1.0, 2.0], dtype=np.float64)
    feat = dual_roi_to_model_input(
        dual,
        mode="logmag_fixed_norm",
        norm_means=means,
        norm_stds=stds,
    )
    assert feat.shape == (2, 34)


def test_dual_roi_batch_real_imag():
    batch = torch.randn(3, 2, 34, dtype=torch.complex64)
    feat = dual_roi_to_model_input(batch, mode="real_imag")
    assert feat.shape == (3, 4, 34)


def test_compute_logmag_norm_stats():
    dual_rois = [
        np.stack(
            [
                np.ones(34, dtype=np.complex64),
                np.full(34, 2.0, dtype=np.complex64),
            ],
            axis=0,
        )
    ]
    means, stds = compute_logmag_norm_stats_from_dual_rois(dual_rois)
    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.all(stds > 0)


def test_save_load_norm_stats(tmp_path: Path):
    means = np.array([1.0, 2.0])
    stds = np.array([0.5, 0.6])
    path = tmp_path / "stats.npz"
    save_cooperative_norm_stats(path, means=means, stds=stds)
    loaded_means, loaded_stds, mode = load_cooperative_norm_stats(path)
    np.testing.assert_allclose(loaded_means, means)
    np.testing.assert_allclose(loaded_stds, stds)
    assert mode == "logmag_fixed_norm"


def test_target_position_rmse_loss_weighted():
    criterion = TargetPositionRmseLoss()
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    weight = torch.tensor([1.0, 4.0], dtype=torch.float32)
    loss = criterion(pred, target, sample_weight=weight)
    # sample0: 0 error; sample1: ||(1,1)||^2 = 2, weight 4 -> sqrt((2*4)/5)
    assert loss.item() == pytest.approx((2.0 * 4.0 / 5.0) ** 0.5, abs=1e-5)


def test_apply_cooperative_feature_augmentation_noise():
    feat = torch.zeros(4, 34)
    out = apply_cooperative_feature_augmentation(feat, noise_std=0.1)
    assert out.shape == feat.shape
    assert not torch.allclose(out, feat)


def test_apply_cooperative_cpi_augmentation_amp_and_noise():
    dual = torch.ones(2, 16, dtype=torch.complex64)
    rng = np.random.default_rng(0)
    out = apply_cooperative_cpi_augmentation(
        dual, amp_scale=0.25, complex_noise_std=0.05, rng=rng
    )
    assert out.shape == dual.shape
    assert out.dtype == dual.dtype
    # Per-station scales differ → magnitudes diverge from 1.
    mags = torch.abs(out).mean(dim=-1)
    assert not torch.allclose(mags, torch.ones(2))


def test_apply_cooperative_cpi_augmentation_seeded_reproducible():
    dual = torch.randn(2, 12, dtype=torch.complex64)
    a = apply_cooperative_cpi_augmentation(
        dual, amp_scale=0.2, complex_noise_std=0.03, rng=np.random.default_rng(11)
    )
    b = apply_cooperative_cpi_augmentation(
        dual, amp_scale=0.2, complex_noise_std=0.03, rng=np.random.default_rng(11)
    )
    c = apply_cooperative_cpi_augmentation(
        dual, amp_scale=0.2, complex_noise_std=0.03, rng=np.random.default_rng(12)
    )
    torch.testing.assert_close(a.real, b.real)
    torch.testing.assert_close(a.imag, b.imag)
    assert not torch.allclose(a.real, c.real)


def test_apply_cooperative_cpi_augmentation_noop():
    dual = torch.randn(2, 8, dtype=torch.complex64)
    out = apply_cooperative_cpi_augmentation(dual, amp_scale=0.0, complex_noise_std=0.0)
    assert out is dual or torch.equal(out.real, dual.real)


def test_apply_real_imag_rms_norm_unit_energy():
    from isac.models import apply_real_imag_rms_norm

    feat = torch.zeros(4, 8)
    feat[0] = 3.0
    feat[1] = 4.0  # station0 rms = 5
    feat[2] = 0.0
    feat[3] = 2.0  # station1 rms = 2
    out = apply_real_imag_rms_norm(feat)
    assert torch.allclose(out[0], torch.full((8,), 3.0 / 5.0))
    assert torch.allclose(out[1], torch.full((8,), 4.0 / 5.0))
    assert torch.allclose(out[2], torch.zeros(8))
    assert torch.allclose(out[3], torch.ones(8))


def test_dual_roi_to_model_input_rms_norm():
    from isac.models import dual_roi_to_model_input

    dual = torch.ones(2, 10, dtype=torch.complex64) * (3.0 + 4.0j)
    out = dual_roi_to_model_input(dual, mode="real_imag", feature_norm="rms")
    assert out.shape == (4, 10)
    # |3+4j| = 5 → re/im become 3/5, 4/5
    assert torch.allclose(out[0], torch.full((10,), 0.6), atol=1e-5)
    assert torch.allclose(out[1], torch.full((10,), 0.8), atol=1e-5)
