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
