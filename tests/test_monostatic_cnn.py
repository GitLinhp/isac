"""单基地 DD-CNN 前向与标签工具测试。"""

import torch

from isac.learning.dd_spectrum import (
    crop_dd_roi,
    dd_spectrum_to_features,
    monostatic_labels_from_kinematics,
)
from isac.learning.monostatic_cnn import MonostaticCNNConfig, MonostaticDelayDopplerCNN


def test_monostatic_cnn_forward_shape():
    cfg = MonostaticCNNConfig(in_channels=2, max_range_m=100.0, max_velocity_mps=10.0)
    model = MonostaticDelayDopplerCNN(cfg)
    x = torch.randn(4, 2, 256, 128)
    y = model(x)
    assert y.shape == (4, 2)
    assert (y[:, 0] >= 0).all()
    assert (y[:, 1] >= -cfg.max_velocity_mps).all()
    assert (y[:, 1] <= cfg.max_velocity_mps).all()


def test_dd_feature_and_crop():
    h_dd = torch.randn(512, 2048, dtype=torch.complex64)
    roi = crop_dd_roi(h_dd, offset=64)
    assert roi.shape == (128, 64)
    feat = dd_spectrum_to_features(h_dd, offset=64)
    assert feat.shape == (2, 128, 64)


def test_monostatic_labels():
    bs = [0.0, 0.0, 0.0]
    tgt = [30.0, 0.0, 0.0]
    vel = [5.0, 0.0, 0.0]
    r, v = monostatic_labels_from_kinematics(tgt, vel, bs)
    assert abs(r - 30.0) < 1e-6
    assert abs(v - 5.0) < 1e-6
