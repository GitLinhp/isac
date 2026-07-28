"""Cooperative Monostatic CNN 与位置 RMSE 损失测试。"""

import pytest
import torch

from isac.models import (
    COOPERATIVE_POOL_MODES,
    CooperativeMonostaticCNN,
    TargetPositionRmseLoss,
    dual_range_profile_to_features,
    dual_range_profiles_to_features,
    range_profile_to_features,
)
from isac.models.model_design import Conv1dResidualBlock


def test_range_profile_to_features_shape():
    profile = torch.randn(34, dtype=torch.complex64)
    feat = range_profile_to_features(profile)
    assert feat.shape == (2, 34)
    assert feat.dtype == torch.float32


def test_dual_range_profile_to_features_shape():
    p0 = torch.randn(34, dtype=torch.complex64)
    p1 = torch.randn(34, dtype=torch.complex64)
    feat = dual_range_profile_to_features(p0, p1)
    assert feat.shape == (4, 34)


def test_dual_range_profiles_batch_features_shape():
    batch = torch.randn(3, 2, 34, dtype=torch.complex64)
    feat = dual_range_profiles_to_features(batch)
    assert feat.shape == (3, 4, 34)


def test_cooperative_monostatic_cnn_forward_complex():
    model = CooperativeMonostaticCNN(in_channels=4)
    model.eval()
    dual = torch.randn(2, 34, dtype=torch.complex64)
    with torch.no_grad():
        xy = model(dual)
    assert xy.shape == (1, 2)


def test_cooperative_monostatic_cnn_batch_forward():
    model = CooperativeMonostaticCNN(in_channels=4)
    model.eval()
    batch = torch.randn(4, 2, 34, dtype=torch.complex64)
    with torch.no_grad():
        xy = model(batch)
    assert xy.shape == (4, 2)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_cooperative_monostatic_cnn_num_layers(num_layers: int):
    model = CooperativeMonostaticCNN(in_channels=4, num_layers=num_layers)
    model.eval()
    batch = torch.randn(2, 2, 34, dtype=torch.complex64)
    with torch.no_grad():
        xy = model(batch)
    assert xy.shape == (2, 2)


def test_cooperative_monostatic_cnn_num_layers_invalid():
    with pytest.raises(ValueError, match="num_layers"):
        CooperativeMonostaticCNN(num_layers=0)


@pytest.mark.parametrize("pool_mode", list(COOPERATIVE_POOL_MODES))
def test_cooperative_monostatic_cnn_pool_modes(pool_mode: str):
    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        pool_mode=pool_mode,
        multiscale_bins=4,
    )
    model.eval()
    batch = torch.randn(3, 2, 48, dtype=torch.complex64)
    with torch.no_grad():
        xy = model(batch)
    assert xy.shape == (3, 2)
    assert torch.isfinite(xy).all()


def test_cooperative_monostatic_cnn_pool_mode_invalid():
    with pytest.raises(ValueError, match="pool_mode"):
        CooperativeMonostaticCNN(pool_mode="not_a_mode")


def test_cooperative_monostatic_cnn_gap_checkpoint_compat(tmp_path):
    """默认 gap head 的 state_dict 键与旧 Sequential 布局兼容。"""
    from isac.models.model_design import load_cooperative_monostatic_cnn_checkpoint

    model = CooperativeMonostaticCNN(in_channels=4, base_channels=16, num_layers=2)
    ckpt_path = tmp_path / "gap.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.2,
            "model_type": "1d",
        },
        ckpt_path,
    )
    loaded = load_cooperative_monostatic_cnn_checkpoint(ckpt_path, "cpu")
    assert loaded.pool_mode == "gap"
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        assert loaded(batch).shape == (2, 2)


def test_cooperative_monostatic_cnn_soft_argmax_checkpoint_roundtrip(tmp_path):
    from isac.models.model_design import load_cooperative_monostatic_cnn_checkpoint

    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        pool_mode="soft_argmax",
    )
    ckpt_path = tmp_path / "soft.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.2,
            "model_type": "1d",
            "pool_mode": "soft_argmax",
            "multiscale_bins": 8,
            "soft_argmax_temp": 1.0,
        },
        ckpt_path,
    )
    loaded = load_cooperative_monostatic_cnn_checkpoint(ckpt_path, "cpu")
    assert loaded.pool_mode == "soft_argmax"
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        assert loaded(batch).shape == (2, 2)


@pytest.mark.parametrize("fusion_mode", ["early", "late"])
@pytest.mark.parametrize("pool_mode", ["gap", "attention", "soft_argmax"])
def test_cooperative_monostatic_cnn_fusion_modes(fusion_mode: str, pool_mode: str):
    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        pool_mode=pool_mode,
        fusion_mode=fusion_mode,
    )
    model.eval()
    batch = torch.randn(3, 2, 48, dtype=torch.complex64)
    with torch.no_grad():
        xy = model(batch)
    assert xy.shape == (3, 2)
    assert torch.isfinite(xy).all()
    assert model.fusion_mode == fusion_mode


def test_cooperative_monostatic_cnn_late_fusion_odd_channels_invalid():
    with pytest.raises(ValueError, match="偶数"):
        CooperativeMonostaticCNN(in_channels=3, fusion_mode="late")


def test_cooperative_monostatic_cnn_fusion_mode_invalid():
    with pytest.raises(ValueError, match="fusion_mode"):
        CooperativeMonostaticCNN(fusion_mode="mid")


@pytest.mark.parametrize("fusion_mode", ["early", "late"])
def test_cooperative_monostatic_cnn_aux_range(fusion_mode: str):
    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        fusion_mode=fusion_mode,
        aux_range=True,
        pool_mode="gap",
    )
    model.eval()
    batch = torch.randn(4, 4, 40)
    with torch.no_grad():
        xy, ranges = model.forward_with_aux(batch)
        xy_only = model(batch)
    assert xy.shape == (4, 2)
    assert ranges is not None and ranges.shape == (4, 2)
    assert torch.allclose(xy, xy_only)


def test_cooperative_monostatic_cnn_late_aux_checkpoint_roundtrip(tmp_path):
    from isac.models.model_design import load_cooperative_monostatic_cnn_checkpoint

    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        fusion_mode="late",
        aux_range=True,
        pool_mode="attention",
    )
    ckpt_path = tmp_path / "late_aux.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.2,
            "model_type": "1d",
            "pool_mode": "attention",
            "multiscale_bins": 8,
            "soft_argmax_temp": 1.0,
            "fusion_mode": "late",
            "aux_range": True,
        },
        ckpt_path,
    )
    loaded = load_cooperative_monostatic_cnn_checkpoint(ckpt_path, "cpu")
    assert loaded.fusion_mode == "late"
    assert loaded.aux_range is True
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        xy, ranges = loaded.forward_with_aux(batch)
    assert xy.shape == (2, 2)
    assert ranges is not None and ranges.shape == (2, 2)


def test_monostatic_ranges_from_xy_and_aux_loss():
    from isac.models import aux_range_rmse_loss, monostatic_ranges_from_xy

    xy = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
    ranges = monostatic_ranges_from_xy(
        xy, dev0_xy=(0.0, -2.0), dev1_xy=(-2.0, 0.0)
    )
    assert ranges.shape == (2, 2)
    # (0,0) → r0=2, r1=2
    assert float(ranges[0, 0]) == pytest.approx(2.0, abs=1e-5)
    assert float(ranges[0, 1]) == pytest.approx(2.0, abs=1e-5)
    loss = aux_range_rmse_loss(ranges, ranges.clone())
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_conv1d_residual_block_forward():
    block = Conv1dResidualBlock(32, 32)
    x = torch.randn(2, 32, 17)
    y = block(x)
    assert y.shape == x.shape


def test_target_position_rmse_loss_finite():
    criterion = TargetPositionRmseLoss()
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.1, -0.1], [0.9, 1.1]], dtype=torch.float32)
    loss = criterion(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_target_position_rmse_loss_zero_on_match():
    criterion = TargetPositionRmseLoss()
    xy = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    loss = criterion(xy, xy.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
