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

    model = CooperativeMonostaticCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        pool_mode="gap",
        fusion_mode="early",
    )
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
        fusion_mode="early",
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
            "fusion_mode": "early",
            "aux_range": False,
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


def test_localize_xy_two_monostatic_ranges_torch_matches_numpy():
    from isac.models import localize_xy_two_monostatic_ranges_torch
    from isac.sensing.localization import localize_xy_two_monostatic_ranges

    pos0 = (0.0, -2.0)
    pos1 = (-2.0, 0.0)
    target = (0.0, 0.0)
    r0 = float(torch.linalg.vector_norm(torch.tensor(target) - torch.tensor(pos0)))
    r1 = float(torch.linalg.vector_norm(torch.tensor(target) - torch.tensor(pos1)))
    est_np = localize_xy_two_monostatic_ranges(pos0, r0, pos1, r1)
    est_t = localize_xy_two_monostatic_ranges_torch(
        torch.tensor([r0]),
        torch.tensor([r1]),
        pos0_xy=torch.tensor(pos0),
        pos1_xy=torch.tensor(pos1),
    )
    assert est_t.shape == (1, 2)
    assert float(est_t[0, 0]) == pytest.approx(est_np[0], abs=1e-5)
    assert float(est_t[0, 1]) == pytest.approx(est_np[1], abs=1e-5)


def test_localize_xy_two_monostatic_ranges_torch_batch_and_differentiable():
    from isac.models import localize_xy_two_monostatic_ranges_torch

    r0 = torch.tensor([2.0, 2.5], requires_grad=True)
    r1 = torch.tensor([2.0, 2.2], requires_grad=True)
    xy = localize_xy_two_monostatic_ranges_torch(
        r0,
        r1,
        pos0_xy=torch.tensor([0.0, -2.0]),
        pos1_xy=torch.tensor([-2.0, 0.0]),
    )
    assert xy.shape == (2, 2)
    assert torch.isfinite(xy).all()
    xy.sum().backward()
    assert r0.grad is not None and torch.isfinite(r0.grad).all()
    assert r1.grad is not None and torch.isfinite(r1.grad).all()
    assert (r0.grad.abs() > 0).any()
    assert (r1.grad.abs() > 0).any()


def _geom_cnn(**kwargs) -> CooperativeMonostaticCNN:
    defaults = dict(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        fusion_mode="late",
        geom_residual=True,
        pool_mode="attention",
        dropout=0.0,
        dev0_xy=(0.0, -2.0),
        dev1_xy=(-2.0, 0.0),
    )
    defaults.update(kwargs)
    return CooperativeMonostaticCNN(**defaults)


def test_cooperative_monostatic_cnn_geom_residual_forward_shapes():
    model = _geom_cnn()
    model.eval()
    batch = torch.randn(3, 4, 40)
    with torch.no_grad():
        xy, ranges = model.forward_with_aux(batch)
        xy_only = model(batch)
    assert xy.shape == (3, 2)
    assert xy_only.shape == (3, 2)
    assert torch.allclose(xy, xy_only)
    assert ranges is not None and ranges.shape == (3, 2)
    assert (ranges > 0).all()
    assert torch.isfinite(xy).all()


def test_cooperative_monostatic_cnn_geom_residual_equals_geom_plus_delta():
    """xy = xy_geom + Δxy：将 xy_head 置零后输出应等于纯几何交会。"""
    from isac.models import localize_xy_two_monostatic_ranges_torch

    model = _geom_cnn()
    model.eval()
    with torch.no_grad():
        for p in model.xy_head.parameters():
            p.zero_()
    batch = torch.randn(4, 4, 40)
    with torch.no_grad():
        xy, ranges = model.forward_with_aux(batch)
        xy_geom = localize_xy_two_monostatic_ranges_torch(
            ranges[:, 0],
            ranges[:, 1],
            pos0_xy=model.dev0_xy,
            pos1_xy=model.dev1_xy,
        )
    assert torch.allclose(xy, xy_geom, atol=1e-5, rtol=1e-5)


def test_cooperative_monostatic_cnn_geom_residual_backward_end_to_end():
    model = _geom_cnn(stopgrad_geom=False)
    model.train()
    batch = torch.randn(3, 4, 40, requires_grad=True)
    xy, ranges = model.forward_with_aux(batch)
    assert ranges is not None
    loss = xy.pow(2).mean() + 0.01 * ranges.pow(2).mean()
    loss.backward()
    assert batch.grad is not None and torch.isfinite(batch.grad).all()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.range_head.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.xy_head.parameters()
    )


def test_cooperative_monostatic_cnn_geom_residual_stopgrad_blocks_range_path():
    """stopgrad_geom=True 时，仅对 xy 反传不应更新 range_head。"""
    model = _geom_cnn(stopgrad_geom=True)
    model.train()
    batch = torch.randn(3, 4, 40)
    xy, _ranges = model.forward_with_aux(batch)
    xy.pow(2).mean().backward()
    assert all(
        p.grad is None or float(p.grad.abs().sum()) == 0.0
        for p in model.range_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.xy_head.parameters()
    )


def test_cooperative_monostatic_cnn_geom_residual_requires_late():
    with pytest.raises(ValueError, match="geom_residual"):
        CooperativeMonostaticCNN(fusion_mode="early", geom_residual=True)


def test_cooperative_monostatic_cnn_geom_residual_checkpoint_roundtrip(tmp_path):
    from isac.models.model_design import load_cooperative_monostatic_cnn_checkpoint

    model = _geom_cnn(stopgrad_geom=True, dropout=0.2)
    model.eval()
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        xy_before, ranges_before = model.forward_with_aux(batch)

    ckpt_path = tmp_path / "geom.pth"
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
            "aux_range": False,
            "geom_residual": True,
            "stopgrad_geom": True,
            "dev0_xy": (0.0, -2.0),
            "dev1_xy": (-2.0, 0.0),
        },
        ckpt_path,
    )
    loaded = load_cooperative_monostatic_cnn_checkpoint(ckpt_path, "cpu")
    assert loaded.geom_residual is True
    assert loaded.stopgrad_geom is True
    assert loaded.dev0_xy is not None
    assert float(loaded.dev0_xy[0]) == pytest.approx(0.0)
    assert float(loaded.dev0_xy[1]) == pytest.approx(-2.0)
    assert float(loaded.dev1_xy[0]) == pytest.approx(-2.0)
    assert float(loaded.dev1_xy[1]) == pytest.approx(0.0)
    loaded.eval()
    with torch.no_grad():
        xy_after, ranges_after = loaded.forward_with_aux(batch)
    assert xy_after.shape == (2, 2)
    assert ranges_after is not None and ranges_after.shape == (2, 2)
    assert torch.allclose(xy_before, xy_after, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ranges_before, ranges_after, atol=1e-6, rtol=1e-6)


def _xattn_cnn(**kwargs) -> CooperativeMonostaticCNN:
    defaults = dict(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        fusion_mode="late",
        cross_attn=True,
        geom_residual=True,
        pool_mode="attention",
        dropout=0.0,
        dev0_xy=(0.0, -2.0),
        dev1_xy=(-2.0, 0.0),
    )
    defaults.update(kwargs)
    return CooperativeMonostaticCNN(**defaults)


def test_bidirectional_station_cross_attention_shapes_and_grad():
    from isac.models.model_design import BidirectionalStationCrossAttention

    attn = BidirectionalStationCrossAttention(32, num_heads=4, dropout=0.0)
    f0 = torch.randn(3, 32, requires_grad=True)
    f1 = torch.randn(3, 32, requires_grad=True)
    o0, o1 = attn(f0, f1)
    assert o0.shape == (3, 32) and o1.shape == (3, 32)
    (o0.sum() + o1.sum()).backward()
    assert f0.grad is not None and torch.isfinite(f0.grad).all()
    assert f1.grad is not None and torch.isfinite(f1.grad).all()
    assert (f0.grad.abs() > 0).any() and (f1.grad.abs() > 0).any()


def test_cooperative_monostatic_cnn_cross_attn_forward_shapes():
    model = _xattn_cnn()
    model.eval()
    batch = torch.randn(3, 4, 40)
    with torch.no_grad():
        xy, ranges = model.forward_with_aux(batch)
        xy_only = model(batch)
    assert xy.shape == (3, 2)
    assert xy_only.shape == (3, 2)
    assert torch.allclose(xy, xy_only)
    assert ranges is not None and ranges.shape == (3, 2)
    assert (ranges > 0).all()
    assert torch.isfinite(xy).all()
    assert model.station_cross_attn is not None


def test_cooperative_monostatic_cnn_cross_attn_without_geom():
    model = _xattn_cnn(geom_residual=False)
    model.eval()
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        xy, ranges = model.forward_with_aux(batch)
    assert xy.shape == (2, 2)
    assert ranges is None
    assert torch.isfinite(xy).all()


def test_cooperative_monostatic_cnn_cross_attn_backward():
    model = _xattn_cnn()
    model.train()
    batch = torch.randn(3, 4, 40, requires_grad=True)
    xy, ranges = model.forward_with_aux(batch)
    assert ranges is not None
    loss = xy.pow(2).mean() + 0.01 * ranges.pow(2).mean()
    loss.backward()
    assert batch.grad is not None and torch.isfinite(batch.grad).all()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.station_cross_attn.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.xy_head.parameters()
    )


def test_cooperative_monostatic_cnn_cross_attn_requires_late():
    with pytest.raises(ValueError, match="cross_attn"):
        CooperativeMonostaticCNN(fusion_mode="early", cross_attn=True)


def test_cooperative_monostatic_cnn_cross_attn_checkpoint_roundtrip(tmp_path):
    from isac.models.model_design import load_cooperative_monostatic_cnn_checkpoint

    model = _xattn_cnn(dropout=0.2, cross_attn_heads=2)
    model.eval()
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        xy_before, ranges_before = model.forward_with_aux(batch)

    ckpt_path = tmp_path / "xattn.pth"
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
            "aux_range": False,
            "geom_residual": True,
            "stopgrad_geom": False,
            "cross_attn": True,
            "cross_attn_heads": 2,
            "dev0_xy": (0.0, -2.0),
            "dev1_xy": (-2.0, 0.0),
        },
        ckpt_path,
    )
    loaded = load_cooperative_monostatic_cnn_checkpoint(ckpt_path, "cpu")
    assert loaded.cross_attn is True
    assert loaded.cross_attn_heads == 2
    assert loaded.geom_residual is True
    assert loaded.station_cross_attn is not None
    loaded.eval()
    with torch.no_grad():
        xy_after, ranges_after = loaded.forward_with_aux(batch)
    assert torch.allclose(xy_before, xy_after, atol=1e-6, rtol=1e-6)
    assert ranges_before is not None and ranges_after is not None
    assert torch.allclose(ranges_before, ranges_after, atol=1e-6, rtol=1e-6)


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
