"""preprocess 模块形状与类型测试。"""

import torch

from isac.models.preprocess import (
    kinematics_to_range_velocity,
    kinematics_to_target_bins,
    normalize_spectrum_batch,
    spectrum_tensor_to_features,
)
from isac.sensing.spectrum import SensingPerformance


def _sensing_performance() -> SensingPerformance:
    from sionna.phy.ofdm import ResourceGrid

    rg = ResourceGrid(
        num_ofdm_symbols=512,
        fft_size=2048,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=32,
        dc_null=False,
        device="cpu",
    )
    return SensingPerformance(rg, carrier_frequency=6e9)


def test_normalize_spectrum_batch_2d():
    spec = torch.randn(32, 16, dtype=torch.complex64)
    batch = normalize_spectrum_batch(spec)
    assert batch.shape == (1, 32, 16)


def test_normalize_spectrum_batch_3d():
    spec = torch.randn(2, 32, 16, dtype=torch.complex64)
    batch = normalize_spectrum_batch(spec)
    assert batch.shape == (2, 32, 16)


def test_kinematics_to_range_velocity_batch():
    pos = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    vel = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bs = torch.tensor([0.0, 0.0, 0.0])
    r, v = kinematics_to_range_velocity(pos, vel, bs)
    assert r.shape == (2,)
    assert v.shape == (2,)


def test_kinematics_to_target_bins_matches_features_pipeline():
    sp = _sensing_performance()
    pos = torch.tensor([[30.0, 0.0, 0.0]])
    vel = torch.tensor([[5.0, 0.0, 0.0]])
    bs = torch.tensor([0.0, 0.0, 0.0])
    bins = kinematics_to_target_bins(
        pos,
        vel,
        bs,
        sensing_performance=sp,
        num_doppler_bins=256,
    )
    assert bins.shape == (1, 2)
    spec = torch.randn(256, 128, dtype=torch.complex64)
    feats = spectrum_tensor_to_features(spec)
    assert feats.shape == (1, 2, 256, 128)
