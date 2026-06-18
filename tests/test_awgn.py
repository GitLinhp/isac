"""AWGN 块：接收端 SNR (dB) 定标与 Sionna 等价性。"""
import math

import pytest
import torch
from sionna.phy.channel import AWGN as SionnaAWGN

from isac.channel.awgn import AWGN

_DEVICE = "cpu"


def test_awgn_zero_signal_raises() -> None:
    awgn = AWGN(device=_DEVICE)
    x = torch.zeros(8, dtype=torch.complex64, device=_DEVICE)
    with pytest.raises(ValueError, match="信号功率须为正"):
        awgn(x, 10.0)


def test_awgn_output_shape() -> None:
    awgn = AWGN(device=_DEVICE)
    x = torch.randn(4, 16, dtype=torch.complex64, device=_DEVICE)
    y = awgn(x, 15.0)
    assert y.shape == x.shape


def test_awgn_noise_variance_matches_sionna() -> None:
    """本地 AWGN(snr_db) 与 Sionna AWGN(no_rx) 噪声功率统计一致。"""
    snr_db = 12.5
    x = torch.randn(64, 128, dtype=torch.complex64, device=_DEVICE)
    sig_p = float(torch.mean(torch.abs(x) ** 2).item())
    no_rx = sig_p / (10.0 ** (snr_db / 10.0))

    local_awgn = AWGN(device=_DEVICE)
    sionna_awgn = SionnaAWGN(device=_DEVICE)
    n_trials = 100
    local_noise_p: list[float] = []
    sionna_noise_p: list[float] = []

    for seed in range(n_trials):
        torch.manual_seed(seed)
        y_local = local_awgn(x, snr_db)
        local_noise_p.append(float(torch.mean(torch.abs(y_local - x) ** 2).item()))

        torch.manual_seed(seed)
        y_sionna = sionna_awgn(x, no_rx)
        sionna_noise_p.append(float(torch.mean(torch.abs(y_sionna - x) ** 2).item()))

    mean_local = sum(local_noise_p) / n_trials
    mean_sionna = sum(sionna_noise_p) / n_trials
    assert mean_local == pytest.approx(mean_sionna, rel=0.05)
    assert mean_local == pytest.approx(no_rx, rel=0.15)


def test_awgn_achieves_target_snr_db() -> None:
    """Monte Carlo：实测接收 SNR 应接近目标 snr_db。"""
    awgn = AWGN(device=_DEVICE)
    x = torch.randn(256, 128, dtype=torch.complex64, device=_DEVICE)
    sig_p = float(torch.mean(torch.abs(x) ** 2).item())
    snr_db = 20.0
    n_trials = 200
    snr_est_db: list[float] = []

    for seed in range(n_trials):
        torch.manual_seed(seed)
        y = awgn(x, snr_db)
        noise_p = float(torch.mean(torch.abs(y - x) ** 2).item())
        snr_est_db.append(10.0 * math.log10(sig_p / noise_p))

    mean_snr_db = sum(snr_est_db) / len(snr_est_db)
    assert mean_snr_db == pytest.approx(snr_db, abs=0.5)
