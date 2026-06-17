"""AWGN 块：接收端 SNR (dB) 定标与 Sionna 等价性。"""
from __future__ import annotations

import math

import pytest
import torch
from sionna.phy.channel import AWGN as SionnaAWGN

from isac.channel.awgn import AWGN, snr_db_to_noise_power


def test_snr_db_to_noise_power_matches_formula() -> None:
    sig_p = 2.5
    snr_db = 10.0
    no = snr_db_to_noise_power(sig_p, snr_db)
    assert no == pytest.approx(sig_p / 10.0)


def test_snr_db_to_noise_power_zero_signal_raises() -> None:
    with pytest.raises(ValueError, match="信号功率须为正"):
        snr_db_to_noise_power(0.0, 10.0)


def test_awgn_zero_signal_raises() -> None:
    awgn = AWGN()
    x = torch.zeros(8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="信号功率须为正"):
        awgn(x, 10.0)


def test_awgn_output_shape() -> None:
    awgn = AWGN()
    x = torch.randn(4, 16, dtype=torch.complex64)
    y = awgn(x, 15.0)
    assert y.shape == x.shape


def test_awgn_matches_sionna_with_same_seed() -> None:
    torch.manual_seed(123)
    x = torch.randn(32, 64, dtype=torch.complex64)
    snr_db = 12.5
    sig_p = float(torch.mean(torch.abs(x) ** 2).item())
    no_rx = snr_db_to_noise_power(sig_p, snr_db)

    torch.manual_seed(456)
    y_sionna = SionnaAWGN()(x, no_rx)

    torch.manual_seed(456)
    y_local = AWGN()(x, snr_db)

    assert torch.allclose(y_local, y_sionna, rtol=0.0, atol=0.0)


def test_awgn_achieves_target_snr_db() -> None:
    """Monte Carlo：实测接收 SNR 应接近目标 snr_db。"""
    awgn = AWGN()
    x = torch.randn(256, 128, dtype=torch.complex64)
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
