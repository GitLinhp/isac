"""MUSIC 近距保护窗：物理距离 → 时延 bin 换算。"""

import math

import pytest
from sionna.phy.ofdm import ResourceGrid

from isac.sensing.detection.music_estimator import MUSICEstimator
from isac.sensing.spectrum import SensingPerformance


def _sp(subcarrier_spacing: float) -> SensingPerformance:
    rg = ResourceGrid(
        num_ofdm_symbols=512,
        fft_size=2048,
        subcarrier_spacing=subcarrier_spacing,
        cyclic_prefix_length=512,
        dc_null=False,
        device="cpu",
    )
    return SensingPerformance(rg, carrier_frequency=6e9)


def test_near_delay_guard_bins_monostatic_15_and_30_khz() -> None:
    sp_15 = _sp(15_000.0)
    sp_30 = _sp(30_000.0)

    assert sp_15.near_delay_guard_bins(1.0, "monostatic") == 1
    assert sp_30.near_delay_guard_bins(1.0, "monostatic") == 1

    # 旧固定 4 bin @ 15 kHz 会跳过约 19.5 m；新逻辑仅约 4.9 m
    old_guard_m = 4 * sp_15.range_resolution
    new_guard_m = sp_15.near_delay_guard_bins(1.0, "monostatic") * sp_15.range_resolution
    assert old_guard_m > 15.0
    assert new_guard_m < 5.0


def test_near_delay_guard_bins_zero_and_bistatic() -> None:
    sp = _sp(30_000.0)

    assert sp.near_delay_guard_bins(0.0, "monostatic") == 0
    assert sp.near_delay_guard_bins(-1.0, "monostatic") == 0

    mono = sp.near_delay_guard_bins(1.0, "monostatic")
    bi = sp.near_delay_guard_bins(1.0, "bistatic")
    assert bi == mono // 2 or bi == int(math.ceil(1.0 / sp.bistatic_range_resolution))


def test_near_delay_guard_bins_invalid_sens_mode() -> None:
    sp = _sp(30_000.0)
    with pytest.raises(ValueError, match="sens_mode"):
        sp.near_delay_guard_bins(1.0, "invalid")


def test_music_get_search_range_uses_physical_guard() -> None:
    sp = _sp(15_000.0)
    est = MUSICEstimator(device="cpu", sensing_performance=sp, near_range_guard_m=1.0)

    delay_start, delay_end, dop_start, dop_end = est._get_search_range(
        None,
        num_subcarriers=2048,
        num_symbols=512,
        sens_mode="monostatic",
        near_range_guard_m=1.0,
    )

    assert delay_start == 1
    assert delay_end == 2048
    assert dop_start == 0
    assert dop_end == 512

    # 显式 search_range 覆盖默认保护窗
    explicit = est._get_search_range(
        (0, 2048, 0, 512),
        num_subcarriers=2048,
        num_symbols=512,
        sens_mode="monostatic",
        near_range_guard_m=1.0,
    )
    assert explicit == (0, 2048, 0, 512)
