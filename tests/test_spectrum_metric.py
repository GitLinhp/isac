"""SpectrumMetric 与 DelayDopplerRoi 坐标换算单元测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.sensing.metric import SpectrumMetric
from isac.sensing.spectrum import DelayDopplerRoi, SensingPerformance


def _sp() -> SimpleNamespace:
    return SimpleNamespace(
        range_resolution_monostatic=2.44,
        velocity_resolution_monostatic=1.171,
        delay_resolution=2.44 / (3e8 / 2),
        doppler_resolution=1.171 * (3e8 / 2) / 6e9,
        carrier_frequency=6e9,
        rg=SimpleNamespace(num_ofdm_symbols=512, fft_size=2048),
    )


def _metric_stub() -> SpectrumMetric:
    return SpectrumMetric(_sp())


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


def _roi(
    sp: SensingPerformance,
    max_range_m: float = 310.0,
    max_velocity_mps: float = 150.0,
) -> DelayDopplerRoi:
    return DelayDopplerRoi(
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
        sensing_performance=sp,
    )


def test_roi_bin_counts_match_dd_spectrum():
    roi = DelayDopplerRoi(
        max_range_m=310.0,
        max_velocity_mps=150.0,
        sensing_performance=_sp(),  # type: ignore[arg-type]
    )
    assert roi.delay_bin_count() == 128
    assert roi.doppler_half_bins() == 128


def test_bin_slices_inverse_physical_limits():
    sp = _sensing_performance()
    roi = _roi(sp)
    h_full = torch.zeros(512, 2048)
    slices = roi.bin_slices(h_full)
    dop_start, dop_end, _, delay_end = slices
    # 对称 ROI：长度 2*half+1，半宽 = (dop_end - dop_start) // 2
    max_range_m = (delay_end - 1) * sp.range_resolution_monostatic
    max_velocity_mps = ((dop_end - dop_start) // 2) * sp.velocity_resolution_monostatic
    assert max_range_m == pytest.approx(309.96, rel=1e-3)
    assert max_velocity_mps == pytest.approx(149.888, rel=1e-3)
    assert (dop_end - dop_start) % 2 == 1


def test_small_roi_velocity_axis_symmetric():
    """小 ROI（dop_half=2）速度轴须关于 0 对称，避免负速度侧少 1 bin。"""
    sp = _sensing_performance()
    roi = DelayDopplerRoi(
        max_range_m=30.0,
        max_velocity_mps=5.0,
        sensing_performance=sp,
    )
    h_full = torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size)
    dop_start, dop_end, _, _ = roi.bin_slices(h_full)
    dop_half = roi.doppler_half_bins()
    assert dop_end - dop_start == 2 * dop_half + 1
    assert dop_start == sp.rg.num_ofdm_symbols // 2 - dop_half
    assert dop_end == sp.rg.num_ofdm_symbols // 2 + dop_half + 1

    v = sp.velocity_bins_monostatic[dop_start:dop_end]
    assert v.max() == pytest.approx(-v.min(), rel=1e-3)
    assert abs(v[dop_half]) == pytest.approx(0.0, abs=1e-9)


def test_local_bins_match_global_axes_after_symmetric_roi():
    sp = _sensing_performance()
    m = SpectrumMetric(sp)
    roi = _roi(sp)
    slices = roi.bin_slices(
        torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size),
    )
    dop_start, dop_end, delay_start, delay_end = slices
    num_doppler = dop_end - dop_start

    local_delays = [0, 10, 50]
    local_dops = [0, 32, num_doppler - 1]
    tau_local, fd_local = m.local_bins_to_tau_fd(
        local_delays,
        local_dops,
        num_doppler_bins=num_doppler,
    )
    global_delays = [delay_start + d for d in local_delays]
    global_dops = [dop_start + d for d in local_dops]
    tau_global, fd_global = m.global_bin_to_tau_fd(global_delays, global_dops)

    assert tau_local.numpy() == pytest.approx(tau_global, rel=1e-9)
    assert fd_local.numpy() == pytest.approx(fd_global, rel=1e-6)


@pytest.mark.parametrize("sens_mode", ["monostatic", "bistatic"])
def test_local_bins_match_range_velocity_axes(sens_mode: str):
    sp = _sensing_performance()
    m = SpectrumMetric(sp)
    roi = _roi(sp)
    slices = roi.bin_slices(
        torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size),
        sens_mode=sens_mode,  # type: ignore[arg-type]
    )
    dop_start, dop_end, delay_start, delay_end = slices
    num_doppler = dop_end - dop_start

    local_delays = torch.tensor([0, 20], dtype=torch.float64)
    local_dops = torch.tensor([10, 40], dtype=torch.float64)
    _, _, range_m, v_mps = m.local_bins_to_range_velocity(
        local_delays,
        local_dops,
        num_doppler_bins=num_doppler,
        sens_mode=sens_mode,  # type: ignore[arg-type]
    )

    global_delays = (local_delays + delay_start).numpy().astype(int)
    global_dops = (local_dops + dop_start).numpy().astype(int)
    expected_r = getattr(sp, f"range_bins_{sens_mode}")[global_delays]
    expected_v = getattr(sp, f"velocity_bins_{sens_mode}")[global_dops]
    assert range_m.numpy() == pytest.approx(expected_r, rel=1e-9)
    assert v_mps.numpy() == pytest.approx(expected_v, rel=1e-6)


def test_axes_for_roi_dd_and_rv():
    sp = _sensing_performance()
    roi = _roi(sp)
    h_full = torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size)
    roi.crop(h_full)
    dop_start, dop_end, delay_start, delay_end = roi.slices  # type: ignore[misc]

    x_dd, y_dd, x_label_dd, y_label_dd = roi.axes("dd")
    assert x_label_dd == "Delay (ns)"
    assert y_label_dd == "Doppler (Hz)"
    assert x_dd[0] == pytest.approx(sp.delay_bins[delay_start])
    assert y_dd[0] == pytest.approx(sp.doppler_bins[dop_start])

    x_rv, y_rv, _, _ = roi.axes("rv", sens_mode="bistatic")
    assert x_rv[0] == pytest.approx(sp.range_bins_bistatic[delay_start])
    assert y_rv[0] == pytest.approx(sp.velocity_bins_bistatic[dop_start])


def test_bin_slices_bistatic_physical_limits():
    """双基地 ROI：TOML 物理上界与裁切后 bistatic RV 轴一致（非 2 倍）。"""
    sp = _sensing_performance()
    roi = DelayDopplerRoi(
        max_range_m=100.0,
        max_velocity_mps=10.0,
        sensing_performance=sp,
    )
    slices = roi.bin_slices(
        torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size),
        sens_mode="bistatic",
    )
    dop_start, dop_end, _, delay_end = slices
    max_range_m = (delay_end - 1) * sp.range_resolution_bistatic
    max_velocity_mps = ((dop_end - dop_start) // 2) * sp.velocity_resolution_bistatic
    assert max_range_m == pytest.approx(97.6, rel=0.05)
    assert max_velocity_mps == pytest.approx(9.36, rel=0.15)


def test_physical_to_local_bins_roundtrip():
    sp = _sensing_performance()
    m = SpectrumMetric(sp)
    roi = _roi(sp)
    slices = roi.bin_slices(
        torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size),
    )
    num_doppler = slices[1] - slices[0]

    local_delays = torch.tensor([0.0, 20.0, 50.0], dtype=torch.float64)
    local_dops = torch.tensor([10.0, 64.0, num_doppler - 1], dtype=torch.float64)
    _, _, range_m, v_mps = m.local_bins_to_range_velocity(
        local_delays,
        local_dops,
        num_doppler_bins=num_doppler,
        sens_mode="monostatic",
    )
    delay_back, dop_back = m.physical_to_local_bins(
        range_m,
        v_mps,
        num_doppler_bins=num_doppler,
        sens_mode="monostatic",
    )
    assert delay_back.numpy() == pytest.approx(local_delays.numpy(), rel=1e-9)
    assert dop_back.numpy() == pytest.approx(local_dops.numpy(), rel=1e-9)
