"""SpectrumMetric 坐标换算单元测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.sensing.metric import SpectrumMetric
from isac.sensing.spectrum import SensingPerformance

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


def test_roi_bin_counts_match_dd_spectrum():
    m = _metric_stub()
    assert m.roi_delay_bin_count(310.0) == 128
    assert m.roi_doppler_half_bins(150.0) == 128


def test_bin_slices_inverse_physical_limits():
    m = _metric_stub()
    sp = _sp()
    roi = m.bin_slices(512, 2048, max_range_m=310.0, max_velocity_mps=150.0)
    dop_start, dop_end, _, delay_end = roi
    max_range_m = (delay_end - 1) * sp.range_resolution_monostatic
    max_velocity_mps = ((dop_end - dop_start) // 2) * sp.velocity_resolution_monostatic
    assert max_range_m == pytest.approx(309.96, rel=1e-3)
    assert max_velocity_mps == pytest.approx(149.888, rel=1e-3)


def test_local_bins_match_global_axes_after_symmetric_roi():
    sp = _sensing_performance()
    m = SpectrumMetric(sp)
    roi = m.bin_slices(
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        max_range_m=310.0,
        max_velocity_mps=150.0,
    )
    dop_start, dop_end, delay_start, delay_end = roi
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
    roi = m.bin_slices(
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        max_range_m=310.0,
        max_velocity_mps=150.0,
    )
    dop_start, dop_end, delay_start, delay_end = roi
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
    m = SpectrumMetric(sp)
    roi = m.bin_slices(
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        max_range_m=310.0,
        max_velocity_mps=150.0,
    )
    dop_start, dop_end, delay_start, delay_end = roi

    x_dd, y_dd, x_label_dd, y_label_dd = m.axes_for_roi(roi, "dd")
    assert x_label_dd == "Delay (ns)"
    assert y_label_dd == "Doppler (Hz)"
    assert x_dd[0] == pytest.approx(sp.delay_bins[delay_start])
    assert y_dd[0] == pytest.approx(sp.doppler_bins[dop_start])

    x_rv, y_rv, _, _ = m.axes_for_roi(roi, "rv", sens_mode="bistatic")
    assert x_rv[0] == pytest.approx(sp.range_bins_bistatic[delay_start])
    assert y_rv[0] == pytest.approx(sp.velocity_bins_bistatic[dop_start])