"""Cooperative monostatic 离线 DSP 与双站定位测试。"""

from __future__ import annotations

import numpy as np
import pytest

from isac.sensing.localization import (
    localize_xy_two_monostatic_ranges,
    position_rmse_xy,
)
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_FFT_LEN,
    DEFAULT_TRANSPOSE_LEN,
    DEFAULT_ZEROPADDING_FAC,
    cooperative_range_bin_step_m,
    compute_1d_cfar_threshold,
    default_range_cfar_detector,
    divide_cpi_to_complex_range_profile,
    estimate_monostatic_range_esprit_m,
    estimate_monostatic_range_m,
    grc_cooperative_processing_params,
)


def test_divide_cpi_to_complex_range_profile_shape() -> None:
    vlen_range = DEFAULT_FFT_LEN * DEFAULT_ZEROPADDING_FAC
    vlen_divide = vlen_range * DEFAULT_TRANSPOSE_LEN
    cpi = np.ones(vlen_divide, dtype=np.complex64)
    profile = divide_cpi_to_complex_range_profile(cpi)
    assert profile.shape == (vlen_range,)
    assert profile.dtype == np.complex64


def test_grc_processing_params_match_expected_vlen() -> None:
    params = grc_cooperative_processing_params()
    assert params["vlen_divide_cpi"] == 32768
    assert params["vlen_range"] == 8192
    assert params["range_bin_step"] == pytest.approx(
        cooperative_range_bin_step_m(), rel=1e-6
    )


def test_localize_xy_two_monostatic_known_geometry() -> None:
    # dev0 at (0,0), dev1 at (4,0), target at (2, 1.5)
    target = (2.0, 1.5)
    r0 = float(np.hypot(target[0] - 0.0, target[1] - 0.0))
    r1 = float(np.hypot(target[0] - 4.0, target[1] - 0.0))
    est = localize_xy_two_monostatic_ranges((0.0, 0.0), r0, (4.0, 0.0), r1, y_hint=1.5)
    assert est[0] == pytest.approx(target[0], abs=1e-6)
    assert est[1] == pytest.approx(target[1], abs=1e-6)
    assert position_rmse_xy(est, target) == pytest.approx(0.0, abs=1e-6)


def test_localize_xy_invalid_range_raises() -> None:
    with pytest.raises(ValueError, match="invalid monostatic ranges"):
        localize_xy_two_monostatic_ranges((0.0, 0.0), float("nan"), (1.0, 0.0), 2.0)


def test_compute_1d_cfar_threshold_shape() -> None:
    magnitude = np.abs(np.random.default_rng(0).normal(size=64)) + 1.0
    detector = default_range_cfar_detector()
    threshold = compute_1d_cfar_threshold(magnitude, detector)
    assert threshold.shape == magnitude.shape


def test_grc_processing_params_include_esprit_defaults() -> None:
    params = grc_cooperative_processing_params()
    assert params["esprit_num_sources"] == 1
    assert params["esprit_subarray_size"] == 16
    assert params["esprit_window_size"] == 32


def test_estimate_monostatic_range_esprit_m_returns_finite_peak() -> None:
    vlen = DEFAULT_FFT_LEN * DEFAULT_ZEROPADDING_FAC
    step = cooperative_range_bin_step_m()
    peak_bin = 20
    x = np.arange(vlen, dtype=np.float64)
    profile = (8.0 * np.exp(-0.5 * ((x - peak_bin) / 1.5) ** 2)).astype(np.complex64)
    params = grc_cooperative_processing_params()

    r_esprit = estimate_monostatic_range_esprit_m(
        profile,
        range_bin_step=params["range_bin_step"],
        range_roi=params["range_roi"],
        num_sources=params["esprit_num_sources"],
        subarray_size=params["esprit_subarray_size"],
        window_size=params["esprit_window_size"],
    )

    assert np.isfinite(r_esprit)
    assert abs(r_esprit - peak_bin * step) < step


def test_estimate_monostatic_range_with_and_without_cfar() -> None:
    vlen_range = DEFAULT_FFT_LEN * DEFAULT_ZEROPADDING_FAC
    vlen_divide = vlen_range * DEFAULT_TRANSPOSE_LEN
    rng = np.random.default_rng(1)
    cpi = rng.normal(size=vlen_divide) + 1j * rng.normal(size=vlen_divide)
    cpi = cpi.astype(np.complex64)
    profile = divide_cpi_to_complex_range_profile(cpi)
    params = grc_cooperative_processing_params()
    cfar_detector = default_range_cfar_detector()

    r_plain = estimate_monostatic_range_m(
        profile,
        range_bin_step=params["range_bin_step"],
        range_roi=params["range_roi"],
    )
    r_cfar = estimate_monostatic_range_m(
        profile,
        range_bin_step=params["range_bin_step"],
        range_roi=params["range_roi"],
        cfar_detector=cfar_detector,
    )

    assert np.isfinite(r_plain) or np.isnan(r_plain)
    assert np.isfinite(r_cfar) or np.isnan(r_cfar)
