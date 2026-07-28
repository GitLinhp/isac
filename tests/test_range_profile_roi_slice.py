"""``compute_range_roi`` 与 DEFAULT_RANGE_ROI 测试。"""

from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_RANGE_ROI,
    cooperative_range_bin_step_m,
    grc_cooperative_processing_params,
)
from isac_imp.range_profile_roi_slice import compute_range_roi


def test_default_range_roi_is_zero_to_three_point_five_meters() -> None:
    assert DEFAULT_RANGE_ROI == (0.0, 3.5)


def test_grc_params_range_roi_matches_default() -> None:
    params = grc_cooperative_processing_params()
    assert params["range_roi"] == DEFAULT_RANGE_ROI


def test_compute_range_roi_bin_counts_for_three_point_five_and_five_m() -> None:
    step = cooperative_range_bin_step_m()
    vlen_in = 8192
    _, num_bins_3_5, _ = compute_range_roi(
        range_roi=(0.0, 3.5),
        range_bin_step=step,
        vlen_in=vlen_in,
    )
    _, num_bins_5, _ = compute_range_roi(
        range_roi=(0.0, 5.0),
        range_bin_step=step,
        vlen_in=vlen_in,
    )
    assert num_bins_3_5 == 24
    assert num_bins_5 == 34
    assert num_bins_5 > num_bins_3_5
