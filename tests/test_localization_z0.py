"""z=0 平面协同定位：两圆求交与 sensing_cooperative 拓扑闭式解。"""

import numpy as np
import pytest

from isac.sensing.localization import (
    ground_circle_radius_sq,
    intersect_circles_xy,
    localize_xy_z0_colocated_tx_mono_bistatic,
    position_rmse_xy,
    select_xy_solution,
)


def test_ground_circle_radius_sq_cooperative_heights():
    # r_m=36.06, z_rx=30 -> R^2 = 36.06^2 - 900
    r_sq = ground_circle_radius_sq(36.06, 30.0, z_target_m=0.0)
    assert r_sq == pytest.approx(36.06**2 - 900.0, rel=1e-6)


def test_localize_xy_truth_ranges_cooperative_topology():
    """sensing_cooperative.toml 真值量测：r_m≈36.06, r_b≈78.48 → (-5, 0)。"""
    mono = [-25.0, 0.0, 30.0]
    bi = [25.0, 0.0, 30.0]
    x, y = localize_xy_z0_colocated_tx_mono_bistatic(
        mono,
        bi,
        r_mono_slant_m=36.06,
        r_bistatic_sum_m=78.48,
        z_target_m=0.0,
        y_hint=0.0,
        tx_pos=mono,
    )
    assert x == pytest.approx(-5.0, abs=1e-2)
    assert y == pytest.approx(0.0, abs=1e-2)


def test_localize_xy_music_ranges_cooperative_topology():
    """终端 MUSIC 估计：r_m≈34.16, r_b≈78.07。"""
    mono = [-25.0, 0.0, 30.0]
    bi = [25.0, 0.0, 30.0]
    x, y = localize_xy_z0_colocated_tx_mono_bistatic(
        mono,
        bi,
        r_mono_slant_m=34.16,
        r_bistatic_sum_m=78.07,
        z_target_m=0.0,
        y_hint=0.0,
        tx_pos=mono,
    )
    assert x == pytest.approx(-7.6, abs=0.15)
    assert abs(y) < 0.15


def test_intersect_circles_two_solutions_y_disambiguation():
    c1 = (0.0, 0.0)
    c2 = (6.0, 0.0)
    r1_sq = 25.0
    r2_sq = 25.0
    sols = intersect_circles_xy(c1, r1_sq, c2, r2_sq)
    assert len(sols) == 2
    picked = select_xy_solution(sols, y_hint=-1.0)
    assert picked[1] < 0.0


def test_horn_beam_prefers_origin_side_cooperative_geometry():
    """dev0=(0,-2), dev1=(-2,0), horns aim origin → pick near-origin intersection."""
    from isac.sensing.localization import (
        horn_beam_score_xy,
        localize_xy_two_monostatic_ranges,
    )

    dev0 = (0.0, -2.0)
    dev1 = (-2.0, 0.0)
    # True target near origin; ranges yield two intersections mirrored across baseline.
    true_xy = (-0.4, -0.4)
    r0 = float(np.hypot(true_xy[0] - dev0[0], true_xy[1] - dev0[1]))
    r1 = float(np.hypot(true_xy[0] - dev1[0], true_xy[1] - dev1[1]))
    sols = intersect_circles_xy(dev0, r0 * r0, dev1, r1 * r1)
    assert len(sols) == 2

    horn_pick = select_xy_solution(sols, station0_xy=dev0, station1_xy=dev1)
    min_abs_y = select_xy_solution(sols)  # no stations → |y| min
    assert horn_beam_score_xy(horn_pick, dev0, dev1) >= horn_beam_score_xy(
        sols[0] if horn_pick == sols[1] else sols[1],
        dev0,
        dev1,
    )
    # Horn pick should be the Euclidean-nearest to truth for this geometry.
    best = min(sols, key=lambda p: (p[0] - true_xy[0]) ** 2 + (p[1] - true_xy[1]) ** 2)
    assert horn_pick[0] == pytest.approx(best[0], abs=1e-9)
    assert horn_pick[1] == pytest.approx(best[1], abs=1e-9)

    est = localize_xy_two_monostatic_ranges(dev0, r0, dev1, r1)
    assert est[0] == pytest.approx(best[0], abs=1e-9)
    assert est[1] == pytest.approx(best[1], abs=1e-9)

    est_miny = localize_xy_two_monostatic_ranges(
        dev0, r0, dev1, r1, use_horn_disambiguation=False
    )
    assert est_miny[0] == pytest.approx(min_abs_y[0], abs=1e-9)
    assert est_miny[1] == pytest.approx(min_abs_y[1], abs=1e-9)


def test_quasi_monostatic_ellipse_recovers_truth():
    from isac.sensing.localization import (
        localize_xy_two_monostatic_ranges,
        localize_xy_two_quasi_monostatic_path_sums,
        path_sum_xy,
    )

    # Fixed ±5 cm geometry (independent of pipeline defaults after offset sweep).
    tx0, rx0 = (0.05, -2.0), (-0.05, -2.0)
    tx1, rx1 = (-2.0, 0.05), (-2.0, -0.05)
    mid0, mid1 = (0.0, -2.0), (-2.0, 0.0)

    true_xy = (-0.4, -0.4)
    r0 = 0.5 * path_sum_xy(true_xy, tx0, rx0)
    r1 = 0.5 * path_sum_xy(true_xy, tx1, rx1)
    est = localize_xy_two_quasi_monostatic_path_sums(tx0, rx0, r0, tx1, rx1, r1)
    assert est[0] == pytest.approx(true_xy[0], abs=1e-3)
    assert est[1] == pytest.approx(true_xy[1], abs=1e-3)

    circ = localize_xy_two_monostatic_ranges(mid0, r0, mid1, r1)
    assert position_rmse_xy(est, circ) < 0.05


def test_localize_raises_when_tx_not_colocated():
    mono = [-25.0, 0.0, 30.0]
    bi = [25.0, 0.0, 30.0]
    with pytest.raises(ValueError, match="未共址"):
        localize_xy_z0_colocated_tx_mono_bistatic(
            mono,
            bi,
            r_mono_slant_m=36.0,
            r_bistatic_sum_m=78.0,
            tx_pos=[100.0, 0.0, 30.0],
        )


def test_position_rmse_xy():
    assert position_rmse_xy((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
