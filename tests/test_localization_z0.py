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
