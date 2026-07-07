"""数据采集：位姿更新后 h_dd 一致性集成测试。"""

from __future__ import annotations

import numpy as np
import pytest
import sionna.phy.config
import torch
from scipy.constants import speed_of_light as c

from isac import PROJECT_ROOT
from isac.collection import RoiKinematicsSampler
from isac.sensing.geometry import delay_to_range, doppler_to_velocity
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.numerical import cartesian_direction_to_yaw_pitch_roll

_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@pytest.fixture
def collection_system() -> System:
    sionna.phy.config.device = "cpu"
    set_random_seed(42)
    return System(config=load_config(_CONFIG), device="cpu")


def _simulate_episode_h_dd(
    system: System,
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    snr_db: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """更新目标位姿并跑完整采集链；过滤未通过时返回 ``None``。"""
    rt_simulator = system.components.rt_simulator
    _, target = next(iter(rt_simulator.rt_targets.items()))
    ori = cartesian_direction_to_yaw_pitch_roll(vel.reshape(1, 3))[0]

    target(position=pos, velocity=vel, orientation=ori)
    if not rt_simulator.scene_filter(pos):
        return None
    rt_simulator.paths(update=True)
    if not rt_simulator.paths_intersect_target(target):
        return None

    geom = rt_simulator.rx_target_tx_geometric
    true_range = geom.range_tensor[0, 0, 0]
    true_velocity = geom.vel_tensor[0, 0, 0]
    comps = system.components
    _, x_rg, x_time = system.transmit()
    y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)
    h_freq = comps.ls_channel_estimator(x_rg, y_rg)
    h_dd = comps.delay_doppler_spectrum(h_freq)
    return h_dd, true_range, true_velocity


def _peak_range_velocity_from_h_dd(
    system: System,
    h_dd: torch.Tensor,
) -> tuple[float, float]:
    """ROI 裁剪 h_dd 主峰 → 单基地距离/径向速度。"""
    dd = system.components.delay_doppler_spectrum
    assert dd is not None
    sp = dd.sensing_performance
    n_sym = sp.rg.num_ofdm_symbols
    dop_start, _, delay_start, _ = dd.bin_slices(
        torch.zeros(n_sym, sp.rg.fft_size, dtype=torch.complex64)
    )

    mag = h_dd.abs()
    flat_idx = int(mag.argmax().item())
    d_local, t_local = divmod(flat_idx, int(mag.shape[-1]))
    delay_bin = delay_start + t_local
    dop_bin = dop_start + d_local

    delay_resolution = 2.0 * sp.range_resolution / c
    doppler_resolution = sp.velocity_resolution * 2.0 * sp.carrier_frequency / c
    tau_s = delay_bin * delay_resolution
    fd_hz = (dop_bin - n_sym // 2) * doppler_resolution

    range_m = float(delay_to_range(tau_s, sp.carrier_frequency, "monostatic"))
    vel_mps = float(
        doppler_to_velocity(fd_hz, sp.carrier_frequency, "monostatic")
    )
    return range_m, vel_mps


def test_h_dd_changes_with_pose_update(collection_system: System) -> None:
    """不同位姿下 h_dd 不应完全相同（无跨 episode 缓存）。"""
    pos1 = np.array([1.0, 1.5, 0.0], dtype=np.float64)
    vel1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    pos2 = np.array([-1.0, -1.5, 0.0], dtype=np.float64)
    vel2 = np.array([0.0, 2.0, 0.0], dtype=np.float64)

    result1 = _simulate_episode_h_dd(
        collection_system, pos=pos1, vel=vel1, snr_db=None
    )
    result2 = _simulate_episode_h_dd(
        collection_system, pos=pos2, vel=vel2, snr_db=None
    )
    if result1 is None or result2 is None:
        pytest.skip("固定位姿未通过采集过滤")
    h_dd1, _, _ = result1
    h_dd2, _, _ = result2

    diff_norm = torch.linalg.norm(h_dd1 - h_dd2).item()
    assert diff_norm > 0.0, "不同位姿的 h_dd 完全相同，可能存在陈旧缓存"


def test_h_dd_peak_aligns_with_geometry_truth(collection_system: System) -> None:
    """h_dd 主峰与 rx_target_tx_geometric 在 3 bin 容差内对齐。"""
    pos = np.array([1.2, -0.8, 0.0], dtype=np.float64)
    vel = np.array([1.5, -0.5, 0.0], dtype=np.float64)

    result = _simulate_episode_h_dd(
        collection_system, pos=pos, vel=vel, snr_db=None
    )
    if result is None:
        pytest.skip("固定位姿未通过采集过滤")
    h_dd, true_range, true_velocity = result
    est_range, est_vel = _peak_range_velocity_from_h_dd(collection_system, h_dd)

    sp = collection_system.components.delay_doppler_spectrum.sensing_performance
    range_tol = 3.0 * sp.range_resolution
    vel_tol = 3.0 * sp.velocity_resolution

    assert est_range == pytest.approx(float(true_range.item()), abs=range_tol)
    assert est_vel == pytest.approx(float(true_velocity.item()), abs=vel_tol)


def test_consecutive_sampler_episodes_have_distinct_h_dd(
    collection_system: System,
) -> None:
    """连续 pop 5 条采样 kinematics，至少 2 条采纳 episode 的 h_dd 互不相同。"""
    sampler = RoiKinematicsSampler(
        roi=[-2.5, 2.5, -4.5, 4.5],
        position_sampling_mode="uniform",
        speed_range=[0.1, 3.0],
        speed_sampling_mode="uniform",
        num_samples=50,
    )

    h_dd_list: list[torch.Tensor] = []
    attempts = 0
    while len(h_dd_list) < 5 and attempts < 50:
        if len(sampler) == 0:
            break
        pos, vel, _ = sampler.pop()
        attempts += 1
        result = _simulate_episode_h_dd(
            collection_system, pos=pos, vel=vel, snr_db=None
        )
        if result is None:
            continue
        h_dd, _, _ = result
        h_dd_list.append(h_dd)

    assert len(h_dd_list) >= 2, "未能采纳足够 episode 用于差分检查"
    distinct = 0
    for i in range(len(h_dd_list)):
        for j in range(i + 1, len(h_dd_list)):
            if torch.linalg.norm(h_dd_list[i] - h_dd_list[j]).item() > 0.0:
                distinct += 1
    assert distinct >= 1, "连续 episode 的 h_dd 全部相同，可能存在陈旧缓存"
