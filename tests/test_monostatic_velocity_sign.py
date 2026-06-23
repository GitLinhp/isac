"""RT 单基地速度符号：DD→MUSIC 与 geom.vel_tensor 同号；静态目标链路不受影响。"""

from __future__ import annotations

import argparse

import pytest
import torch

from isac.data_structures import SystemComponents, SystemParams
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.channel_paths import paths_cfr_per_tx_torch


def _build_cooperative_components(device: str = "cpu") -> SystemComponents:
    set_random_seed(42)
    params = SystemParams.from_dict(
        load_config("simulation/sensing/sensing_cooperative.toml")
    )
    return SystemComponents.build_from_params(params, device=device)


def _music_velocity(
    comps: SystemComponents,
    h: torch.Tensor,
    *,
    sens_mode: str = "monostatic",
) -> float:
    h_work = h
    if comps.moving_target_indication is not None:
        h_work = comps.moving_target_indication(h_work, axis=-2)
    h_dd = comps.delay_doppler_spectrum(h_work)
    _, velocities, _ = comps.music_estimator(
        spectrum_tensor=h_dd,
        sens_mode=sens_mode,
        num_sources=1,
    )
    assert velocities.numel() > 0
    return float(velocities.reshape(-1)[0].item())


def test_rt_monostatic_cfr_per_tx_velocity_sign_matches_geometry():
    """协作拓扑：单基地 bs1_tx 的 MUSIC 速度符号与 geom.vel_tensor 一致。"""
    comps = _build_cooperative_components()
    scene = comps.rt_scene
    geom = scene.rx_target_tx_geometric
    rg = comps.rg

    h_per_tx = paths_cfr_per_tx_torch(rg, scene, device=torch.device("cpu"))
    mono_idx = geom.tx_names.index("bs1_tx")
    true_v = float(geom.vel_tensor[0, 0, mono_idx].item())
    assert true_v > 0.0

    est_v = _music_velocity(comps, h_per_tx["bs1_tx"], sens_mode="monostatic")
    v_res = float(comps.sensing_performance.velocity_resolution)
    assert est_v * true_v > 0.0
    assert abs(est_v - true_v) < 2.0 * v_res


def test_rt_bistatic_cfr_per_tx_not_auto_flipped():
    """双基地 bs2_tx 不施加单基地符号校正（仍与 geom 可能异号，但不应被 flip 成正值）。"""
    comps = _build_cooperative_components()
    scene = comps.rt_scene
    geom = scene.rx_target_tx_geometric
    rg = comps.rg

    h_per_tx = paths_cfr_per_tx_torch(rg, scene, device=torch.device("cpu"))
    bi_idx = geom.tx_names.index("bs2_tx")
    true_v = float(geom.vel_tensor[0, 0, bi_idx].item())
    assert true_v > 0.0

    est_v = _music_velocity(comps, h_per_tx["bs2_tx"], sens_mode="bistatic")
    assert est_v * true_v < 0.0


def test_static_target_sensing_velocity_stays_positive():
    """非 RT 静态目标：System.sensing 不翻转，MUSIC 速度仍与 CLI 真值同号。"""
    set_random_seed(42)
    args = argparse.Namespace(
        batch_size=1,
        config_file="simulation/sensing/static_target_simulation.toml",
        device="cpu",
        seed=42,
    )
    system = System(args)
    comps = system.components
    comps.delay_doppler_spectrum.device = torch.device("cpu")

    true_v = float(comps.static_target_sim.params.velocity_mps)
    assert true_v > 0.0

    _, x_rg, x_time = system.transmit()
    y_time = comps.channel(x_time, domain="time")
    y_rg = comps.demodulator(y_time).squeeze()

    result = system.sensing(
        x_rg,
        y_rg,
        evaluate=True,
        display_performance=False,
        display_geometry=False,
        run_music=True,
        compute_rmse=False,
        spectrum_file=None,
        sens_mode="monostatic",
    )
    est_v = float(result.est_velocities.reshape(-1)[0].item())
    assert est_v * true_v > 0.0
