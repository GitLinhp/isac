"""RT 单基地/双基地速度符号：DD→MUSIC 与 geom.vel_tensor 同号；静态目标链路不受影响。"""

from __future__ import annotations

import pytest
import torch

from isac.data_structures import SystemComponents, SystemParams
from isac.system import System
from isac.utils import load_config, set_random_seed
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
    _, velocities, _ = comps.music_evaluator.estimate(
        spectrum_tensor=h_dd,
        sens_mode=sens_mode,
        num_sources=1,
    )
    assert velocities.numel() > 0
    return float(velocities.reshape(-1)[0].item())


def test_rt_monostatic_cfr_per_tx_velocity_sign_matches_geometry():
    """协作拓扑：单基地 bs1_tx 的 MUSIC 速度符号与 geom.vel_tensor 一致。"""
    from isac.channel import RTChannel

    comps = _build_cooperative_components()
    scene = comps.rt_simulator
    geom = scene.rx_target_tx_geometric
    rg = comps.rg
    channel = comps.channel
    assert isinstance(channel, RTChannel)

    h_pairs = channel.cfr_split(
        rg.num_ofdm_symbols,
        1 / rg.ofdm_symbol_duration,
    )
    mono_idx = geom.tx_names.index("bs1_tx")
    true_v = float(geom.vel_tensor[0, 0, mono_idx].item())
    assert true_v > 0.0

    rx_name = geom.rx_names[0]
    est_v = _music_velocity(
        comps,
        h_pairs[RTChannel.cfr_pair_key(rx_name, "bs1_tx")],
        sens_mode="monostatic",
    )
    v_res = float(comps.sensing_performance.velocity_resolution)
    assert est_v * true_v > 0.0
    assert abs(est_v - true_v) < 2.0 * v_res


def test_rt_bistatic_cfr_per_tx_velocity_sign_matches_geometry():
    """双基地 bs2_tx：``doppler_to_velocity`` 取反后与 ``geom.vel_tensor`` 同号。"""
    from isac.channel import RTChannel

    comps = _build_cooperative_components()
    scene = comps.rt_simulator
    geom = scene.rx_target_tx_geometric
    rg = comps.rg
    channel = comps.channel
    assert isinstance(channel, RTChannel)

    h_pairs = channel.cfr_split(
        rg.num_ofdm_symbols,
        1 / rg.ofdm_symbol_duration,
    )
    bi_idx = geom.tx_names.index("bs2_tx")
    true_v = float(geom.vel_tensor[0, 0, bi_idx].item())
    assert true_v > 0.0

    rx_name = geom.rx_names[0]
    est_v = _music_velocity(
        comps,
        h_pairs[RTChannel.cfr_pair_key(rx_name, "bs2_tx")],
        sens_mode="bistatic",
    )
    v_res = float(comps.sensing_performance.velocity_resolution)
    assert est_v * true_v > 0.0
    assert abs(est_v - true_v) < 2.0 * v_res


def test_static_target_sensing_velocity_stays_positive():
    """非 RT 静态目标：System.sensing 不翻转，MUSIC 速度仍与 CLI 真值同号。"""
    set_random_seed(42)
    config = load_config("simulation/sensing/static_target_simulation.toml")
    system = System(
        config=config,
        device="cpu",
    )
    comps = system.components
    comps.delay_doppler_spectrum.device = torch.device("cpu")

    true_v = float(comps.rcs_scene.target.velocity_mps)
    assert true_v > 0.0

    _, x_rg, x_time = system.transmit()
    y_time = comps.channel(x_rg, x_time, domain="time")
    y_rg = comps.demodulator(y_time).squeeze()

    h_freq = comps.ls_channel_estimator(x_rg, y_rg)
    h_dd = comps.delay_doppler_spectrum(h_freq)
    _, est_velocities, _ = comps.music_evaluator.estimate(
        spectrum_tensor=h_dd,
        sens_mode="monostatic",
        num_sources=1,
        log_peaks=False,
    )
    est_v = float(est_velocities.reshape(-1)[0].item())
    assert est_v * true_v > 0.0
