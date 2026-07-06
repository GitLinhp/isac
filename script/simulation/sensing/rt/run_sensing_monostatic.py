"""单基地 ISAC 感知评估脚本：端到端仿真链 + MUSIC 谱峰 + 与几何真值对齐的 RMSE 日志。"""

import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import load_config, match_peaks_and_compute_radial_rmse, set_random_seed

SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "sensing_monostatic"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 单基地感知评估")

    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_monostatic.toml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="计算设备类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="信道施加域：frequency 或 time",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="range_velocity",
        choices=["delay_doppler", "range_velocity"],
        help="谱图与 MUSIC 日志 metric",
    )

    return parser.parse_args()


def main() -> None:
    # --- 参数解析 ---
    args = argument_parser()
    set_random_seed(args.seed)

    # --- 加载配置 ---
    config = load_config(args.config_file)
    system = System(
        config=config,
        device=args.device,
    )
    comps = system.components

    # --- 渲染场景 ---
    comps.rt_simulator.render_to_file(
        filename=str(SCRIPT_OUT_DIR / "sensing_monostatic_scene.png")
    )

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道 ---
    domain = args.domain  # 信道施加域
    snr_db = system.params.channel.snr_db  # 信噪比

    y_out = comps.channel(x_rg, x_time, domain=domain, snr_db=snr_db)
    if domain == "time":
        y_rg = comps.demodulator(y_out)
    else:
        y_rg = y_out

    # --- 感知 ---
    comps.sensing_performance()

    h_freq = comps.ls_channel_estimator(x_rg, y_rg)
    # h = comps.moving_target_indication(h, axis=-2)
    h_dd = comps.delay_doppler_spectrum(h_freq)

    comps.delay_doppler_spectrum.visualize(
        file_name=SCRIPT_OUT_DIR / "sensing_monostatic_delay_doppler_spectrum.png",
        metric_mode=args.metric_mode,
        to_db=False,
    )
    est_ranges, est_velocities, _ = comps.music_estimator(
        spectrum_tensor=h_dd,
        metric_mode=args.metric_mode,
        sens_mode="monostatic",
    )
    geom = comps.rt_simulator.rx_target_tx_geometric
    match_peaks_and_compute_radial_rmse(
        est_ranges=est_ranges,
        est_velocities=est_velocities,
        true_ranges=geom.range_tensor,
        true_velocities=geom.vel_tensor,
        label="单基地感知",
    )


if __name__ == "__main__":
    main()
