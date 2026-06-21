"""单基地 ISAC 感知评估脚本：端到端仿真链 + MUSIC 谱峰 + 与几何真值对齐的 RMSE 日志。"""

import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import match_peaks_and_compute_radial_rmse, set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 单基地感知评估")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="sensing_monostatic.toml",
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
    args = argument_parser()
    set_random_seed(args.seed)
    system = System(args)

    scene = system.components.rt_scene
    domain = args.domain
    comps = system.components

    script_out_dir = PROJECT_ROOT / "out" / "sensing_monostatic"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    scene.render_to_file(filename=script_out_dir / "sensing_monostatic_scene.png")

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道 ---
    if domain == "frequency":
        y_rg = system.apply_channel(x_rg, domain=domain)
        _, h_delay_doppler = system.sensing(x_rg, y_rg)
    elif domain == "time":
        y_time = system.apply_channel(x_time, domain=domain)
        _, h_delay_doppler = system.sensing(x_rg, y_time=y_time)
    else:
        raise ValueError(f"不支持的域: {domain}")

    # --- 显示感知结果 ---
    comps.sensing_performance.display_performance()
    comps.delay_doppler_spectrum.visualize(
        offset=50,
        file_name=script_out_dir / "sensing_monostatic_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
    )

    geom = scene.rx_target_tx_geometric
    geom.display()

    est_ranges, est_velocities, _ = comps.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode=args.metric_mode,
    )

    match_peaks_and_compute_radial_rmse(
        est_ranges=est_ranges,
        est_velocities=est_velocities,
        true_ranges=geom.range_tensor,
        true_velocities=geom.vel_tensor,
        label="单基地感知",
    )


if __name__ == "__main__":
    main()
