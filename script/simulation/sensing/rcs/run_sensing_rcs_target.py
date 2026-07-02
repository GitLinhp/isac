"""静态目标 ISAC 感知评估：gr-radar 风格点目标时域信道 + MUSIC 谱峰 + RMSE 日志。"""

import argparse

import torch

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import load_config, set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 静态目标点散射信道感知评估"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/static_target_simulation.toml",
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
    config = load_config(args.config_file)
    system = System(
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )

    rcs_scene = system.components.rcs_scene
    if rcs_scene is None:
        raise ValueError("channel.type='rcs' 要求已构建 rcs_scene 组件")

    script_out_dir = PROJECT_ROOT / "out" / "static_target_simulation"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    comps = system.components
    comps.delay_doppler_spectrum.device = torch.device(args.device)

    target = rcs_scene.target
    range_m = target.range_m
    velocity_mps = target.velocity_mps
    samp_rate = system.params.ofdm.samp_rate
    center_freq = system.params.carrier_frequency

    print(
        f"静态目标参数: range={range_m:.2f} m, velocity={velocity_mps:.2f} m/s, "
        f"samp_rate={samp_rate}, center_freq={center_freq:.3e} Hz"
    )

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道（RCS 点目标仅时域）---
    snr_db = system.params.channel.snr_db
    y_time = system.components.channel(x_time, domain="time", snr_db=snr_db)
    y_rg = system.components.demodulator(y_time).squeeze()

    # --- 感知 ---
    device = torch.device(args.device)
    true_ranges = torch.tensor([range_m], dtype=torch.float64, device=device)
    true_velocities = torch.tensor([velocity_mps], dtype=torch.float64, device=device)

    h_dd = system.compute_sensing_spectrum(x_rg, y_rg)
    system.display_sensing_performance()
    system.visualize_sensing_spectrum(
        h_dd,
        file=script_out_dir / "static_target_delay_doppler_spectrum.png",
        metric_mode=args.metric_mode,
    )
    music = system.estimate_sensing_music(h_dd, metric_mode=args.metric_mode)
    system.evaluate_sensing_rmse(
        music,
        true_ranges=true_ranges,
        true_velocities=true_velocities,
        label="静态目标仿真",
    )


if __name__ == "__main__":
    main()
