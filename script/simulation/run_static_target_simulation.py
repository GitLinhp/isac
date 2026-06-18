"""静态目标 ISAC 感知评估：gr-radar 风格点目标时域信道 + MUSIC 谱峰 + RMSE 日志。"""

import argparse

import torch

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import match_peaks_and_compute_radial_rmse, set_random_seed


def _first_float(values: float | list[float] | tuple[float, ...]) -> float:
    if isinstance(values, (list, tuple)):
        return float(values[0])
    return float(values)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 静态目标点散射信道感知评估"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="static_target_simulation.toml",
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
    system = System(args)

    static_sim = system.components.static_target_sim
    if static_sim is None:
        raise ValueError("channel.type='rcs' 要求已构建 static_target 组件")

    script_out_dir = PROJECT_ROOT / "out" / "static_target_simulation"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    comps = system.components
    comps.delay_doppler_spectrum.device = torch.device(args.device)

    params = static_sim.params
    range_m = _first_float(params.range_m)
    velocity_mps = _first_float(params.velocity_mps)

    print(
        f"静态目标参数: range={range_m:.2f} m, velocity={velocity_mps:.2f} m/s, "
        f"samp_rate={params.samp_rate}, center_freq={params.center_freq:.3e} Hz"
    )

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道（RCS 点目标仅时域）---
    y_time = system.apply_channel(x_time, domain="time")

    # --- 感知 ---
    result = system.sensing(x_rg, y_time=y_time)

    # --- 显示感知结果 ---
    comps.sensing_performance.display_performance()
    comps.delay_doppler_spectrum.visualize(
        offset=50,
        file_name=script_out_dir / "static_target_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
    )

    device = torch.device(args.device)
    true_ranges = torch.tensor([range_m], dtype=torch.float64, device=device)
    true_velocities = torch.tensor([velocity_mps], dtype=torch.float64, device=device)

    est_ranges, est_velocities, _ = comps.music_estimator(
        spectrum_tensor=result.h_delay_doppler,
        metric_mode=args.metric_mode,
    )

    match_peaks_and_compute_radial_rmse(
        est_ranges=est_ranges,
        est_velocities=est_velocities,
        true_ranges=true_ranges,
        true_velocities=true_velocities,
        label="静态目标仿真",
    )


if __name__ == "__main__":
    main()
