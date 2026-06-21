import argparse
import math

import torch

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 感知基线")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file", type=str, default="sensing_baseline.toml", help="配置文件路径"
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

    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    set_random_seed(args.seed)
    system = System(args)

    domain = args.domain
    comps = system.components

    script_out_dir = PROJECT_ROOT / "out" / "sensing_baseline"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    system.components.rt_scene.render_to_file(
        filename=script_out_dir / "sensing_baseline_scene.png"
    )
    system.components.rt_scene.get("reflector").velocity = [0, 0, -20]
    system.components.rt_scene.get("bs1_tx").velocity = [30, 0, 0]

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道 ---
    if domain == "frequency":
        y_rg = system.components.channel(x_rg, domain=domain)
        _, h_delay_doppler = system.sensing(x_rg, y_rg)
    elif domain == "time":
        y_time = system.components.channel(x_time, domain=domain)
        _, h_delay_doppler = system.sensing(x_rg, y_time=y_time)
    else:
        raise ValueError(f"不支持的域: {domain}")

    # --- 显示感知结果 ---
    comps.sensing_performance.display_performance()
    comps.delay_doppler_spectrum.visualize(
        offset=20,
        file_name=script_out_dir / "sensing_baseline_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode="delay_doppler",
        backend="matplotlib",
    )
    comps.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode="dd",
    )

    rt_scene = system.components.rt_scene
    print("Delay - LoS Path (ns) :", rt_scene.paths.tau[0, 0, 0] / 1e-9)
    print("Doppler - LoS Path (Hz) :", rt_scene.paths.doppler[0, 0, 0])
    print(
        "Delay - Reflected Path (ns) :",
        rt_scene.paths.tau[0, 0, 1].numpy() / 1e-9,
    )
    print(
        "Doppler - Reflected Path (Hz) :",
        rt_scene.paths.doppler[0, 0, 1],
    )


if __name__ == "__main__":
    main()
