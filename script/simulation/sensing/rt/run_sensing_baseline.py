import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import load_config, set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 感知基线")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_baseline.toml",
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

    domain = args.domain

    script_out_dir = PROJECT_ROOT / "out" / "sensing_baseline"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    system.components.rt_simulator.render_to_file(
        filename=script_out_dir / "sensing_baseline_scene.png"
    )
    rt = system.components.rt_simulator
    rt.scene.objects["reflector"].velocity = [0, 0, -20]
    rt.transceivers["bs1"].tx.velocity = [30, 0, 0]

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()

    # --- 应用信道 ---
    snr_db = system.params.channel.snr_db
    if domain == "frequency":
        y_rg = system.components.channel(x_rg, domain=domain, snr_db=snr_db)
    elif domain == "time":
        y_time = system.components.channel(x_time, domain=domain, snr_db=snr_db)
        y_rg = system.components.demodulator(y_time).squeeze()
    else:
        raise ValueError(f"不支持的域: {domain}")

    h_dd = system.compute_sensing_spectrum(x_rg, y_rg)
    system.components.sensing_performance()
    system.visualize_sensing_spectrum(
        h_dd,
        file=script_out_dir / "sensing_baseline_delay_doppler_spectrum.png",
        metric_mode="delay_doppler",
    )
    system.estimate_sensing_music(h_dd, metric_mode="delay_doppler")

    rt_simulator = system.components.rt_simulator
    print("Delay - LoS Path (ns) :", rt_simulator.paths.tau[0, 0, 0] / 1e-9)
    print("Doppler - LoS Path (Hz) :", rt_simulator.paths.doppler[0, 0, 0])
    print(
        "Delay - Reflected Path (ns) :",
        rt_simulator.paths.tau[0, 0, 1].numpy() / 1e-9,
    )
    print(
        "Doppler - Reflected Path (Hz) :",
        rt_simulator.paths.doppler[0, 0, 1],
    )


if __name__ == "__main__":
    main()
