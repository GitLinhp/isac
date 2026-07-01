"""双基地 ISAC 感知评估脚本：端到端仿真链 + MUSIC 谱峰 + 与几何真值对齐的 RMSE 日志。"""

import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import load_config, set_random_seed


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 双基地感知评估")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file", type=str, default="simulation/sensing/sensing_bistatic.toml", help="配置文件路径"
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
        default="time",
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
    config = load_config(args.config_file)
    system = System(
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )

    scene = system.components.rt_simulator
    domain = args.domain

    script_out_dir = PROJECT_ROOT / "out" / "sensing_bistatic"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    scene.render_to_file(filename=script_out_dir / "sensing_bistatic_scene.png")

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

    system.sensing(
        x_rg,
        y_rg,
        apply_mti=True,
        evaluate=True,
        metric_mode=args.metric_mode,
        sens_mode="bistatic",
        spectrum_file=script_out_dir / "sensing_bistatic_delay_doppler_spectrum.png",
        visualize_offset=100,
        label="双基地感知",
    )


if __name__ == "__main__":
    main()
