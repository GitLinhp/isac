import argparse
import math

import torch

from isac import PROJECT_ROOT
from isac.channel.channel import Channel
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


def _print_dd_spectrum_report(
    dd_noisy: torch.Tensor,
    dd_clean: torch.Tensor,
    *,
    snr_db: float,
) -> None:
    """打印时延–多普勒谱峰值/底噪与处理增益诊断。"""
    a_noisy = torch.abs(dd_noisy).flatten()
    a_clean = torch.abs(dd_clean).flatten()
    peak_n = float(a_noisy.max().item())
    peak_c = float(a_clean.max().item())
    med_n = float(a_noisy.median().item())
    med_c = float(a_clean.median().item())
    dr_n = 20.0 * math.log10(peak_n / med_n) if med_n > 0 else float("inf")
    dr_c = 20.0 * math.log10(peak_c / med_c) if med_c > 0 else float("inf")

    print("=== 时延–多普勒谱诊断 ===")
    print(
        f"无噪谱: peak={peak_c:.6f}, median={med_c:.6e}, "
        f"峰/中值={dr_c:.1f} dB"
    )
    print(
        f"加噪谱: peak={peak_n:.6f}, median={med_n:.6e}, "
        f"峰/中值={dr_n:.1f} dB"
    )
    print(
        f"底噪抬升 (median): {10 * math.log10(med_n / med_c):.1f} dB "
        f"(配置接收 SNR={snr_db:.1f} dB，FFT 后仍可能有 ~60 dB 处理增益)"
    )
    print(
        "提示: to_db=False 线性作图时，median≪peak 的底噪在 3D 图上几乎不可见；"
        "低 SNR 建议 to_db=True。"
    )


def main() -> None:
    args = argument_parser()  # 解析命令行参数
    set_random_seed(args.seed)  # 设置随机种子
    system = System(args)  # 创建系统实例

    system.components.sensing_performance.display_performance()

    script_out_dir = PROJECT_ROOT / "out" / "sensing_baseline"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    system.components.rt_scene.render_to_file(
        filename=script_out_dir / "sensing_baseline_scene.png"
    )

    domain = args.domain  # 获取域
    system.components.rt_scene.get("reflector").velocity = [0, 0, -20]  # 设置反射器速度
    system.components.rt_scene.get("bs1_tx").velocity = [30, 0, 0]  # 设置基站1发射速度

    x_rg = system.tx_symbols_to_resource_grid()
    ch = system.components.channel
    snr_db = system.params.channel.snr_db
    no_comm = ch.noise_power_from_snr_db(
        snr_db,
        system.params.qam.num_bits_per_symbol,
        system.params.channel.coderate,
        system.components.rg,
    )

    if domain == "frequency":
        y_clean = ch(x_rg, domain=domain, snr_db=None)
        y_rg = system.apply_channel(x_rg, domain=domain)

    elif domain == "time":
        x_time = system.components.modulator(x_rg)
        y_time_clean = ch(x_time, domain=domain, snr_db=None)
        y_time = system.apply_channel(x_time, domain=domain)
        y_clean = system.components.demodulator(y_time_clean)
        y_rg = system.components.demodulator(y_time)

    else:
        raise ValueError(f"不支持的域: {domain}")

    h = system.estimate_channel(x_rg, y_rg)
    h_clean = system.estimate_channel(x_rg, y_clean)

    h_delay_doppler = system.components.delay_doppler_spectrum(h)
    dd_clean = system.components.delay_doppler_spectrum(h_clean)
    _print_dd_spectrum_report(h_delay_doppler, dd_clean, snr_db=snr_db)

    system.components.delay_doppler_spectrum.visualize(
        offset=20,
        file_name=script_out_dir / "sensing_baseline_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode="delay_doppler",
        backend="matplotlib",
    )

    system.components.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode="dd",
    )

    print("Delay - LoS Path (ns) :", system.components.rt_scene.paths.tau[0, 0, 0] / 1e-9)
    print("Doppler - LoS Path (Hz) :", system.components.rt_scene.paths.doppler[0, 0, 0])

    print(
        "Delay - Reflected Path (ns) :",
        system.components.rt_scene.paths.tau[0, 0, 1].numpy() / 1e-9,
    )
    print(
        "Doppler - Reflected Path (Hz) :",
        system.components.rt_scene.paths.doppler[0, 0, 1],
    )


if __name__ == "__main__":
    main()
