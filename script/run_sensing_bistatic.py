"""双基地 ISAC 感知评估脚本：端到端仿真链 + MUSIC 谱峰 + 与几何真值对齐的 RMSE 日志。

管线概要
--------
1. 按 ``--domain`` 在频域或时域施加信道，经 ``estimate_channel`` 得到 CFR 相关估计。
2. 计算时延–多普勒谱；再沿多普勒维（``dim=0``）对谱施加 **MTI**（``system.moving_target_indication``），并导出 ``out/sensing_bistatic/sensing_bistatic_delay_doppler_spectrum.png``。
3. MUSIC 提取峰；须传入 ``sens_mode='bistatic'``，使 bin→物理量与折叠路径 ``||T-X||+||R-T||`` 及配套多普勒尺度一致（勿沿用默认单基地的 ``τ·c/2``、``v∝f_d/(2f_c)``）。
4. 用 ``RTScene.rx_target_tx_geometric`` 的 ``range_tensor`` / ``vel_tensor`` 作真值；``select_peak_and_log_radial_rmse`` 做匈牙利匹配并记录 RMSE。

与 ``run_dataset_collection.py`` 中 ``bistatic_eval`` 的 ``sens_mode`` 用法一致。
"""

import argparse

from isac import PROJECT_ROOT
from isac.system import System
from isac.utils import select_peak_and_log_radial_rmse, set_random_seed


def argument_parser() -> argparse.Namespace:
    """解析双基地感知评估所需的设备、随机种子、信道域与速度反演模型。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 双基地感知评估")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file", type=str, default="sensing_bistatic.toml", help="配置文件路径"
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
        help="谱图与 MUSIC 日志 metric：delay_doppler 用时延 (ns) / 多普勒 (Hz)；range_velocity 用距离 (m) / 速度 (m/s)",
    )

    return parser.parse_args()


def main() -> None:
    """构建系统、跑一条双基地感知链，并将估计与 RT 场景真值对比后写入日志。"""
    args = argument_parser()
    set_random_seed(args.seed)
    system = System(args)

    scene = system.components.rt_scene

    domain = args.domain

    # 脚本专属子目录
    script_out_dir = PROJECT_ROOT / "out" / "sensing_bistatic"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    # 显示感知性能
    sensing_perf = system.components.sensing_performance
    sensing_perf.display_performance()

    # 渲染场景到文件
    system.components.rt_scene.render_to_file(
        filename=script_out_dir / "sensing_bistatic_scene.png"
    )

    # --- OFDM 参考信号与信道 ---
    x_rg = system.tx_symbols_to_resource_grid()

    if domain == "frequency":
        y_rg = system.apply_channel(x_rg, domain=domain)
    elif domain == "time":
        x_time = system.components.modulator(x_rg)
        y_time = system.apply_channel(x_time, domain=domain)
        y_rg = system.components.demodulator(y_time)
    else:
        raise ValueError(f"不支持的域: {domain}")

    y_rg = y_rg.squeeze()

    # 信道估计
    h = system.estimate_channel(x_rg, y_rg)

    # MTI：沿符号维（dim=0，对应慢时间）抑制零多普勒杂波后再作时延–多普勒谱
    h = system.moving_target_indication(h, axis=-2)

    # 计算时延–多普勒谱
    h_delay_doppler = system.components.delay_doppler_spectrum(h)

    # 绘制时延–多普勒谱图
    system.components.delay_doppler_spectrum.visualize(
        offset=100,
        file_name=script_out_dir / "sensing_bistatic_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
    )

    # --- 真值：几何折叠路径长度与双基地距离变化率 ---
    geom = scene.rx_target_tx_geometric
    geom.display()
    true_ranges = geom.range_tensor
    true_velocities = geom.vel_tensor

    # --- MUSIC：与真值同一尺度须 sens_mode=bistatic（否则约为单基地换算，距离/速度约偏小二倍）---
    est_ranges, est_velocities, _ = system.components.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode=args.metric_mode,
        sens_mode="bistatic",
    )

    select_peak_and_log_radial_rmse(
        est_ranges=est_ranges,
        est_velocities=est_velocities,
        true_ranges=true_ranges,
        true_velocities=true_velocities,
        log_prefix="双基地感知",
    )


if __name__ == "__main__":
    import sionna

    sionna.phy.config.device = "cpu"
    print(sionna.phy.config.device)

    main()
