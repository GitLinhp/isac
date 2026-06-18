"""静态目标 ISAC 感知评估：gr-radar 风格点目标时域信道 + MUSIC 谱峰 + RMSE 日志。

管线概要
--------
1. 生成 OFDM 参考 ``x_rg``，经 ``modulator`` 得时域发射 IQ。
2. 用 ``static_target_simulator`` 施加点目标散射（替代 RT 射线追踪信道）。
3. 按接收端 SNR 叠加 AWGN，``demodulator`` 回频域后经 ``estimate_channel`` 得 CFR 估计。
4. 计算时延–多普勒谱并导出 ``out/static_target_simulation/static_target_delay_doppler_spectrum.png``。
5. MUSIC 提取峰；真值来自 CLI ``--range_m`` / ``--velocity_mps``，``match_peaks_and_compute_radial_rmse`` 做匈牙利匹配并计算 RMSE。

与 ``run_sensing_monostatic.py`` 的区别
---------------------------------------
- 信道：``static_target_simulator`` 点散射（对齐 gr-radar ``static_target_simulator_cc``），无 Sionna RT。
- 施加域：固定**时域**（仿真器仅接受 IQ 样点流，不支持 ``--domain``）。
- 真值：CLI 目标参数，而非 ``RTScene.rx_target_tx_geometric``。
- 配置：``static_target_simulation.toml``，无 ``[rt_scene]`` 段。

「static」指点散射物理模型，目标仍可配置非零 ``velocity_mps`` 以产生多普勒。
"""

import argparse

import torch

from isac import PROJECT_ROOT
from isac.channel import StaticTargetParams, StaticTargetSimulator
from isac.system import System
from isac.utils import match_peaks_and_compute_radial_rmse, set_random_seed


def argument_parser() -> argparse.Namespace:
    """解析静态目标仿真所需的系统、目标与仿真器参数。"""
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 静态目标点散射信道感知评估"
    )

    # --- 系统 / 感知通用 ---
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="static_target_simulation.toml",
        help="配置文件路径（OFDM 帧结构、SNR、MUSIC 等；无 RT 场景）",
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
        help="随机种子（ZC 源、AWGN、随机相位等）",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="range_velocity",
        choices=["delay_doppler", "range_velocity"],
        help="谱图与 MUSIC 日志 metric：delay_doppler 用时延 (ns) / 多普勒 (Hz)；range_velocity 用距离 (m) / 速度 (m/s)",
    )

    # --- 点目标几何与散射（亦为 RMSE 真值来源）---
    parser.add_argument("--range_m", type=float, default=95.0, help="目标径向距离 (m)")
    parser.add_argument(
        "--velocity_mps", type=float, default=5.0, help="目标径向速度 (m/s)"
    )
    parser.add_argument("--rcs", type=float, default=1e25, help="雷达散射截面")
    parser.add_argument(
        "--azimuth_deg", type=float, default=0.0, help="目标方位角 (deg)"
    )
    parser.add_argument(
        "--position_rx_m", type=float, default=0.0, help="接收天线位置 (m)"
    )

    # --- gr-radar 仿真器开关 ---
    parser.add_argument(
        "--self_coupling_db",
        type=float,
        default=-10.0,
        help="自耦合幅度 (dB)；发射泄漏叠加在回波上",
    )
    parser.add_argument(
        "--no_self_coupling",
        action="store_true",
        help="关闭自耦合（DD 谱零多普勒附近强峰通常来自自耦合）",
    )
    parser.add_argument(
        "--no_rndm_phaseshift",
        action="store_true",
        help="关闭随机相位（便于复现/调试）",
    )

    return parser.parse_args()


def main() -> None:
    """构建系统、跑静态目标时域信道感知链，并将估计与 CLI 真值对比后写入日志。"""
    args: argparse.Namespace = argument_parser()
    set_random_seed(args.seed)
    system = System(args)

    script_out_dir = PROJECT_ROOT / "out" / "static_target_simulation"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    # 打印距离/速度分辨率、最大探测范围等（由 TOML 中 OFDM 帧参数决定）
    system.components.sensing_performance.display_performance()

    # DD 谱与 MUSIC 需与 --device 一致，避免 h 在 CPU 而谱算子在 CUDA
    device = torch.device(args.device)
    system.components.delay_doppler_spectrum.device = device

    # samp_rate 取资源栅格带宽，与 modulator 输出样点率一致
    rg = system.components.rg
    params = StaticTargetParams(
        range_m=args.range_m,
        velocity_mps=args.velocity_mps,
        rcs=args.rcs,
        azimuth_deg=args.azimuth_deg,
        position_rx_m=(args.position_rx_m,),
        samp_rate=int(rg.bandwidth),
        center_freq=system.params.carrier_frequency,
        self_coupling_db=args.self_coupling_db,
        rndm_phaseshift=not args.no_rndm_phaseshift,
        self_coupling=not args.no_self_coupling,
    )

    print(
        f"静态目标参数: range={args.range_m:.2f} m, velocity={args.velocity_mps:.2f} m/s, "
        f"samp_rate={params.samp_rate}, center_freq={params.center_freq:.3e} Hz"
    )

    # --- OFDM 参考信号与时域 static_target 信道 ---
    x_rg = system.tx_symbols_to_resource_grid()
    x_time = system.components.modulator(x_rg)

    ch = system.components.channel
    snr_db = system.params.channel.snr_db

    # 点目标：多普勒 chirp × FFT 分数时延（距离 + 方位）× 可选自耦合
    y_time_clean = StaticTargetSimulator(params)(x_time)
    # 接收端 SNR 定标 AWGN（内部按 E[|y_clean|²] 与 snr_db 计算 no）
    y_time = ch._awgn(y_time_clean, snr_db)
    y_rg = system.components.demodulator(y_time).squeeze()

    # LS 信道估计：h ≈ Y/X（与单基地脚本后续感知链相同）
    h = system.estimate_channel(x_rg, y_rg)

    # 2D IFFT：子载波 × OFDM 符号 → 时延–多普勒
    h_delay_doppler = system.components.delay_doppler_spectrum(h)

    # 绘制时延–多普勒谱图
    system.components.delay_doppler_spectrum.visualize(
        offset=50,
        file_name=script_out_dir / "static_target_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
    )

    # --- 真值：CLI 目标参数（单目标时形状 (1,)）---
    true_ranges = torch.tensor([args.range_m], dtype=torch.float64, device=device)
    true_velocities = torch.tensor(
        [args.velocity_mps], dtype=torch.float64, device=device
    )

    # --- MUSIC：径向距离 (m)、径向速度 (m/s)、伪谱功率；metric_mode 仅影响日志与谱图坐标 ---
    est_ranges, est_velocities, _ = system.components.music_estimator(
        spectrum_tensor=h_delay_doppler,
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
