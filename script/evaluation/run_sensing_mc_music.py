"""蒙特卡洛 ROI 采样 → 在线 MUSIC 估计 → 逐 episode RMSE 统计。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/evaluation/run_sensing_mc_music.py

流程概要
--------
1. 解析 CLI，设置随机种子，批量采样 ROI 内位置与速度。
2. 构建 ``System``，循环更新 RT 目标位姿并过滤镜面反射路径与 ``|true_radial_velocity| > Δv``。
3. 每采纳 episode：发射 → 信道 → LS →（默认 MTI）→ DD 谱 →（默认 CFAR 门限）→ MUSIC → 物理换算 → RMSE；前 10 条另写出 ``{scene_slug}_{sens_mode}_scene_XX.png`` 与 DD 谱调试图。
4. 写出 ``{scene_slug}_{sens_mode}_mc_sensing_metrics.csv`` 与终端 RMSE 汇总表。

输出目录
--------
``out/sensing_mc_music/{scene_slug}_{sens_mode}/``（单次运行子目录），其下文件名前缀同为 ``{scene_slug}_{sens_mode}_``：

- ``{scene_slug}_{sens_mode}_scene.png``：静态场景渲染
- ``{scene_slug}_{sens_mode}_scene_XX.png``：前若干采纳 episode 的 RT 路径调试图
- ``{scene_slug}_{sens_mode}_dd_spectrum_XX.png``：前若干采纳 episode 的 DD 谱调试图
- ``{scene_slug}_{sens_mode}_mc_sensing_metrics.csv``：逐 episode 真值/估计/RMSE

采纳条件
--------
从预采样池 ``pop`` 位姿后，须同时满足：

- 存在与目标的 **镜面反射** 路径（``paths_intersect_target_with_interaction(..., "specular")``）
- ``|true_radial_velocity| > velocity_resolution_{sens_mode}``

池耗尽则 ``RuntimeError``（提示增大 ``--sampler_pool_factor``）。

单 episode 评估链（默认 MTI + CFAR）
------------------------------------
transmit → channel → ls_channel_estimator →（可选 MTI）→ delay_doppler_spectrum
→（可选 2D CFAR）→ MUSICEstimator → SensingEstimator → 径向 RMSE

``--sens_mode``（``monostatic`` / ``bistatic``）影响 DD 谱物理轴、MUSIC 峰换算、
速度分辨率筛选门槛、输出子目录与全部输出文件名。

与 ``run_data_collection.py`` 的区别：不写 HDF5，在线做 MUSIC 评估。
与 ``run_sensing_from_dataset.py`` 的区别：不读 HDF5，内联 MC 仿真链。
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import OUT_DIR
from isac.collection import CollectionMetadata
from isac.sensing import match_peaks_and_compute_radial_rmse
from isac.system import System
from isac.utils import set_random_seed
from isac.utils.misc import csv_float2_scalar, csv_vec3

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator
    from isac.data_structures.system_components import SystemComponents

SCRIPT_OUT_BASE = OUT_DIR / "sensing_mc_music"  # 脚本输出根目录；单次运行写入其子目录
DD_DEBUG_PLOT_COUNT = 10  # 前 N 条采纳 episode 额外写出场景/DD 调试图

# 逐 episode CSV 列：位姿与几何真值对齐采集脚本，另含 MUSIC 估计与径向 RMSE
SENSING_CSV_COLUMNS = [
    "sample_idx",
    "position",
    "velocity",
    "true_range_m",
    "true_radial_velocity_mps",
    "est_range_m",
    "est_radial_velocity_mps",
    "rmse_range_m",
    "rmse_radial_velocity_mps",
]

# Sionna/drjit 在大量 RT 路径更新时会重复触发 AST decorator 警告，不影响仿真结果，
# 在采集长循环前过滤以免刷屏。
warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)


def argument_parser() -> argparse.Namespace:
    """构造蒙特卡洛 MUSIC 评估脚本的全部 CLI 参数。

    蒙特卡洛相关参数经 ``CollectionMetadata.from_args`` 解析，
    用于构建预采样池（``num_samples × sampler_pool_factor`` 条候选位姿）。
    """
    parser = argparse.ArgumentParser(
        description="ISAC — 蒙特卡洛 ROI 采样 + 在线 MUSIC 估计 + RMSE"
    )

    sys_group = parser.add_argument_group("系统配置")
    sys_group.add_argument(
        "--config_file",
        type=str,
        default="config/data_collection/data_collection.toml",
        help="仿真与感知链 TOML 配置路径",
    )
    sys_group.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="PyTorch / Sionna 计算设备",
    )

    mc_group = parser.add_argument_group(
        "蒙特卡洛采样",
        "平面 ROI 内位置与速度采样；预采样池大小 = num_samples × sampler_pool_factor",
    )
    mc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（蒙特卡洛位置/速度采样）",
    )
    mc_group.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="目标采纳 episode 数",
    )
    mc_group.add_argument(
        "--sampler_pool_factor",
        type=int,
        default=5,
        help="预采样池倍数（池大小 = num_samples × 本参数）",
    )
    mc_group.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[-4.5, 4.5, -2.5, 2.5],
        help="平面 ROI 四元组（m），xy 平面；z 由 --roi_z 指定",
    )
    mc_group.add_argument(
        "--roi_z",
        type=float,
        default=0.5,
        help="ROI 内目标位置的固定 z 高度（m）",
    )
    mc_group.add_argument(
        "--position_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="位置采样分布",
    )
    mc_group.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[0.5, 3.0],
        help="速度模值范围 (m/s)",
    )
    mc_group.add_argument(
        "--speed_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="速度模值采样分布",
    )

    eval_group = parser.add_argument_group("MUSIC 评估")
    eval_group.add_argument(
        "--metric_mode",
        type=str,
        default="rv",
        choices=["dd", "rv"],
        help="谱图与 MUSIC 日志 metric",
    )
    eval_group.add_argument(
        "--sens_mode",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="感知模式（单基地/双基地），影响 DD 谱轴、MUSIC 换算与输出文件名",
    )
    eval_group.add_argument(
        "--apply_mti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 LS 估计与 DD 谱之间施加 MTI（须 TOML 含 [mti] 段）；默认开，可用 --no-apply_mti 关闭",
    )
    eval_group.add_argument(
        "--apply_cfar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在裁切 DD 谱上对 |h_dd| 做 2D CFAR 并传入 MUSIC 候选门限（须 TOML 含 [cfar] 段）；"
        "默认开，可用 --no-apply_cfar 关闭",
    )

    return parser.parse_args()


def _preflight_checks(
    system: System,
    *,
    apply_mti: bool = False,
    apply_cfar: bool = False,
) -> RTSimulator:
    """校验 RT 链路与 MUSIC / SensingEstimator 已就绪。

    按 CLI 开关条件校验 TOML 段：

    - 始终：``[rt_simulator]``、``[music]``（MUSICEstimator + SensingEstimator）、``sensing_performance``
    - ``apply_mti=True``：``[mti]``
    - ``apply_cfar=True``：``[cfar]``
    """
    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")
    if system.components.music_estimator is None:
        raise ValueError("MUSIC 估计需要 TOML [music] 段以构建 MUSICEstimator")
    if system.components.sensing_estimator is None:
        raise ValueError("感知评估需要 TOML [music] 段以构建 SensingEstimator")
    if apply_mti and system.components.moving_target_indication is None:
        raise ValueError("--apply_mti 需要 TOML [mti] 段以构建 MovingTargetIndication")
    if apply_cfar and system.components.cfar_detector is None:
        raise ValueError("--apply_cfar 需要 TOML [cfar] 段以构建 CFARDetector")
    if system.components.sensing_performance is None:
        raise ValueError(
            "速度分辨率筛选需要 sensing_performance（[ofdm] + carrier_frequency）"
        )
    return rt_simulator


def _cfar_input_from_spectrum(
    h_dd: torch.Tensor,
    *,
    detector: str,
) -> torch.Tensor:
    """按 CFAR detector 类型从复数 DD 谱构造实数检测输入。"""
    h_abs = torch.abs(h_dd)
    if detector == "squarelaw":
        return h_abs.square()
    if detector == "linear":
        return h_abs
    raise ValueError(f"未知 CFAR detector: {detector!r}")


def _output_stem(scene_slug: str, sens_mode: str) -> str:
    """构造输出子目录名与文件名前缀 ``{scene_slug}_{sens_mode}``。"""
    return f"{scene_slug}_{sens_mode}"


def _output_dir(scene_slug: str, sens_mode: str) -> Path:
    """构造单次运行输出目录 ``out/sensing_mc_music/{scene_slug}_{sens_mode}/``。"""
    return SCRIPT_OUT_BASE / _output_stem(scene_slug, sens_mode)


def _render_scene_png(
    rt_simulator: RTSimulator,
    output_stem: str,
    out_dir: Path,
) -> None:
    """将 RT 场景渲染为 PNG。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rt_simulator.render_to_file(
        filename=f"{output_stem}_scene.png",
        output_dir=out_dir,
    )


def _render_episode_scene_png(
    rt_simulator: RTSimulator,
    *,
    output_stem: str,
    episode_idx: int,
    out_dir: Path,
) -> None:
    """渲染单次采纳 episode 的 RT 场景（含路径）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rt_simulator.render_to_file(
        filename=f"{output_stem}_scene_{episode_idx:02d}.png",
        output_dir=out_dir,
        with_paths=True,
    )


def _log_run_context(
    *,
    config_path: str,
    target_name: str,
    scene_slug: str,
    sens_mode: str,
    num_samples: int,
    metric_mode: str,
    apply_mti: bool,
    apply_cfar: bool,
) -> None:
    """打印场景、配置与评估参数摘要。"""
    mti_label = "开" if apply_mti else "关"
    cfar_label = "开" if apply_cfar else "关"
    print(
        f"目标: {target_name}, 场景: {scene_slug}, 配置: {config_path}\n"
        f"估计器: MUSIC | sens_mode={sens_mode} | metric_mode={metric_mode} | "
        f"MTI={mti_label} | CFAR={cfar_label} | 目标采纳数={num_samples}"
    )


def _log_sensing_performance(comps: SystemComponents) -> None:
    """打印感知性能参数表（若已构建 SensingPerformance）。"""
    if comps.sensing_performance is not None:
        comps.sensing_performance()


@torch.no_grad()
def _evaluate_episode(
    system: System,
    *,
    metric_mode: str,
    sens_mode: str,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
    apply_mti: bool = False,
    apply_cfar: bool = False,
    visualize_file: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单 episode 端到端 MUSIC 评估。

    感知链：OFDM 发射 → 加噪信道 → LS 估计 →（可选 MTI）→ DD 谱
    →（可选 CFAR 门限）→ MUSIC 检峰 → ``SensingEstimator`` 物理换算 → 径向 RMSE。

    参数
    ----
    metric_mode, sens_mode :
        谱图轴与 MUSIC / 物理换算尺度（``monostatic`` 或 ``bistatic``）
    true_range, true_velocity :
        RT 几何真值标量张量，与 ``geom.range_tensor[0,0,0]`` 等一致
    visualize_file :
        非 ``None`` 时写出 DD 谱调试图（含 CFAR 叠加）

    返回
    ----
    (rmse_range, rmse_vel, est_range, est_vel, matched_indices)
        径向距离/速度 RMSE 与估计值；``matched_indices`` 本脚本未使用
    """
    comps = system.components
    # --- 发射与频域信道 ---
    _, x_rg, x_time = system.transmit()
    snr_db = system.params.channel.snr_db
    y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)

    h_freq = comps.ls_channel_estimator(x_rg, y_rg)
    if apply_mti:
        mti = comps.moving_target_indication
        if mti is None:
            raise ValueError("apply_mti=True 需要 TOML [mti] 段")
        h_freq = mti(h_freq)
    h_dd = comps.delay_doppler_spectrum(h_freq)

    # --- 2D CFAR：为 MUSIC 提供候选峰门限 ---
    cfar_th: torch.Tensor | None = None
    if apply_cfar:
        cfar = comps.cfar_detector
        if cfar is None:
            raise ValueError("apply_cfar=True 需要 TOML [cfar] 段")
        cfar_input = _cfar_input_from_spectrum(h_dd, detector=cfar.detector)
        cfar_result = cfar(cfar_input, mode="2d")
        if isinstance(cfar_result, torch.Tensor):
            cfar_th = cfar_result.to(device=h_dd.device, dtype=h_dd.real.dtype)
        else:
            cfar_th = torch.as_tensor(
                cfar_result, device=h_dd.device, dtype=h_dd.real.dtype
            )

    if visualize_file is not None:
        comps.delay_doppler_spectrum.visualize(
            file_name=visualize_file,
            cfar=cfar_th,
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            to_db=False,
        )
    # --- MUSIC 检峰 → 物理量换算 → 与几何真值对齐的径向 RMSE ---
    peaks = comps.music_estimator(h_dd, num_sources=1, cfar=cfar_th)
    estimate = comps.sensing_estimator(
        peaks,
        metric_mode=metric_mode,
        sens_mode=sens_mode,
        log_peaks=False,
    )
    return match_peaks_and_compute_radial_rmse(
        est_ranges=estimate.est_ranges,
        est_velocities=estimate.est_velocities,
        true_ranges=true_range.reshape(-1),
        true_velocities=true_velocity.reshape(-1),
        label="",
        verbose=False,
    )


def _music_pbar_postfix(
    *,
    true_range: torch.Tensor,
    est_range: torch.Tensor,
    rmse_range: torch.Tensor,
    true_velocity: torch.Tensor,
    est_velocity: torch.Tensor,
    rmse_velocity: torch.Tensor,
) -> str:
    """构造单次 MUSIC 评估的 tqdm postfix（保留两位小数，与 CSV 一致）。"""
    r_t = csv_float2_scalar(true_range)
    r_e = csv_float2_scalar(est_range)
    r_mse = csv_float2_scalar(rmse_range)
    v_t = csv_float2_scalar(true_velocity)
    v_e = csv_float2_scalar(est_velocity)
    v_mse = csv_float2_scalar(rmse_velocity)
    return f"r({r_t},{r_e},{r_mse}) v({v_t},{v_e},{v_mse})"


def _log_dd_debug_plots(out_dir: Path, output_stem: str, n_saved: int) -> None:
    """打印前若干 episode 的 DD 调试图写出摘要。"""
    if n_saved <= 0:
        return
    first = out_dir / f"{output_stem}_dd_spectrum_00.png"
    last = out_dir / f"{output_stem}_dd_spectrum_{n_saved - 1:02d}.png"
    if n_saved == 1:
        print(f"DD 调试图已写入: {first} (共 1 张)")
    else:
        print(f"DD 调试图已写入: {first} ... {last} (共 {n_saved} 张)")


def _log_scene_debug_plots(out_dir: Path, output_stem: str, n_saved: int) -> None:
    """打印前若干 episode 的场景调试图写出摘要。"""
    if n_saved <= 0:
        return
    first = out_dir / f"{output_stem}_scene_00.png"
    last = out_dir / f"{output_stem}_scene_{n_saved - 1:02d}.png"
    if n_saved == 1:
        print(f"场景调试图已写入: {first} (共 1 张)")
    else:
        print(f"场景调试图已写入: {first} ... {last} (共 {n_saved} 张)")


def _run_mc_loop(
    *,
    system: System,
    collection_meta: CollectionMetadata,
    sampler,
    target,
    rt_simulator: RTSimulator,
    metric_mode: str,
    output_stem: str,
    out_dir: Path,
    sens_mode: str,
    apply_mti: bool,
    apply_cfar: bool,
) -> tuple[
    list[dict[str, str | int]],
    list[torch.Tensor],
    list[torch.Tensor],
    float,
    int,
]:
    """蒙特卡洛主循环：采样 → RT 过滤 → MUSIC 评估 → 累积结果。

    从预采样池依次 ``pop`` 候选位姿；未通过过滤的候选仍计入 ``attempts``。
    前 ``DD_DEBUG_PLOT_COUNT`` 条采纳 episode 额外写出场景与 DD 谱调试图。

    返回
    ----
    csv_rows, range_rmses, velocity_rmses, acceptance_rate, attempts
        ``acceptance_rate = accepted / attempts``（拒绝候选计入 attempts）
    """
    csv_rows: list[dict[str, str | int]] = []
    range_rmses: list[torch.Tensor] = []
    velocity_rmses: list[torch.Tensor] = []
    accepted = 0  # 已采纳 episode 数
    attempts = 0  # pop 次数（含被过滤拒绝的候选）
    sp = system.components.sensing_performance
    v_res = float(getattr(sp, f"velocity_resolution_{sens_mode}"))

    with tqdm(total=collection_meta.num_samples, desc="MUSIC 评估", unit="ep") as pbar:
        while accepted < collection_meta.num_samples:
            if len(sampler) == 0:
                raise RuntimeError(
                    f"采样池已耗尽：已采纳 {accepted}/{collection_meta.num_samples} 条。"
                    "请增大 --sampler_pool_factor 或调整过滤条件"
                    "（scene_filter / 镜面反射 / 径向速度分辨率）。"
                )
            pos, vel, ori = sampler.pop()
            attempts += 1

            # 更新目标位姿并重算 RT 路径（结果缓存供后续 channel 复用）
            target(
                position=pos,
                velocity=vel,
                orientation=ori,
            )
            rt_simulator.paths(update=True)

            geom = rt_simulator.rx_target_tx_geometric
            # 单 TX/RX 几何真值：首 RX、首目标、首 TX
            true_range = geom.range_tensor[0, 0, 0]
            true_velocity = geom.vel_tensor[0, 0, 0]
            # 径向速度须超过 sens_mode 对应速度分辨率，避免近零多普勒难以分辨
            if abs(float(true_velocity.item())) <= v_res:
                continue

            # 须存在目标镜面反射路径，否则无法形成可感知回波
            if not rt_simulator.paths_intersect_target_with_interaction(
                target, "specular"
            ):
                continue

            # 前 DD_DEBUG_PLOT_COUNT 条采纳 episode 写出 RT 场景调试图
            if accepted < DD_DEBUG_PLOT_COUNT:
                _render_episode_scene_png(
                    rt_simulator,
                    output_stem=output_stem,
                    episode_idx=accepted,
                    out_dir=out_dir,
                )

            viz_path = None
            if accepted < DD_DEBUG_PLOT_COUNT:
                viz_path = out_dir / f"{output_stem}_dd_spectrum_{accepted:02d}.png"

            # 感知链评估：MUSIC 估计 + 与几何真值对齐的径向 RMSE
            rmse_range, rmse_vel, est_range, est_vel, _ = _evaluate_episode(
                system,
                metric_mode=metric_mode,
                sens_mode=sens_mode,
                true_range=true_range,
                true_velocity=true_velocity,
                apply_mti=apply_mti,
                apply_cfar=apply_cfar,
                visualize_file=viz_path,
            )

            csv_rows.append(
                {
                    "sample_idx": accepted,
                    "position": csv_vec3(pos),
                    "velocity": csv_vec3(vel),
                    "true_range_m": csv_float2_scalar(true_range),
                    "true_radial_velocity_mps": csv_float2_scalar(true_velocity),
                    "est_range_m": csv_float2_scalar(est_range),
                    "est_radial_velocity_mps": csv_float2_scalar(est_vel),
                    "rmse_range_m": csv_float2_scalar(rmse_range),
                    "rmse_radial_velocity_mps": csv_float2_scalar(rmse_vel),
                }
            )
            range_rmses.append(rmse_range)
            velocity_rmses.append(rmse_vel)
            accepted += 1
            pbar.update(1)
            pbar.set_postfix_str(
                _music_pbar_postfix(
                    true_range=true_range,
                    est_range=est_range,
                    rmse_range=rmse_range,
                    true_velocity=true_velocity,
                    est_velocity=est_vel,
                    rmse_velocity=rmse_vel,
                )
            )

    acceptance_rate = accepted / attempts if attempts else 0.0
    return csv_rows, range_rmses, velocity_rmses, acceptance_rate, attempts


def _write_sensing_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    """写出逐 episode 感知指标 CSV。"""
    if not rows:
        print("无 CSV 行，跳过写出")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=SENSING_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SENSING_CSV_COLUMNS})
    print(f"感知指标 CSV 已写入: {path}")


def _rmse_stats_rows(values: torch.Tensor, *, unit: str, var_unit: str) -> list[list]:
    """构造均值/中位数/最小/最大/方差五行表格数据。"""
    return [
        ["均值", float(values.mean().item()), unit],
        ["中位数", float(values.median().item()), unit],
        ["最小值", float(values.min().item()), unit],
        ["最大值", float(values.max().item()), unit],
        ["方差", float(values.var(unbiased=False).item()), var_unit],
    ]


def _print_rmse_summary(
    range_rmses: list[torch.Tensor],
    velocity_rmses: list[torch.Tensor],
    *,
    estimator_label: str,
    n_episodes: int,
) -> None:
    """以表格输出逐 episode 径向距离与速度 RMSE 统计。"""
    suffix = f"({n_episodes} episodes)"
    tables = [
        (
            f"蒙特卡洛 MUSIC 在线评估 ({estimator_label}) — 径向距离 RMSE 统计 {suffix}:",
            torch.stack(range_rmses),
            "m",
            "m²",
        ),
        (
            f"蒙特卡洛 MUSIC 在线评估 ({estimator_label}) — 径向速度 RMSE 统计 {suffix}:",
            torch.stack(velocity_rmses),
            "m/s",
            "(m/s)²",
        ),
    ]
    for title, values, unit, var_unit in tables:
        print(title)
        print(
            tabulate(
                _rmse_stats_rows(values, unit=unit, var_unit=var_unit),
                headers=["统计量", "数值", "单位"],
                tablefmt="simple_grid",
                floatfmt=".4f",
            )
        )


def main() -> None:
    """蒙特卡洛 MUSIC 在线评估入口。

    分阶段：CLI 初始化 → 场景绑定 → MC 评估循环 → CSV / 调试图 / RMSE 汇总落盘。
    """
    # --- 初始化：CLI → 采集元数据 → System ---
    args = argument_parser()
    set_random_seed(args.seed)
    collection_meta = CollectionMetadata.from_args(args)  # 设种子并校验 num_samples 等
    sampler = collection_meta.build_sampler()  # 预生成 pool_size 条候选位姿

    system = System(args.config_file, device=args.device)
    rt_simulator = _preflight_checks(
        system,
        apply_mti=args.apply_mti,
        apply_cfar=args.apply_cfar,
    )

    # --- 场景绑定：RT 目标、scene_slug、output_stem、out_dir ---
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = getattr(rt_simulator.rt_simulator_params, "filename", "None")
    output_stem = _output_stem(scene_slug, args.sens_mode)
    out_dir = _output_dir(scene_slug, args.sens_mode)
    _log_run_context(
        config_path=args.config_file,
        target_name=target_name,
        scene_slug=scene_slug,
        sens_mode=args.sens_mode,
        num_samples=collection_meta.num_samples,
        metric_mode=args.metric_mode,
        apply_mti=args.apply_mti,
        apply_cfar=args.apply_cfar,
    )
    print(f"评估输出: {out_dir}/")
    _render_scene_png(rt_simulator, output_stem, out_dir)
    _log_sensing_performance(system.components)

    # --- 蒙特卡洛评估循环 ---
    csv_rows, range_rmses, velocity_rmses, acceptance_rate, attempts = _run_mc_loop(
        system=system,
        collection_meta=collection_meta,
        sampler=sampler,
        target=target,
        rt_simulator=rt_simulator,
        metric_mode=args.metric_mode,
        output_stem=output_stem,
        out_dir=out_dir,
        sens_mode=args.sens_mode,
        apply_mti=args.apply_mti,
        apply_cfar=args.apply_cfar,
    )

    print(f"接受率: {acceptance_rate:.1%} ({len(csv_rows)}/{attempts})")

    # --- 落盘：CSV + 调试图摘要 + RMSE 汇总 ---
    csv_path = out_dir / f"{output_stem}_mc_sensing_metrics.csv"
    _write_sensing_csv(csv_path, csv_rows)
    n_debug_plots = min(len(csv_rows), DD_DEBUG_PLOT_COUNT)
    _log_scene_debug_plots(out_dir, output_stem, n_debug_plots)
    _log_dd_debug_plots(out_dir, output_stem, n_debug_plots)

    n_episodes = len(csv_rows)
    print(f"性能评估完成: {n_episodes}/{collection_meta.num_samples} episodes (MUSIC)")
    _print_rmse_summary(
        range_rmses,
        velocity_rmses,
        estimator_label="MUSIC",
        n_episodes=n_episodes,
    )


if __name__ == "__main__":
    main()
