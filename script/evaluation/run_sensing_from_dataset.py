"""从 HDF5 数据集回放单基地感知流程（MUSIC 或 CNN 估计 + RMSE 统计）。

须在 **ISAC conda 环境**中运行::

    python script/evaluation/run_sensing_from_dataset.py

流程概要
--------
加载 ``RTDataset``（采集期落盘的 h_dd）→ 构建 ``System`` → 逐 episode 读取 h_dd →
MUSIC 或 CNN 估计 → 与训练标签对齐的 RMSE。

注意：存盘 h_dd 采集时未施加 MTI；MUSIC 评估亦直接使用 h_dd（无法对已存谱补做 MTI）。

估计器分支（``--estimator``）
----------------------------
- ``model``（默认）：``MonostaticDelayDopplerCNN``
- ``music``：``estimate_sensing_music``

真值来源
--------
- MUSIC / CNN 均使用 ``monostatic_labels_from_kinematics``（与训练标签一致）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import (
    DEFAULT_DATASET_H5,
    DEFAULT_MONOSTATIC_CNN_MODEL,
    OUT_DIR,
)
from isac.collection import RTDataset
from isac.sensing import match_peaks_and_compute_radial_rmse
from isac.models import (
    MonostaticCnnCheckpointMeta,
    MonostaticDelayDopplerCNN,
    dd_spectrum_to_features,
    load_monostatic_cnn_checkpoint,
    monostatic_labels_from_kinematics,
    read_monostatic_cnn_checkpoint_meta,
)
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.collection.utils import scene_slug_from_rt_simulator


@dataclass
class EstimatorSetup:
    """估计器运行时状态（MUSIC 或 CNN）。"""

    label: str
    cnn_model: MonostaticDelayDopplerCNN | None
    metric_mode: str


# ---------------------------- 辅助函数 ----------------------------
def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, MonostaticCnnCheckpointMeta, Path]:
    """校验 HDF5 / checkpoint 路径，并从 checkpoint 解析训练配置。"""
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {model_path}")

    ckpt_meta = read_monostatic_cnn_checkpoint_meta(model_path)
    if ckpt_meta.config_file is None or not ckpt_meta.config_file.is_file():
        raise FileNotFoundError(f"checkpoint 缺少有效 config_file: {model_path}")

    return h5_path, model_path, ckpt_meta, ckpt_meta.config_file


@torch.no_grad()
def _estimate_with_model(
    h_dd: torch.Tensor,
    model: MonostaticDelayDopplerCNN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CNN 推理：h_dd → ROI 特征 → 距离/速度估计。"""
    model_device = next(model.parameters()).device
    features = dd_spectrum_to_features(h_dd)
    pred = model(features.unsqueeze(0).to(model_device))
    return (
        pred[0, 0].reshape(-1).to(dtype=torch.float64),
        pred[0, 1].reshape(-1).to(dtype=torch.float64),
    )


def _setup_estimator(
    estimator: str,
    model_path: Path,
    device: torch.device | str,
    *,
    metric_mode: str,
) -> EstimatorSetup:
    """初始化 MUSIC 或 CNN 估计器。"""
    if estimator == "music":
        print(f"估计器: MUSIC | metric_mode={metric_mode}")
        return EstimatorSetup("MUSIC", None, metric_mode)

    cnn_model, cnn_meta = load_monostatic_cnn_checkpoint(model_path, device)
    _print_cnn_estimator_banner(model_path, cnn_meta)
    return EstimatorSetup("CNN", cnn_model, metric_mode)


def _evaluate_episode(
    i: int,
    dataset: RTDataset,
    system: System,
    setup: EstimatorSetup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单 episode：感知估计 → 真值对齐 → 径向 RMSE。"""
    h_dd = dataset.spectrum_tensor(i, device=system.device)
    comps = system.components

    pos = dataset.target_position[i]
    vel = dataset.target_velocity[i]
    range_m, vel_mps = monostatic_labels_from_kinematics(pos, vel, dataset.bs_pos)
    true_range = torch.tensor(range_m, dtype=torch.float64, device=system.device)
    true_velocity = torch.tensor(vel_mps, dtype=torch.float64, device=system.device)

    if setup.label == "MUSIC":
        num_doppler_bins = int(torch.squeeze(h_dd).shape[0])
        peaks_delay, peaks_doppler, peaks_power = comps.music_estimator(
            h_dd,
            num_sources=1,
        )
        result = comps.music_evaluator.evaluate(
            peaks_delay,
            peaks_doppler,
            peaks_power,
            num_doppler_bins=num_doppler_bins,
            true_ranges=true_range.reshape(-1),
            true_velocities=true_velocity.reshape(-1),
            metric_mode=setup.metric_mode,
            sens_mode="monostatic",
            log_peaks=False,
            label=f"episode {i}",
            verbose=False,
        )
        rmse_range_m = result.rmse_range_m
        rmse_velocity_mps = result.rmse_velocity_mps
    else:
        assert setup.cnn_model is not None
        est_ranges, est_velocities = _estimate_with_model(h_dd, setup.cnn_model)

        rmse_range_m, rmse_velocity_mps, _, _, _ = match_peaks_and_compute_radial_rmse(
            est_ranges=est_ranges,
            est_velocities=est_velocities,
            true_ranges=true_range.reshape(-1),
            true_velocities=true_velocity.reshape(-1),
            label=f"episode {i}",
            verbose=False,
        )
    return rmse_range_m, rmse_velocity_mps


def _print_cnn_estimator_banner(
    model_path: Path,
    meta: MonostaticCnnCheckpointMeta,
) -> None:
    """打印 CNN 估计器与 checkpoint 元数据。"""
    epoch_line = f"epoch={meta.epoch}, " if meta.epoch is not None else ""
    print(
        f"估计器: CNN | 模型: {model_path}\n"
        f"  {epoch_line}"
        f"ROI max_range={meta.max_range_m:.1f} m, "
        f"±max_velocity={meta.max_velocity_mps:.1f} m/s, "
        f"Δr={meta.range_resolution:.3f} m, "
        f"Δv={meta.velocity_resolution:.3f} m/s"
    )
    if meta.dataset_h5 is not None:
        print(f"  训练数据集: {meta.dataset_h5}")
    if meta.config_file is not None:
        print(f"  训练配置: {meta.config_file}")


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
            f"数据集回放 ({estimator_label}) — 径向距离 RMSE 统计 {suffix}:",
            torch.stack(range_rmses),
            "m",
            "m²",
        ),
        (
            f"数据集回放 ({estimator_label}) — 径向速度 RMSE 统计 {suffix}:",
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


# ---------------------------- CLI ----------------------------
def argument_parser() -> argparse.Namespace:
    """构造回放脚本的全部 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="ISAC — 从 HDF5 数据集回放感知（MUSIC / CNN + RMSE）"
    )

    parser.add_argument(
        "--dataset_h5",
        type=Path,
        default=DEFAULT_DATASET_H5,
        help="HDF5 数据集路径",
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
        default="rv",
        choices=["dd", "rv"],
        help="谱图与 MUSIC 日志 metric（仅 music 估计器）",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="model",
        choices=["music", "model"],
        help="距离/速度估计器：music（2D-MUSIC）或 model（MonostaticDelayDopplerCNN）",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MONOSTATIC_CNN_MODEL,
        help="checkpoint 路径；所有模式均从此读取训练 TOML，CNN 模式 additionally 加载权重",
    )

    return parser.parse_args()


# ---------------------------- 主函数 ----------------------------
def main() -> None:
    """HDF5 h_dd 回放入口：感知估计 + 逐 episode RMSE + 汇总统计表。"""
    args = argument_parser()
    h5_path, model_path, _ckpt_meta, config_path = _resolve_inputs(args)

    set_random_seed(args.seed)
    dataset = RTDataset.load(h5_path)
    print(f"已加载数据集: {dataset}")

    system = System(load_config(config_path), device=args.device)
    comps = system.components

    rt_simulator = comps.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")

    target_name, _ = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}, 配置: {config_path}")

    setup = _setup_estimator(
        args.estimator,
        model_path,
        system.device,
        metric_mode=args.metric_mode,
    )

    script_out_dir = OUT_DIR / "data_loading"
    script_out_dir.mkdir(parents=True, exist_ok=True)
    rt_simulator.render_to_file(
        filename=f"{scene_slug}_scene.png",
        output_dir=script_out_dir,
    )
    comps.sensing_performance()

    n_episodes = len(dataset)
    range_rmses: list[torch.Tensor] = []
    velocity_rmses: list[torch.Tensor] = []
    range_sum = 0.0
    vel_sum = 0.0

    with tqdm(range(n_episodes), desc="性能评估", unit="ep") as pbar:
        for i in pbar:
            rmse_range_m, rmse_velocity_mps = _evaluate_episode(
                i, dataset, system, setup
            )
            range_rmses.append(rmse_range_m)
            velocity_rmses.append(rmse_velocity_mps)
            range_sum += rmse_range_m.item()
            vel_sum += rmse_velocity_mps.item()
            n_ok = len(range_rmses)
            pbar.set_postfix(
                range_rmse=f"{range_sum / n_ok:.2f}m",
                vel_rmse=f"{vel_sum / n_ok:.2f}m/s",
            )

    print(f"性能评估完成: {n_episodes}/{n_episodes} episodes ({setup.label})")
    _print_rmse_summary(
        range_rmses,
        velocity_rmses,
        estimator_label=setup.label,
        n_episodes=n_episodes,
    )


if __name__ == "__main__":
    main()
