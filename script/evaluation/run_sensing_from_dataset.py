"""从 HDF5 数据集回放单基地感知流程（MUSIC 或 CNN 估计 + RMSE 统计）。

须在 **ISAC conda 环境**中运行::

    python script/evaluation/run_sensing_from_dataset.py

流程概要
--------
加载 ``RTDataset``（采集期落盘的 h_dd）→ 构建 ``System`` → 逐 episode 读取 h_dd →
MUSIC 或 CNN 估计 → 与几何真值对齐的 RMSE。

注意：存盘 h_dd 采集时未施加 MTI；MUSIC 评估亦直接使用 h_dd（无法对已存谱补做 MTI）。

估计器分支（``--estimator``）
----------------------------
- ``music``（默认）：``estimate_sensing_music``
- ``model``：``MonostaticDelayDopplerCNN``

真值来源
--------
- MUSIC：``los_truth_from_kinematics``（RT 默认 Rx–Target–Tx 三元组几何）
- CNN：``monostatic_labels_from_kinematics``（与训练标签一致）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import (
    DEFAULT_DATASET_H5,
    DEFAULT_MONOSTATIC_CNN_MODEL,
    OUT_DIR,
    PROJECT_ROOT,
)
from isac.datasets import RTDataset
from isac.models import (
    MonostaticCnnCheckpointMeta,
    MonostaticDelayDopplerCNN,
    dd_spectrum_to_features,
    load_monostatic_cnn_checkpoint,
    monostatic_labels_from_kinematics,
    read_monostatic_cnn_checkpoint_meta,
)
from isac.system import MusicEstimate, System
from isac.utils import load_config, set_random_seed
from isac.utils.data_collection.channel_export import (
    los_truth_from_kinematics,
    scene_slug_from_rt_simulator,
)


# ---------------------------- 辅助函数 ----------------------------
def _default_config_for_h5(h5_path: Path) -> Path:
    """与 HDF5 同目录的 ``data_collection.toml`` 副本。"""
    sibling = h5_path.parent / "data_collection.toml"
    if sibling.is_file():
        return sibling
    return PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@torch.no_grad()
def _estimate_with_model(
    h_dd: torch.Tensor,
    model: MonostaticDelayDopplerCNN,
    *,
    use_phase: bool,
    sensing_performance,
) -> MusicEstimate:
    """CNN 推理：h_dd → ROI 特征 → 距离/速度估计。

    返回 ``MusicEstimate`` 以便复用 ``System.evaluate_sensing_rmse`` 与 MUSIC 分支对齐。
    """
    model_device = next(model.parameters()).device
    features = dd_spectrum_to_features(
        h_dd,
        use_phase=use_phase,
    )
    pred = model(features.unsqueeze(0).to(model_device))
    return MusicEstimate(
        est_ranges=pred[0, 0].reshape(-1).to(dtype=torch.float64),
        est_velocities=pred[0, 1].reshape(-1).to(dtype=torch.float64),
    )


def _print_cnn_estimator_banner(
    model_path: Path,
    meta: MonostaticCnnCheckpointMeta,
    *,
    use_phase: bool,
) -> None:
    """打印 CNN 估计器与 checkpoint 元数据（epoch、ROI、训练数据集/配置等）。"""
    epoch_line = f"epoch={meta.epoch}, " if meta.epoch is not None else ""
    print(
        f"估计器: CNN | 模型: {model_path}\n"
        f"  {epoch_line}"
        f"ROI max_range={meta.max_range_m:.1f} m, "
        f"±max_velocity={meta.max_velocity_mps:.1f} m/s, "
        f"use_phase={use_phase}, "
        f"Δr={meta.range_resolution:.3f} m, "
        f"Δv={meta.velocity_resolution:.3f} m/s"
    )
    if meta.dataset_h5 is not None:
        print(f"  训练数据集: {meta.dataset_h5}")
    if meta.config_file is not None:
        print(f"  训练配置: {meta.config_file}")


def _rmse_stats_rows(values: torch.Tensor, *, unit: str, var_unit: str) -> list[list]:
    """构造均值/中位数/最小/最大/方差五行表格数据。

    方差为总体方差（``torch.var(unbiased=False)``）。
    """
    return [
        ["均值", float(values.mean().item()), unit],
        ["中位数", float(values.median().item()), unit],
        ["最小值", float(values.min().item()), unit],
        ["最大值", float(values.max().item()), unit],
        ["方差", float(values.var(unbiased=False).item()), var_unit],
    ]


def _print_rmse_stats_table(
    title: str,
    rows: list[list],
) -> None:
    """以 ``simple_grid`` 表格打印 RMSE 统计行。"""
    print(title)
    print(
        tabulate(
            rows,
            headers=["统计量", "数值", "单位"],
            tablefmt="simple_grid",
            floatfmt=".4f",
        )
    )


def _print_rmse_stats_tables(
    range_rmses: list[torch.Tensor],
    velocity_rmses: list[torch.Tensor],
    *,
    estimator_label: str,
    evaluated: int,
) -> None:
    """以表格输出逐 episode 径向距离与速度 RMSE 统计。"""
    suffix = f"({evaluated} episodes)"
    range_vals = torch.stack(range_rmses)
    vel_vals = torch.stack(velocity_rmses)
    _print_rmse_stats_table(
        f"数据集回放 ({estimator_label}) — 径向距离 RMSE 统计 {suffix}:",
        _rmse_stats_rows(range_vals, unit="m", var_unit="m²"),
    )
    _print_rmse_stats_table(
        f"数据集回放 ({estimator_label}) — 径向速度 RMSE 统计 {suffix}:",
        _rmse_stats_rows(vel_vals, unit="m/s", var_unit="(m/s)²"),
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
        "--config_file",
        type=Path,
        default=None,
        help="TOML 配置；CNN 模式默认优先 checkpoint 内训练配置",
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
        "--max_episodes",
        type=int,
        default=None,
        help="最多回放 episode 数；默认全部",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="range_velocity",
        choices=["delay_doppler", "range_velocity"],
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
        help="CNN checkpoint 路径（--estimator model 时使用）",
    )
    parser.add_argument(
        "--no_phase",
        action="store_true",
        help="CNN 特征不含相位通道；未指定时从 checkpoint use_phase 读取",
    )

    return parser.parse_args()


# ---------------------------- 主函数 ----------------------------
def main() -> None:
    """HDF5 h_dd 回放入口：感知估计 + 逐 episode RMSE + 汇总统计表。"""
    args = argument_parser()

    # --- 路径校验与 CNN 元数据预读 ---
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    cnn_peek_meta: MonostaticCnnCheckpointMeta | None = None
    if args.estimator == "model":
        model_path = Path(args.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"模型 checkpoint 不存在: {model_path}")
        cnn_peek_meta = read_monostatic_cnn_checkpoint_meta(model_path)

    # --- 配置文件解析（CNN 优先 checkpoint 训练配置）---
    if args.config_file is not None:
        config_path = Path(args.config_file).resolve()
    elif (
        cnn_peek_meta is not None
        and cnn_peek_meta.config_file is not None
        and cnn_peek_meta.config_file.is_file()
    ):
        config_path = cnn_peek_meta.config_file
    else:
        config_path = _default_config_for_h5(h5_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # --- 数据集 / System 构建 ---
    set_random_seed(args.seed)
    dataset = RTDataset.load(h5_path)
    print(f"已加载数据集: {dataset}")

    config = load_config(config_path)
    system = System(
        config=config,
        device=args.device,
    )

    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")

    target_name, _ = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}, 配置: {config_path}")

    # --- 估计器初始化（MUSIC 或 CNN）---
    cnn_model: MonostaticDelayDopplerCNN | None = None
    use_phase = True
    estimator_label = "MUSIC"
    if args.estimator == "model":
        model_path = Path(args.model_path).resolve()
        cnn_model, cnn_meta = load_monostatic_cnn_checkpoint(
            model_path,
            system.device,
        )
        use_phase = False if args.no_phase else cnn_meta.use_phase
        estimator_label = "CNN"
        _print_cnn_estimator_banner(model_path, cnn_meta, use_phase=use_phase)
    else:
        print(f"估计器: MUSIC | metric_mode={args.metric_mode}")

    script_out_dir = OUT_DIR / "data_loading"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    rt_simulator.render_to_file(
        filename=f"{scene_slug}_scene.png",
        output_dir=script_out_dir,
    )

    system.components.sensing_performance()

    n_episodes = len(dataset)
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)

    range_rmses: list[torch.Tensor] = []
    velocity_rmses: list[torch.Tensor] = []
    range_sum = 0.0
    vel_sum = 0.0

    # --- 逐 episode：读取 h_dd → 感知估计 → RMSE ---
    with tqdm(range(n_episodes), desc="性能评估", unit="ep") as pbar:
        for i in pbar:
            h_dd = dataset.spectrum_tensor(i, device=system.device)

            if args.estimator == "music":
                estimate = system.estimate_sensing_music(
                    h_dd,
                    metric_mode=args.metric_mode,
                    log_peaks=False,
                )
            else:
                assert cnn_model is not None
                estimate = _estimate_with_model(
                    h_dd,
                    cnn_model,
                    use_phase=use_phase,
                    sensing_performance=system.components.sensing_performance,
                )

            pos = dataset.target_position[i]
            vel = dataset.target_velocity[i]
            if args.estimator == "model":
                range_m, vel_mps = monostatic_labels_from_kinematics(
                    pos, vel, dataset.bs_pos
                )
                true_range = torch.tensor(
                    range_m, dtype=torch.float64, device=system.device
                )
                true_velocity = torch.tensor(
                    vel_mps, dtype=torch.float64, device=system.device
                )
            else:
                true_range, true_velocity = los_truth_from_kinematics(
                    pos, vel, rt_simulator, system.device
                )
            rmse = system.evaluate_sensing_rmse(
                estimate,
                true_ranges=true_range,
                true_velocities=true_velocity,
                label=f"性能评估 ep={i}",
                verbose=False,
            )
            range_rmses.append(rmse.rmse_range_m)
            velocity_rmses.append(rmse.rmse_velocity_mps)
            range_sum += rmse.rmse_range_m.item()
            vel_sum += rmse.rmse_velocity_mps.item()
            n_ok = len(range_rmses)
            # 运行均值：仅统计已完成 episode
            pbar.set_postfix(
                range_rmse=f"{range_sum / n_ok:.2f}m",
                vel_rmse=f"{vel_sum / n_ok:.2f}m/s",
            )

    evaluated = len(range_rmses)
    print(f"性能评估完成: {evaluated}/{n_episodes} episodes ({estimator_label})")

    # --- 汇总：距离/速度 RMSE 统计表 ---
    _print_rmse_stats_tables(
        range_rmses,
        velocity_rmses,
        estimator_label=estimator_label,
        evaluated=evaluated,
    )


if __name__ == "__main__":
    main()
