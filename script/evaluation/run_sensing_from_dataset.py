"""从 HDF5 数据集回放感知流程（MUSIC 或 CNN 估计 + RMSE 统计）。

须在 **ISAC conda 环境**中运行::

    python script/evaluation/run_sensing_from_dataset.py

双基地 CNN 示例::

    python script/evaluation/run_sensing_from_dataset.py \\
        --dataset_h5 data/empty_room_bistatic_30kHz/empty_room_bistatic_mc_sionna_dataset.h5 \\
        --config_file config/data_collection/data_collection_bistatic.toml \\
        --sens_mode bistatic \\
        --model_path models/sensing_cnn/bistatic/best_model.pth

流程概要
--------
加载 ``RTDataset`` → 构建 ``System`` → 逐 episode 读取 ``spectrum_tensor`` →
MUSIC 或 CNN 估计局部 bin → 转 ``MusicPeaks`` → 与几何真值对齐的 RMSE。

估计器（``--estimator``）
------------------------
- ``music``：``MUSICEstimator`` → ``MusicPeaks``
- ``model``（默认）：``SensingCNN`` → ``(B, 2)`` bin → ``MusicPeaks``

真值：``kinematics_to_range_velocity``（与训练标签一致；mono/bistatic 由 ``--sens_mode`` 控制）。

注意：存盘 h_dd 采集时未施加 MTI；回放亦直接使用 h_dd。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import (
    DEFAULT_BISTATIC_SENSING_CNN_MODEL,
    DEFAULT_DATASET_H5,
    DEFAULT_MONOSTATIC_CNN_MODEL,
    OUT_DIR,
)
from isac.collection import RTDataset, sensing_attrs_from_system
from isac.data_structures.types import MusicPeaks, SensMode
from isac.models import (
    SensingCNN,
    kinematics_to_range_velocity,
    load_sensing_cnn_checkpoint,
)
from isac.sensing import match_peaks_and_compute_radial_rmse
from isac.system import System
from isac.utils import set_random_seed

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator
    from isac.data_structures.system_components import SystemComponents
    from isac.sensing.detection.music_estimator import MUSICEstimator
    from isac.sensing.evaluation.sensing_estimator import SensingEstimator


@dataclass(frozen=True)
class EvalInputs:
    """校验后的回放输入路径。"""

    h5_path: Path
    model_path: Path
    config_path: Path
    sens_mode: SensMode


def _resolve_config_path(h5_path: Path, config_file: Path | None) -> Path:
    """解析采集 TOML：显式指定或 HDF5 同目录唯一 ``data_collection*.toml``。"""
    if config_file is not None:
        config_path = config_file.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    candidates = sorted(h5_path.parent.glob("data_collection*.toml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"HDF5 同目录未找到 data_collection*.toml: {h5_path.parent}；"
            "请通过 --config_file 指定"
        )
    raise ValueError(
        f"HDF5 同目录存在多个 TOML {candidates!r}，请通过 --config_file 指定"
    )


@dataclass
class EstimatorSetup:
    """估计器运行时状态（MUSIC 或 CNN）。"""

    label: str
    cnn_model: SensingCNN | None
    metric_mode: str


def _resolve_inputs(args: argparse.Namespace) -> EvalInputs:
    """校验 HDF5 / checkpoint 路径；TOML 由 ``--config_file`` 或 HDF5 同目录解析。"""
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    config_path = _resolve_config_path(h5_path, args.config_file)

    if args.model_path is None:
        model_path = (
            DEFAULT_BISTATIC_SENSING_CNN_MODEL
            if args.sens_mode == "bistatic"
            else DEFAULT_MONOSTATIC_CNN_MODEL
        )
    else:
        model_path = args.model_path.resolve()
    if args.estimator == "model" and not model_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {model_path}")

    return EvalInputs(
        h5_path=h5_path,
        model_path=model_path,
        config_path=config_path,
        sens_mode=args.sens_mode,
    )


def _assert_bistatic_topology(rt_simulator: RTSimulator) -> None:
    """校验 TOML 为分离收发拓扑，且首条链路为双基地。"""
    geom = rt_simulator.rx_target_tx_geometric
    if len(geom.tx_names) < 1 or len(geom.rx_names) < 1:
        raise ValueError(
            f"双基地评估需要至少 1 个 TX 与 1 个 RX，"
            f"收到 tx={geom.tx_names!r}, rx={geom.rx_names!r}"
        )
    if not bool(geom.type_tensor[0, 0, 0].item()):
        raise ValueError(
            "首条链路 type_tensor[0,0,0] 为单基地；"
            "请使用分离收发 TOML（如 data_collection_bistatic.toml）"
        )


def _resolve_tx_pos(system: System, sens_mode: SensMode) -> torch.Tensor | None:
    """双基地时返回首 TX 位置；单基地返回 ``None``。"""
    if sens_mode != "bistatic":
        return None
    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("双基地评估要求 channel.type='rt'")
    _assert_bistatic_topology(rt_simulator)
    geom = rt_simulator.rx_target_tx_geometric
    tx_pos = rt_simulator.tx_states[geom.tx_names[0]][0]
    return torch.tensor(tx_pos, dtype=torch.float32)


def _preflight_checks(system: System) -> RTSimulator:
    """校验 RT 链路与 SensingEstimator 已就绪。"""
    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")
    if system.components.sensing_estimator is None:
        raise ValueError("感知评估需要 TOML [music] 段以构建 SensingEstimator")
    return rt_simulator


def _estimate_peaks(
    h_dd: torch.Tensor,
    setup: EstimatorSetup,
    music_estimator: MUSICEstimator | None,
) -> MusicPeaks:
    """MUSIC 或 CNN 统一峰值估计入口。"""
    if setup.cnn_model is None:
        if music_estimator is None:
            raise ValueError("MUSIC 估计需要 TOML [music] 段以构建 MUSICEstimator")
        return music_estimator(h_dd, num_sources=1)

    assert setup.cnn_model is not None
    bins = setup.cnn_model(h_dd)
    return MusicPeaks.from_local_bins(bins[0, 0], bins[0, 1], device=bins.device)


def _episode_ground_truth(
    dataset: RTDataset,
    idx: int,
    device: torch.device | str,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单 episode 几何真值：距离/速度与 sens_mode 一致。"""
    pos = torch.as_tensor(dataset.target_position[idx], device=device).reshape(1, 3)
    vel = torch.as_tensor(dataset.target_velocity[idx], device=device).reshape(1, 3)
    bs = torch.as_tensor(dataset.bs_pos, device=device)
    range_m, vel_mps = kinematics_to_range_velocity(
        pos,
        vel,
        bs,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    return range_m.reshape(-1), vel_mps.reshape(-1)


def _setup_estimator(
    estimator: str,
    model_path: Path,
    device: torch.device | str,
    *,
    metric_mode: str,
    sens_mode: SensMode,
    sensing: dict[str, Any] | None = None,
) -> EstimatorSetup:
    """初始化 MUSIC 或 CNN 估计器。"""
    if estimator == "music":
        print(f"估计器: MUSIC | sens_mode={sens_mode} | metric_mode={metric_mode}")
        label = f"MUSIC ({sens_mode})"
        return EstimatorSetup(label, None, metric_mode)

    if sensing is None:
        raise ValueError("CNN 估计器需要 sensing 属性（来自 System / TOML）")
    cnn_model = load_sensing_cnn_checkpoint(model_path, device)
    _print_cnn_estimator_banner(model_path, sensing, sens_mode=sens_mode)
    return EstimatorSetup(f"CNN ({sens_mode})", cnn_model, metric_mode)


@torch.no_grad()
def _evaluate_episode(
    idx: int,
    dataset: RTDataset,
    system: System,
    setup: EstimatorSetup,
    sensing_estimator: SensingEstimator,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单 episode：读谱 → 估峰 → 物理换算 → RMSE。"""
    h_dd = dataset.spectrum_tensor(idx, device=system.device)
    true_range, true_velocity = _episode_ground_truth(
        dataset,
        idx,
        system.device,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    peaks = _estimate_peaks(h_dd, setup, system.components.music_estimator)
    estimate = sensing_estimator(
        peaks,
        metric_mode=setup.metric_mode,
        sens_mode=sens_mode,
        log_peaks=False,
    )
    return match_peaks_and_compute_radial_rmse(
        est_ranges=estimate.est_ranges,
        est_velocities=estimate.est_velocities,
        true_ranges=true_range.reshape(-1),
        true_velocities=true_velocity.reshape(-1),
        label=f"episode {idx}",
        verbose=False,
    )[:2]


def _run_evaluation_loop(
    dataset: RTDataset,
    system: System,
    setup: EstimatorSetup,
    sensing_estimator: SensingEstimator,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """逐 episode 评估，返回距离/速度 RMSE 列表。"""
    range_rmses: list[torch.Tensor] = []
    velocity_rmses: list[torch.Tensor] = []
    range_sum = 0.0
    vel_sum = 0.0

    with tqdm(range(len(dataset)), desc="性能评估", unit="ep") as pbar:
        for i in pbar:
            rmse_range_m, rmse_velocity_mps = _evaluate_episode(
                i,
                dataset,
                system,
                setup,
                sensing_estimator,
                sens_mode=sens_mode,
                tx_pos=tx_pos,
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

    return range_rmses, velocity_rmses


def _log_run_context(
    dataset: RTDataset,
    config_path: Path,
    target_name: str,
    scene_slug: str,
) -> None:
    """打印数据集、场景与配置摘要。"""
    print(f"已加载数据集: {dataset}")
    print(f"目标: {target_name}, 场景: {scene_slug}, 配置: {config_path}")


def _render_scene_png(
    rt_simulator: RTSimulator,
    scene_slug: str,
    out_dir: Path,
) -> None:
    """将 RT 场景渲染为 PNG。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rt_simulator.render_to_file(
        filename=f"{scene_slug}_scene.png",
        output_dir=out_dir,
    )


def _log_sensing_performance(comps: SystemComponents) -> None:
    """打印感知性能参数表（若已构建 SensingPerformance）。"""
    if comps.sensing_performance is not None:
        comps.sensing_performance()


def _print_cnn_estimator_banner(
    model_path: Path,
    sensing: dict[str, Any],
    *,
    sens_mode: SensMode,
) -> None:
    """打印 CNN 估计器与 TOML 感知属性摘要。"""
    range_label = "折叠路径长" if sens_mode == "bistatic" else "斜距"
    print(
        f"估计器: CNN | sens_mode={sens_mode} | 模型: {model_path}\n"
        f"  ROI max_{range_label}={sensing['max_range_m']:.1f} m, "
        f"±max_velocity={sensing['max_velocity_mps']:.1f} m/s, "
        f"Δr={sensing['range_resolution']:.3f} m, "
        f"Δv={sensing['velocity_resolution']:.3f} m/s"
    )


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
    sens_mode: SensMode,
) -> None:
    """以表格输出逐 episode 距离/速度 RMSE 统计。"""
    suffix = f"({n_episodes} episodes)"
    range_title = (
        "折叠路径长 RMSE 统计" if sens_mode == "bistatic" else "径向距离 RMSE 统计"
    )
    velocity_title = (
        "路径变化率 RMSE 统计" if sens_mode == "bistatic" else "径向速度 RMSE 统计"
    )
    tables = [
        (
            f"数据集回放 ({estimator_label}) — {range_title} {suffix}:",
            torch.stack(range_rmses),
            "m",
            "m²",
        ),
        (
            f"数据集回放 ({estimator_label}) — {velocity_title} {suffix}:",
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
        help="采集 TOML；默认在 HDF5 同目录查找 data_collection*.toml（须唯一）",
    )
    parser.add_argument(
        "--sens_mode",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="感知模式；影响真值换算、SensingEstimator 与默认模型路径",
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
        help="距离/速度估计器：music（2D-MUSIC）或 model（SensingCNN）",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="checkpoint 路径（CNN 模式）；默认 models/sensing_cnn/{sens_mode}/best_model.pth",
    )

    return parser.parse_args()


def main() -> None:
    """HDF5 h_dd 回放入口：感知估计 + 逐 episode RMSE + 汇总统计表。"""
    args = argument_parser()
    inputs = _resolve_inputs(args)
    sens_mode = inputs.sens_mode

    set_random_seed(args.seed)
    dataset = RTDataset.load(inputs.h5_path)
    system = System(inputs.config_path, device=args.device)
    rt_simulator = _preflight_checks(system)

    target_name, _ = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = getattr(rt_simulator.rt_simulator_params, "filename", "None")
    _log_run_context(dataset, inputs.config_path, target_name, scene_slug)

    tx_pos_tensor = _resolve_tx_pos(system, sens_mode)
    tx_pos = tx_pos_tensor.to(system.device) if tx_pos_tensor is not None else None

    # 初始化 dd_spectrum_roi.slices，供 SensingEstimator 读取 num_doppler_bins
    sensing_attrs = sensing_attrs_from_system(system, sens_mode=sens_mode)
    sensing = sensing_attrs if args.estimator == "model" else None
    setup = _setup_estimator(
        args.estimator,
        inputs.model_path,
        system.device,
        metric_mode=args.metric_mode,
        sens_mode=sens_mode,
        sensing=sensing,
    )
    _render_scene_png(rt_simulator, scene_slug, OUT_DIR / "data_loading")
    _log_sensing_performance(system.components)

    sensing_estimator = system.components.sensing_estimator
    assert sensing_estimator is not None  # _preflight_checks 已保证

    range_rmses, velocity_rmses = _run_evaluation_loop(
        dataset,
        system,
        setup,
        sensing_estimator,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )

    n_episodes = len(dataset)
    print(f"性能评估完成: {n_episodes}/{n_episodes} episodes ({setup.label})")
    _print_rmse_summary(
        range_rmses,
        velocity_rmses,
        estimator_label=setup.label,
        n_episodes=n_episodes,
        sens_mode=sens_mode,
    )


if __name__ == "__main__":
    main()
