"""训练单基地时延–多普勒 CNN：原始复数谱 → 局部 bin 回归。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/model_training/run_train_monostatic_cnn.py

数据流
------
1. ``RTDataset.load`` 返回原始 ``spectrum_tensor`` 与运动学
2. ``kinematics_to_target_bins`` 从运动学生成 ``(B, 2)`` 局部 bin 监督
3. ``MonostaticDelayDopplerCNN.forward`` 输出可微 ``(B, 2)`` 预测
4. ``MonostaticSensingLoss`` 在 bin 空间优化
5. 验证除 bin 损失外，经 ``SensingEstimator`` 报告物理距离/速度 RMSE

流程概要
--------
1. 解析 CLI，加载 HDF5 与同目录 ``data_collection.toml``
2. ``random_split`` 划分 train/val，``DataLoader``
3. Adam 训练；``val_loss`` 最小时保存 ``best_model.pth``
4. 每 ``--save_every`` epoch 保存周期性检查点并更新训练曲线

前置条件
--------
``--dataset_h5`` 所在目录须存在 ``data_collection.toml``（与采集脚本落盘副本一致），
用于构建 ``System``、``SensingPerformance`` 与验证用 ``SensingEstimator``。

输出产物
--------
由 ``--output``（默认 ``DEFAULT_MONOSTATIC_CNN_MODEL``）推导：
最优 ``best_model.pth``、``checkpoints/checkpoint_XXX.pth``、
``training_curve.png``、``checkpoint_final.pth``。

下游评估
--------
训练完成后可用 ``script/evaluation/run_sensing_from_dataset.py --estimator model`` 回放 RMSE。

checkpoint 约定
--------------
仅写入 ``model_state_dict``、``in_channels``、``base_channels``、``dropout``；
ROI / 分辨率等感知参数由 TOML 提供，不序列化到 checkpoint。

详见 ``docs/monostatic_delay_doppler_cnn.md``。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from isac import DEFAULT_DATASET_H5, DEFAULT_MONOSTATIC_CNN_MODEL
from isac.collection import RTDataset, sensing_attrs_from_system
from isac.data_structures.types import MusicPeaks
from isac.models import (
    MonostaticDelayDopplerCNN,
    MonostaticSensingLoss,
    kinematics_to_range_velocity,
    kinematics_to_target_bins,
)
from isac.sensing.spectrum.sensing_performance import SensingPerformance
from isac.system import System
from isac.utils import load_config

if TYPE_CHECKING:
    from isac.sensing.evaluation.sensing_estimator import SensingEstimator


@dataclass(frozen=True)
class TrainInputs:
    """校验后的训练输入路径。

    ``config_path`` 固定为 ``h5_path.parent / "data_collection.toml"``。
    """

    h5_path: Path
    config_path: Path


@dataclass(frozen=True)
class TrainPaths:
    """训练产物输出路径。

    - ``best_model``：``val_loss`` 最优 checkpoint（``--output``）
    - ``checkpoint_dir``：周期性 ``checkpoint_XXX.pth`` 目录
    - ``training_curve``：损失与 RMSE 曲线 PNG
    - ``checkpoint_final``：最终 epoch 权重
    """


def argument_parser() -> argparse.Namespace:
    """构造训练脚本的全部 CLI 参数。

    ``data_collection.toml`` 固定解析为 HDF5 同目录下的副本（见 ``_resolve_train_inputs``）。
    """
    parser = argparse.ArgumentParser(
        description="ISAC — 训练单基地时延–多普勒 CNN（h_dd → 距离/速度）"
    )

    # --- 数据与划分 ---
    data_group = parser.add_argument_group("数据与划分")
    data_group.add_argument(
        "--dataset_h5",
        type=Path,
        default=DEFAULT_DATASET_H5,
        help="HDF5 数据集路径",
    )
    data_group.add_argument(
        "--val_ratio", type=float, default=0.2, help="验证集比例"
    )

    # --- 训练超参 ---
    train_group = parser.add_argument_group("训练超参")
    train_group.add_argument("--epochs", type=int, default=200, help="训练轮数")
    train_group.add_argument("--batch_size", type=int, default=64, help="批大小")
    train_group.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    train_group.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="训练设备",
    )
    train_group.add_argument("--seed", type=int, default=42, help="随机种子")

    # --- 输出与检查点 ---
    out_group = parser.add_argument_group(
        "输出与检查点",
        "val_loss 最优写入 --output；每 save_every epoch 写周期性 checkpoint 并刷新训练曲线",
    )
    out_group.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MONOSTATIC_CNN_MODEL,
        help="val_loss 最优模型保存路径",
    )
    out_group.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="每隔多少 epoch 保存周期性检查点并更新训练曲线",
    )

    return parser.parse_args()


# --- 路径解析与预检 ---


def _resolve_train_inputs(args: argparse.Namespace) -> TrainInputs:
    """校验 HDF5 与采集配置 TOML 路径。"""
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    config_path = h5_path.parent / "data_collection.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    return TrainInputs(h5_path=h5_path, config_path=config_path)


def _preflight_training(
    system: System,
) -> tuple[SensingEstimator, SensingPerformance]:
    """校验标签生成与物理 RMSE 评估所需组件已构建。

    - ``SensingEstimator``（TOML ``[music]``）：验证集物理 RMSE
    - ``SensingPerformance``（TOML ``[ofdm]`` 等）：``kinematics_to_target_bins`` 标签
    """
    sensing_estimator = system.components.sensing_estimator
    if sensing_estimator is None:
        raise ValueError("训练验证 RMSE 需要 TOML [music] 段以构建 SensingEstimator")

    sensing_performance = system.components.sensing_performance
    if sensing_performance is None:
        raise ValueError("训练标签生成需要 TOML [ofdm] 与 [carrier_frequency]")

    return sensing_estimator, sensing_performance


# --- DataLoader 与 batch ---


def _collate_batch(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """将单样本 dict 堆叠为 batch 张量。

    键与 ``RTDataset.__getitem__`` 一致；``bs_pos`` 每样本重复存储，训练时取 ``[0]``。
    """
    return {
        "spectrum_tensor": torch.stack(
            [s["spectrum_tensor"] for s in samples], dim=0
        ),
        "target_position": torch.stack(
            [s["target_position"] for s in samples], dim=0
        ),
        "target_velocity": torch.stack(
            [s["target_velocity"] for s in samples], dim=0
        ),
        "bs_pos": torch.stack([s["bs_pos"] for s in samples], dim=0),
        "slot": torch.stack([s["slot"] for s in samples], dim=0),
    }


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 batch 搬至 device；``bs`` 取首条（同一数据集内基站位置恒定）。"""
    spectrum = batch["spectrum_tensor"].to(device)
    pos = batch["target_position"].to(device)
    vel = batch["target_velocity"].to(device)
    bs = batch["bs_pos"][0].to(device)
    return spectrum, pos, vel, bs


# --- 训练核心（bin 监督 + 物理 RMSE）---


def _predict_and_target_bins(
    model: MonostaticDelayDopplerCNN,
    spectrum: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    bs: torch.Tensor,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CNN 前向 + 运动学监督：返回 ``(y_bins, target_bins)``，形状均为 ``(B, 2)``。"""
    y_bins = model(spectrum)
    target_bins = kinematics_to_target_bins(
        pos,
        vel,
        bs,
        sensing_performance=sensing_performance,
        num_doppler_bins=num_doppler_bins,
    )
    return y_bins, target_bins


def _bins_row_to_peaks(
    bins: torch.Tensor,
    row: int,
    device: torch.device | str,
) -> MusicPeaks:
    """单行局部 bin ``(2,)`` → ``MusicPeaks``，供 ``SensingEstimator`` 换算物理量。"""
    return MusicPeaks.from_local_bins(
        bins[row, 0], bins[row, 1], device=device
    )


def _train_step(
    batch: dict[str, torch.Tensor],
    model: MonostaticDelayDopplerCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
) -> float:
    """单 batch 前向、反传与参数更新，返回标量 loss。"""
    spectrum, pos, vel, bs = _batch_to_device(batch, device)
    optimizer.zero_grad()
    y_bins, target_bins = _predict_and_target_bins(
        model,
        spectrum,
        pos,
        vel,
        bs,
        sensing_performance,
        num_doppler_bins,
    )
    loss = criterion(y_bins, target_bins)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _evaluate(
    model: MonostaticDelayDopplerCNN,
    loader: DataLoader,
    criterion: MonostaticSensingLoss,
    sensing_estimator: SensingEstimator,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
) -> tuple[float, float, float]:
    """验证集：bin MSE 损失 + 经 SensingEstimator 的物理 RMSE。"""
    # val_loss 在 bin 空间；RMSE 将预测 bin 经 SensingEstimator 换为物理量，
    # 真值用 kinematics_to_range_velocity（与 run_sensing_from_dataset 一致）。
    model.eval()
    total_loss = 0.0
    range_sq = 0.0
    vel_sq = 0.0
    n = 0

    for batch in loader:
        spectrum, pos, vel, bs = _batch_to_device(batch, device)
        y_bins, target_bins = _predict_and_target_bins(
            model,
            spectrum,
            pos,
            vel,
            bs,
            sensing_performance,
            num_doppler_bins,
        )
        total_loss += criterion(y_bins, target_bins).item() * spectrum.size(0)

        y_range, y_vel = kinematics_to_range_velocity(pos, vel, bs)
        # SensingEstimator 按单峰 batch 接口，逐样本 MusicPeaks 换算
        for i in range(spectrum.size(0)):
            peaks = _bins_row_to_peaks(y_bins, i, device)
            estimate = sensing_estimator(
                peaks,
                sens_mode="monostatic",
                log_peaks=False,
            )
            range_sq += (estimate.est_ranges[0] - y_range[i]) ** 2
            vel_sq += (estimate.est_velocities[0] - y_vel[i]) ** 2
        n += spectrum.size(0)

    if n == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / n,
        math.sqrt(range_sq / n),
        math.sqrt(vel_sq / n),
    )


# --- train/val 划分与 DataLoader 构建 ---


def _build_dataloaders(
    full_ds: RTDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int, int]:
    """划分 train/val 并构建 DataLoader（``args.seed`` 固定 random_split 可复现）。"""
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    if n_train < 1:
        raise ValueError(
            f"训练样本不足：len={len(full_ds)}, val_ratio={args.val_ratio}"
        )

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "collate_fn": _collate_batch,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, n_train, n_val


# --- 模型、检查点与可视化 ---


def _build_model_and_optim(
    device: torch.device | str,
    lr: float,
) -> tuple[MonostaticDelayDopplerCNN, torch.optim.Optimizer, MonostaticSensingLoss, int]:
    """构建 CNN、Adam 与 bin 空间损失。

    CNN 仅 ``in_channels=2``，不传入 ``sensing_attrs``；感知参数仅用于标签与日志。
    """
    in_channels = 2
    model = MonostaticDelayDopplerCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MonostaticSensingLoss()
    return model, optimizer, criterion, in_channels


def _train_paths(output: Path) -> TrainPaths:
    """由 ``--output`` 推导检查点与曲线路径。"""
    run_dir = output.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return TrainPaths(
        best_model=output,
        checkpoint_dir=ckpt_dir,
        training_curve=run_dir / "training_curve.png",
        checkpoint_final=run_dir / "checkpoint_final.pth",
    )


def _log_train_banner(
    inputs: TrainInputs,
    paths: TrainPaths,
    sensing: dict[str, Any],
    n_train: int,
    n_val: int,
) -> None:
    """打印数据集、ROI 与输出路径摘要。"""
    print(
        f"数据集: {inputs.h5_path} | 配置: {inputs.config_path}\n"
        f"训练 {n_train} / 验证 {n_val} | "
        f"ROI max_range={sensing['max_range_m']:.1f} m, "
        f"±max_velocity={sensing['max_velocity_mps']:.1f} m/s | "
        f"Δr={sensing['range_resolution']:.3f} m, "
        f"Δv={sensing['velocity_resolution']:.3f} m/s\n"
        f"检查点目录: {paths.checkpoint_dir} | 曲线: {paths.training_curve}"
    )


def _checkpoint_payload(
    model: MonostaticDelayDopplerCNN,
    *,
    in_channels: int,
) -> dict[str, Any]:
    """构造 checkpoint 字典（仅权重与结构超参）。

    键：``model_state_dict``、``in_channels``、``base_channels``、``dropout``。
    不含 ``epoch``、``config_file`` 及感知 ROI/分辨率字段。
    """
    return {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "dropout": model.dropout,
    }


def _plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    """绘制训练/验证损失与物理 RMSE 曲线并保存 PNG。"""
    if not history["epoch"]:
        return

    fig, (ax_loss, ax_rmse) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_rmse.plot(epochs, history["val_rmse_range_m"], label="Range RMSE (m)")
    ax_rmse.plot(epochs, history["val_rmse_velocity_mps"], label="Velocity RMSE (m/s)")
    ax_rmse.set_xlabel("Epoch")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _train_one_epoch(
    epoch: int,
    epochs: int,
    train_loader: DataLoader,
    model: MonostaticDelayDopplerCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
) -> float:
    """单 epoch 训练，返回平均 batch loss。"""
    model.train()
    train_loss = 0.0
    batch_bar = tqdm(
        train_loader,
        desc=f"epoch {epoch:03d}/{epochs:03d}",
        unit="batch",
    )
    for batch in batch_bar:
        loss_item = _train_step(
            batch,
            model,
            criterion,
            optimizer,
            sensing_performance,
            num_doppler_bins,
            device,
        )
        train_loss += loss_item
        batch_bar.set_postfix(loss=f"{loss_item:.4f}")
    return train_loss / max(len(train_loader), 1)


def _run_training_epochs(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: MonostaticDelayDopplerCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    sensing_estimator: SensingEstimator,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    in_channels: int,
    paths: TrainPaths,
    device: torch.device | str,
) -> float:
    """多 epoch 训练、验证、检查点与曲线更新；返回最优 val_loss。"""
    best_val_loss = float("inf")
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_rmse_range_m": [],
        "val_rmse_velocity_mps": [],
    }

    for epoch in range(1, args.epochs + 1):
        mean_train_loss = _train_one_epoch(
            epoch,
            args.epochs,
            train_loader,
            model,
            criterion,
            optimizer,
            sensing_performance,
            num_doppler_bins,
            device,
        )
        val_loss, val_rmse_r, val_rmse_v = _evaluate(
            model,
            val_loader,
            criterion,
            sensing_estimator,
            sensing_performance,
            num_doppler_bins,
            device,
        )
        tqdm.write(
            f"epoch {epoch:03d} | train_loss={mean_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | RMSE_range={val_rmse_r:.3f} m | "
            f"RMSE_vel={val_rmse_v:.3f} m/s"
        )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse_range_m"].append(val_rmse_r)
        history["val_rmse_velocity_mps"].append(val_rmse_v)

        payload = _checkpoint_payload(model, in_channels=in_channels)

        # val_loss 创新低时覆盖 --output（最优模型）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, paths.best_model)

        is_periodic = epoch % args.save_every == 0
        is_final = epoch == args.epochs
        if is_periodic:
            torch.save(payload, paths.checkpoint_dir / f"checkpoint_{epoch:03d}.pth")
        # 周期性或最终 epoch 刷新训练曲线
        if is_periodic or is_final:
            _plot_training_history(history, paths.training_curve)

    # 训练结束另存最终 checkpoint 并再次刷新曲线
    torch.save(
        _checkpoint_payload(model, in_channels=in_channels),
        paths.checkpoint_final,
    )
    _plot_training_history(history, paths.training_curve)
    return best_val_loss


def main() -> None:
    """训练入口：加载数据 → 构建模型 → epoch 循环 → 保存检查点与曲线。"""
    # --- 初始化：CLI 校验与随机种子 ---
    args = argument_parser()
    if args.save_every < 1:
        raise ValueError("save_every 须 >= 1")

    inputs = _resolve_train_inputs(args)
    torch.manual_seed(args.seed)
    device = args.device

    # --- System 预检：标签生成 + 验证 RMSE 所需组件 ---
    system = System(load_config(inputs.config_path), device=device)
    sensing_estimator, sensing_performance = _preflight_training(system)
    sensing = sensing_attrs_from_system(system)  # 标签 num_doppler_bins 与日志用
    num_doppler_bins = int(sensing["num_doppler_bins"])

    # --- 数据与模型：RTDataset、DataLoader、CNN/Adam/Loss ---
    full_ds = RTDataset.load(inputs.h5_path)
    train_loader, val_loader, n_train, n_val = _build_dataloaders(full_ds, args)
    model, optimizer, criterion, in_channels = _build_model_and_optim(
        device, args.lr
    )

    paths = _train_paths(Path(args.output))
    _log_train_banner(inputs, paths, sensing, n_train, n_val)

    # --- 训练循环：epoch 训练、验证、检查点与曲线 ---
    best_val_loss = _run_training_epochs(
        args,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        sensing_estimator,
        sensing_performance,
        num_doppler_bins,
        in_channels,
        paths,
        device,
    )

    # --- 收尾：打印最优与最终产物路径 ---
    print(f"最优 val_loss={best_val_loss:.4f}，模型已保存至 {paths.best_model.resolve()}")
    print(f"最终检查点已保存至 {paths.checkpoint_final.resolve()}")
    print(f"训练曲线已保存至 {paths.training_curve.resolve()}")


if __name__ == "__main__":
    main()
