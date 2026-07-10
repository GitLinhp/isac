"""训练时延–多普勒 Sensing CNN：原始复数谱 → 局部 bin 回归（单/双基地）。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/model_training/run_train_sensing_cnn.py

双基地示例::

    python script/model_training/run_train_sensing_cnn.py \\
        --dataset_h5 data/empty_room_bistatic_30kHz/empty_room_bistatic_mc_sionna_dataset.h5 \\
        --config_file config/data_collection/data_collection_bistatic.toml \\
        --sens_mode bistatic

数据流
------
1. ``RTDataset.load`` 返回原始 ``spectrum_tensor`` 与运动学
2. ``kinematics_to_target_bins(sens_mode=...)`` 生成 ``(B, 2)`` 局部 bin 监督
3. ``SensingCNN.forward`` 输出可微 ``(B, 2)`` 预测
4. ``MonostaticSensingLoss`` 在 bin 空间优化
5. 验证经 ``SensingEstimator(sens_mode=...)`` 报告物理 RMSE

``--sens_mode``
---------------
- ``monostatic``（默认）：斜距/径向速度标签与 ROI
- ``bistatic``：折叠路径长/路径变化率；须分离收发 TOML

前置条件
--------
HDF5 同目录须存在唯一 ``data_collection*.toml``（或通过 ``--config_file`` 指定）。

输出产物
--------
默认 ``models/sensing_cnn/{sens_mode}/best_model.pth`` 及同目录检查点/曲线。

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

from isac import (
    DEFAULT_BISTATIC_SENSING_CNN_MODEL,
    DEFAULT_DATASET_H5,
    DEFAULT_SENSING_CNN_MODEL,
)
from isac.collection import RTDataset, sensing_attrs_from_system
from isac.data_structures.types import MusicPeaks, SensMode
from isac.models import (
    MonostaticSensingLoss,
    SensingCNN,
    kinematics_to_range_velocity,
    kinematics_to_target_bins,
)
from isac.sensing.spectrum.sensing_performance import SensingPerformance
from isac.system import System
from isac.utils import set_random_seed

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator
    from isac.sensing.evaluation.sensing_estimator import SensingEstimator


@dataclass(frozen=True)
class TrainInputs:
    """校验后的训练输入路径。"""

    h5_path: Path
    config_path: Path
    sens_mode: SensMode


@dataclass(frozen=True)
class TrainPaths:
    """训练产物输出路径。"""

    best_model: Path
    checkpoint_dir: Path
    training_curve: Path
    checkpoint_final: Path


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


def _resolve_train_inputs(args: argparse.Namespace) -> TrainInputs:
    """校验 HDF5 与采集配置 TOML 路径。"""
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    config_path = _resolve_config_path(h5_path, args.config_file)
    return TrainInputs(
        h5_path=h5_path,
        config_path=config_path,
        sens_mode=args.sens_mode,
    )


def _assert_bistatic_topology(rt_simulator: RTSimulator) -> None:
    """校验 TOML 为分离收发拓扑，且首条链路为双基地。"""
    geom = rt_simulator.rx_target_tx_geometric
    if len(geom.tx_names) < 1 or len(geom.rx_names) < 1:
        raise ValueError(
            f"双基地训练需要至少 1 个 TX 与 1 个 RX，"
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
        raise ValueError("双基地训练要求 channel.type='rt'")
    _assert_bistatic_topology(rt_simulator)
    geom = rt_simulator.rx_target_tx_geometric
    tx_pos = rt_simulator.tx_states[geom.tx_names[0]][0]
    return torch.tensor(tx_pos, dtype=torch.float32)


def _collate_batch(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """将单样本 dict 堆叠为 batch 张量。"""
    return {
        "spectrum_tensor": torch.stack([s["spectrum_tensor"] for s in samples], dim=0),
        "target_position": torch.stack([s["target_position"] for s in samples], dim=0),
        "target_velocity": torch.stack([s["target_velocity"] for s in samples], dim=0),
        "bs_pos": torch.stack([s["bs_pos"] for s in samples], dim=0),
        "slot": torch.stack([s["slot"] for s in samples], dim=0),
    }


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 batch 搬至 device；``bs`` 取首条（同一数据集内 RX/bs1 位置恒定）。"""
    spectrum = batch["spectrum_tensor"].to(device)
    pos = batch["target_position"].to(device)
    vel = batch["target_velocity"].to(device)
    bs = batch["bs_pos"][0].to(device)
    return spectrum, pos, vel, bs


def _predict_and_target_bins(
    model: SensingCNN,
    spectrum: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    bs: torch.Tensor,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CNN 前向 + 运动学监督：返回 ``(y_bins, target_bins)``。"""
    y_bins = model(spectrum)
    target_bins = kinematics_to_target_bins(
        pos,
        vel,
        bs,
        sensing_performance=sensing_performance,
        num_doppler_bins=num_doppler_bins,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    return y_bins, target_bins


def _bins_row_to_peaks(
    bins: torch.Tensor,
    row: int,
    device: torch.device | str,
) -> MusicPeaks:
    """单行局部 bin → ``MusicPeaks``。"""
    return MusicPeaks.from_local_bins(bins[row, 0], bins[row, 1], device=device)


def _train_step(
    batch: dict[str, torch.Tensor],
    model: SensingCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
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
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    loss = criterion(y_bins, target_bins)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _evaluate(
    model: SensingCNN,
    loader: DataLoader,
    criterion: MonostaticSensingLoss,
    sensing_estimator: SensingEstimator,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> tuple[float, float, float]:
    """验证集：bin MSE 损失 + 经 SensingEstimator 的物理 RMSE。"""
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
            sens_mode=sens_mode,
            tx_pos=tx_pos,
        )
        total_loss += criterion(y_bins, target_bins).item() * spectrum.size(0)

        y_range, y_vel = kinematics_to_range_velocity(
            pos, vel, bs, sens_mode=sens_mode, tx_pos=tx_pos
        )
        for i in range(spectrum.size(0)):
            peaks = _bins_row_to_peaks(y_bins, i, device)
            estimate = sensing_estimator(
                peaks,
                sens_mode=sens_mode,
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


def _build_dataloaders(
    full_ds: RTDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int, int]:
    """划分 train/val 并构建 DataLoader。"""
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


def _build_model_and_optim(
    device: torch.device | str,
    lr: float,
    weight_decay: float,
    num_layers: int,
    *,
    base_channels: int,
    dropout: float,
) -> tuple[SensingCNN, torch.optim.Optimizer, MonostaticSensingLoss, int]:
    """构建 CNN、Adam 与 bin 空间损失。"""
    in_channels = 2
    model = SensingCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MonostaticSensingLoss()
    return model, optimizer, criterion, in_channels


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """``val_loss`` 停滞时按 ``lr_factor`` 衰减学习率。"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )


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


def _range_rmse_label(sens_mode: SensMode) -> str:
    """验证 RMSE 日志中的距离轴文案。"""
    if sens_mode == "bistatic":
        return "RMSE_fold_path"
    return "RMSE_range"


def _log_train_banner(
    inputs: TrainInputs,
    paths: TrainPaths,
    sensing: dict[str, Any],
    n_train: int,
    n_val: int,
    *,
    num_layers: int,
    base_channels: int,
    dropout: float,
    early_stopping_patience: int,
) -> None:
    """打印数据集、ROI 与输出路径摘要。"""
    early_stop_label = (
        f"{early_stopping_patience} epoch" if early_stopping_patience > 0 else "关"
    )
    range_label = "折叠路径长" if inputs.sens_mode == "bistatic" else "斜距"
    print(
        f"数据集: {inputs.h5_path} | 配置: {inputs.config_path}\n"
        f"sens_mode={inputs.sens_mode} | 训练 {n_train} / 验证 {n_val} | "
        f"ROI max_{range_label}={sensing['max_range_m']:.1f} m, "
        f"±max_velocity={sensing['max_velocity_mps']:.1f} m/s | "
        f"Δr={sensing['range_resolution']:.3f} m, "
        f"Δv={sensing['velocity_resolution']:.3f} m/s\n"
        f"模型 num_layers={num_layers}, base_channels={base_channels}, "
        f"dropout={dropout}, early_stopping={early_stop_label}\n"
        f"检查点目录: {paths.checkpoint_dir} | 曲线: {paths.training_curve}"
    )


def _checkpoint_payload(
    model: SensingCNN,
    *,
    in_channels: int,
) -> dict[str, Any]:
    """构造 checkpoint 字典（仅权重与结构超参）。"""
    return {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }


def _plot_training_history(
    history: dict[str, list[float]],
    path: Path,
    *,
    sens_mode: SensMode,
) -> None:
    """绘制训练/验证损失与物理 RMSE 曲线并保存 PNG。"""
    if not history["epoch"]:
        return

    range_label = "Fold path RMSE (m)" if sens_mode == "bistatic" else "Range RMSE (m)"
    fig, (ax_loss, ax_rmse) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_rmse.plot(epochs, history["val_rmse_range_m"], label=range_label)
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
    model: SensingCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    device: torch.device | str,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
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
            sens_mode=sens_mode,
            tx_pos=tx_pos,
        )
        train_loss += loss_item
        batch_bar.set_postfix(loss=f"{loss_item:.4f}")
    return train_loss / max(len(train_loader), 1)


def _run_training_epochs(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SensingCNN,
    criterion: MonostaticSensingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    sensing_estimator: SensingEstimator,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    in_channels: int,
    paths: TrainPaths,
    device: torch.device | str,
    *,
    sens_mode: SensMode,
    tx_pos: torch.Tensor | None,
) -> float:
    """多 epoch 训练、验证、检查点与曲线更新；返回最优 val_loss。"""
    best_val_loss = float("inf")
    epochs_without_improve = 0
    range_log = _range_rmse_label(sens_mode)
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
            sens_mode=sens_mode,
            tx_pos=tx_pos,
        )
        val_loss, val_rmse_r, val_rmse_v = _evaluate(
            model,
            val_loader,
            criterion,
            sensing_estimator,
            sensing_performance,
            num_doppler_bins,
            device,
            sens_mode=sens_mode,
            tx_pos=tx_pos,
        )
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        if cur_lr < prev_lr:
            tqdm.write(f"lr reduced: {prev_lr:.2e} -> {cur_lr:.2e}")
        tqdm.write(
            f"epoch {epoch:03d} | train_loss={mean_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | {range_log}={val_rmse_r:.3f} m | "
            f"RMSE_vel={val_rmse_v:.3f} m/s | lr={cur_lr:.2e}"
        )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse_range_m"].append(val_rmse_r)
        history["val_rmse_velocity_mps"].append(val_rmse_v)

        payload = _checkpoint_payload(model, in_channels=in_channels)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(payload, paths.best_model)
        else:
            epochs_without_improve += 1

        is_periodic = epoch % args.save_every == 0
        is_final = epoch == args.epochs
        if is_periodic:
            torch.save(payload, paths.checkpoint_dir / f"checkpoint_{epoch:03d}.pth")
        if is_periodic or is_final:
            _plot_training_history(history, paths.training_curve, sens_mode=sens_mode)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improve >= args.early_stopping_patience
        ):
            tqdm.write(
                f"early stopping at epoch {epoch} "
                f"(best val_loss={best_val_loss:.4f})"
            )
            break

    torch.save(
        _checkpoint_payload(model, in_channels=in_channels),
        paths.checkpoint_final,
    )
    _plot_training_history(history, paths.training_curve, sens_mode=sens_mode)
    return best_val_loss


def argument_parser() -> argparse.Namespace:
    """构造训练脚本的全部 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="ISAC — 训练 Sensing CNN（h_dd → 距离/速度，mono/bistatic）"
    )

    data_group = parser.add_argument_group("数据与划分")
    data_group.add_argument(
        "--dataset_h5",
        type=Path,
        default=DEFAULT_DATASET_H5,
        help="HDF5 数据集路径",
    )
    data_group.add_argument(
        "--config_file",
        type=Path,
        default=None,
        help="采集 TOML；默认在 HDF5 同目录查找 data_collection*.toml（须唯一）",
    )
    data_group.add_argument(
        "--sens_mode",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="感知模式；影响标签 bin 换算、验证 RMSE 与默认输出目录",
    )
    data_group.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")

    train_group = parser.add_argument_group("训练超参")
    train_group.add_argument("--epochs", type=int, default=200, help="训练轮数")
    train_group.add_argument("--batch_size", type=int, default=64, help="批大小")
    train_group.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    train_group.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Adam L2 系数"
    )
    train_group.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="val_loss 连续多少 epoch 无改善后衰减学习率",
    )
    train_group.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="学习率衰减倍率（lr *= factor）",
    )
    train_group.add_argument("--lr_min", type=float, default=1e-6, help="学习率下限")
    train_group.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="训练设备",
    )
    train_group.add_argument("--seed", type=int, default=42, help="随机种子")
    train_group.add_argument(
        "--num_layers", type=int, default=3, help="残差编码块数量（>= 1）"
    )
    train_group.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="stem 与首层残差基础通道数",
    )
    train_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="回归头 Dropout 概率",
    )
    train_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="val_loss 连续多少 epoch 无改善后早停；0 表示禁用",
    )

    out_group = parser.add_argument_group("输出与检查点")
    out_group.add_argument(
        "--output",
        type=Path,
        default=None,
        help="val_loss 最优模型保存路径；默认 models/sensing_cnn/{sens_mode}/best_model.pth",
    )
    out_group.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="每隔多少 epoch 保存周期性检查点并更新训练曲线",
    )

    return parser.parse_args()


def main() -> None:
    """加载数据 → 构建模型 → epoch 循环 → 保存检查点与曲线。"""
    args = argument_parser()
    if args.save_every < 1:
        raise ValueError("save_every 须 >= 1")
    if args.num_layers < 1:
        raise ValueError("num_layers 须 >= 1")
    if args.base_channels < 1:
        raise ValueError("base_channels 须 >= 1")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout 须在 [0, 1) 内")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 须 >= 0")

    if args.output is None:
        args.output = (
            DEFAULT_BISTATIC_SENSING_CNN_MODEL
            if args.sens_mode == "bistatic"
            else DEFAULT_SENSING_CNN_MODEL
        )

    inputs = _resolve_train_inputs(args)
    set_random_seed(args.seed)
    device = args.device
    sens_mode = inputs.sens_mode

    system = System(inputs.config_path, device=device)
    sensing_performance = system.components.sensing_performance
    sensing_estimator = system.components.sensing_estimator
    if sensing_performance is None:
        raise ValueError("训练需要 sensing_performance（[ofdm] + carrier_frequency）")
    if sensing_estimator is None:
        raise ValueError("验证 RMSE 需要 TOML [music] 段以构建 SensingEstimator")

    sensing = sensing_attrs_from_system(system, sens_mode=sens_mode)
    num_doppler_bins = int(sensing["num_doppler_bins"])
    tx_pos_tensor = _resolve_tx_pos(system, sens_mode)
    tx_pos = tx_pos_tensor.to(device) if tx_pos_tensor is not None else None

    full_ds = RTDataset.load(inputs.h5_path)
    train_loader, val_loader, n_train, n_val = _build_dataloaders(full_ds, args)
    model, optimizer, criterion, in_channels = _build_model_and_optim(
        device,
        args.lr,
        args.weight_decay,
        args.num_layers,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    scheduler = _build_lr_scheduler(optimizer, args)

    paths = _train_paths(Path(args.output))
    _log_train_banner(
        inputs,
        paths,
        sensing,
        n_train,
        n_val,
        num_layers=args.num_layers,
        base_channels=args.base_channels,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
    )

    best_val_loss = _run_training_epochs(
        args,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        sensing_estimator,
        sensing_performance,
        num_doppler_bins,
        in_channels,
        paths,
        device,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )

    print(
        f"最优 val_loss={best_val_loss:.4f}，模型已保存至 {paths.best_model.resolve()}"
    )
    print(f"最终检查点已保存至 {paths.checkpoint_final.resolve()}")
    print(f"训练曲线已保存至 {paths.training_curve.resolve()}")


if __name__ == "__main__":
    main()
