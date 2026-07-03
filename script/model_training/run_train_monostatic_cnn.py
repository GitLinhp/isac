"""训练单基地时延–多普勒 CNN：HDF5 CFR → h_dd 特征 → 距离/速度回归。

须在 **ISAC conda 环境**中运行（Sionna RT 与 OFDM 感知链依赖完整 CUDA/Sionna 栈）::

    /opt/miniconda3/envs/ISAC/bin/python script/model_training/run_train_monostatic_cnn.py
    # 或
    conda run -n ISAC python script/model_training/run_train_monostatic_cnn.py

数据流与 ``run_sensing_from_dataset.py`` 一致：存储 CFR → transmit → channel →
``compute_sensing_spectrum`` → ``dd_spectrum_to_features`` → CNN。

流程概要
--------
1. 解析路径/CLI，构建 ``MonostaticSensingTorchDataset``（在线 h_dd 特征）
2. ``random_split`` 划分 train/val，``DataLoader``（``num_workers=0``）
3. ``MonostaticDelayDopplerCNN`` + ``MonostaticSensingLoss`` + Adam 训练
4. 每 ``--save_every`` epoch 保存 ``checkpoints/checkpoint_XXX.pth`` 并更新 ``training_curve.png``
5. ``val_loss`` 最小时保存 ``best_model.pth``；训练结束保存 ``checkpoint_final.pth`` 与最终曲线
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from isac import DEFAULT_DATASET_H5, DEFAULT_MONOSTATIC_CNN_MODEL
from isac.models import (
    MonostaticDelayDopplerCNN,
    MonostaticSensingLoss,
    MonostaticSensingTorchDataset,
)


def argument_parser() -> argparse.Namespace:
    """构造训练脚本的全部 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="ISAC — 训练单基地时延–多普勒 CNN（h_dd → 距离/速度）"
    )
    parser.add_argument(
        "--dataset_h5",
        type=Path,
        default=DEFAULT_DATASET_H5,
        help="HDF5 数据集路径",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--offset", type=int, default=128, help="DD 谱 ROI 半宽 (bin)")

    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="训练设备",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MONOSTATIC_CNN_MODEL,
        help="val_loss 最优模型保存路径",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="每隔多少 epoch 保存周期性检查点并更新训练曲线",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def _collate_batch(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """将单样本 dict 堆叠为 batch 张量。

    输入每项含 ``features``、``range_m``、``velocity_mps``、``slot``。
    """
    return {
        "features": torch.stack([s["features"] for s in samples], dim=0),
        "range_m": torch.stack([s["range_m"] for s in samples], dim=0),
        "velocity_mps": torch.stack([s["velocity_mps"] for s in samples], dim=0),
        "slot": torch.stack([s["slot"] for s in samples], dim=0),
    }


def _checkpoint_payload(
    model: MonostaticDelayDopplerCNN,
    *,
    epoch: int,
    in_channels: int,
    full_ds: MonostaticSensingTorchDataset,
    h5_path: Path,
    config_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """构造 checkpoint 字典（周期性 / 最优 / 最终保存复用）。"""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "dropout": model.dropout,
        "range_resolution": model.range_resolution,
        "velocity_resolution": model.velocity_resolution,
        "offset": model.offset,
        "use_phase": full_ds.use_phase,
        "dataset_h5": str(h5_path),
        "config_file": str(config_path),
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


@torch.no_grad()
def _evaluate(
    model: MonostaticDelayDopplerCNN,
    loader: DataLoader,
    criterion: MonostaticSensingLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """验证集评估。

    输入:
    ----------
    - model : 已移动至 ``device`` 的 ``MonostaticDelayDopplerCNN``
    - loader : 验证集 ``DataLoader``
    - criterion : 已移动至 ``device`` 的 ``MonostaticSensingLoss``
    - device : 评估设备

    返回:
    ----------
    - val_loss : bin 空间复合损失（batch 均值）
    - rmse_range_m : 距离物理 RMSE (m)
    - rmse_velocity_mps : 速度物理 RMSE (m/s)
    """
    model.eval()
    total_loss = 0.0
    range_sq = 0.0
    vel_sq = 0.0
    n = 0
    for batch in loader:
        x = batch["features"].to(device)
        y_range = batch["range_m"].to(device)
        y_vel = batch["velocity_mps"].to(device)
        y_bins = model.forward_bins(x)
        target_bins = MonostaticSensingLoss.target_bins_from_physical_labels(
            y_range,
            y_vel,
            range_resolution=model.range_resolution,
            velocity_resolution=model.velocity_resolution,
        )
        total_loss += criterion(y_bins, target_bins).item() * x.size(0)
        pred = model.bins_to_physical(y_bins)
        range_sq += torch.sum((pred[:, 0] - y_range) ** 2).item()
        vel_sq += torch.sum((pred[:, 1] - y_vel) ** 2).item()
        n += x.size(0)
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, (range_sq / n) ** 0.5, (vel_sq / n) ** 0.5


def main() -> None:
    """训练入口：数据集 → 模型 → epoch 循环 → 检查点与训练曲线。"""
    args = argument_parser()
    if args.save_every < 1:
        raise ValueError("save_every 须 >= 1")

    # --- 路径与配置 ---
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")
    config_path = h5_path.parent / "data_collection.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    torch.manual_seed(args.seed)
    device = args.device

    # --- 数据集与 DataLoader（num_workers=0：Sionna/System 不宜多进程）---
    full_ds = MonostaticSensingTorchDataset(
        h5_path,
        config_file=config_path,
        offset=args.offset,
        device=device,
    )
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

    # --- 模型、损失与优化器 ---
    in_channels = 2 if full_ds.use_phase else 1
    model = MonostaticDelayDopplerCNN(
        in_channels=in_channels,
        range_resolution=full_ds.range_resolution,
        velocity_resolution=full_ds.velocity_resolution,
        offset=args.offset,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MonostaticSensingLoss()

    out_path = Path(args.output)
    run_dir = out_path.parent
    ckpt_dir = run_dir / "checkpoints"
    curve_path = run_dir / "training_curve.png"
    final_path = run_dir / "checkpoint_final.pth"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"数据集: {h5_path} | 配置: {config_path}\n"
        f"训练 {n_train} / 验证 {n_val} | "
        f"ROI offset={args.offset} | "
        f"Δr={model.range_resolution:.3f} m, "
        f"Δv={model.velocity_resolution:.3f} m/s | "
        f"ROI span ~{model.roi_max_range_m:.1f} m, ±{model.roi_max_velocity_mps:.1f} m/s\n"
        f"检查点目录: {ckpt_dir} | 曲线: {curve_path}"
    )

    best_val_loss = float("inf")
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_rmse_range_m": [],
        "val_rmse_velocity_mps": [],
    }

    # --- 训练循环 ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batch_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d}/{args.epochs:03d}",
            unit="batch",
        )
        for batch in batch_bar:
            x = batch["features"].to(device)
            y_range = batch["range_m"].to(device)
            y_vel = batch["velocity_mps"].to(device)

            optimizer.zero_grad()
            y_bins = model.forward_bins(x)
            target_bins = MonostaticSensingLoss.target_bins_from_physical_labels(
                y_range,
                y_vel,
                range_resolution=model.range_resolution,
                velocity_resolution=model.velocity_resolution,
            )
            loss = criterion(y_bins, target_bins)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        mean_train_loss = train_loss / max(len(train_loader), 1)
        val_loss, val_rmse_r, val_rmse_v = _evaluate(
            model, val_loader, criterion, device
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

        payload = _checkpoint_payload(
            model,
            epoch=epoch,
            in_channels=in_channels,
            full_ds=full_ds,
            h5_path=h5_path,
            config_path=config_path,
            args=args,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, out_path)

        is_periodic = epoch % args.save_every == 0
        is_final = epoch == args.epochs
        if is_periodic:
            torch.save(payload, ckpt_dir / f"checkpoint_{epoch:03d}.pth")
        if is_periodic or is_final:
            _plot_training_history(history, curve_path)

    torch.save(
        _checkpoint_payload(
            model,
            epoch=args.epochs,
            in_channels=in_channels,
            full_ds=full_ds,
            h5_path=h5_path,
            config_path=config_path,
            args=args,
        ),
        final_path,
    )
    _plot_training_history(history, curve_path)

    print(f"最优 val_loss={best_val_loss:.4f}，模型已保存至 {out_path.resolve()}")
    print(f"最终检查点已保存至 {final_path.resolve()}")
    print(f"训练曲线已保存至 {curve_path.resolve()}")


if __name__ == "__main__":
    main()
