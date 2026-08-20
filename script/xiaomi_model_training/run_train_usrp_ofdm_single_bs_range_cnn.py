#!/usr/bin/env python3
"""训练单站 USRP OFDM 测距 CNN：距离谱 → 估计距离 (m)。

须在 ISAC conda 环境中、从仓库根目录运行::

    python script/xiaomi_model_training/run_train_usrp_ofdm_single_bs_range_cnn.py

数据流
------
1. ``SingleBsRangeTorchDataset``：H5 ``profiles`` → ROI + real_imag 特征
2. 全体帧 ``random_split`` 按 ``val_ratio`` 划分
3. ``SingleBsRangeCNN`` 回归标量距离
4. ``TargetRangeRmseLoss`` 优化；按 val RMSE 保存 best_model.pth
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from isac import (
    DEFAULT_XIAOMI_SINGLE_BS_RANGE_CNN_MODEL,
    DEFAULT_XIAOMI_SINGLE_BS_RANGE_H5,
    PROJECT_ROOT,
)
from isac.utils import set_random_seed
from isac.xiaomi_models import (
    DEFAULT_RANGE_ROI,
    FEATURE_MODES,
    SingleBsRangeCNN,
    SingleBsRangeTorchDataset,
    TargetRangeRmseLoss,
    feature_in_channels,
    save_single_bs_range_cnn_checkpoint,
)


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if not (hi > lo):
        raise argparse.ArgumentTypeError(f"range-roi 须满足 min < max，收到 {lo} {hi}")
    return lo, hi


def _collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([s["features"] for s in samples], dim=0),
        "target_range": torch.stack([s["target_range"] for s in samples], dim=0),
        "session_index": torch.stack([s["session_index"] for s in samples], dim=0),
        "frame_index": torch.stack([s["frame_index"] for s in samples], dim=0),
    }


def _build_dataloaders(
    full_ds: SingleBsRangeTorchDataset,
    *,
    val_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    n_total = len(full_ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError(f"训练样本不足：len={n_total}, val_ratio={val_ratio}")

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": _collate_batch,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, n_train, n_val


def _train_step(
    batch: dict[str, torch.Tensor],
    model: SingleBsRangeCNN,
    criterion: TargetRangeRmseLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    features = batch["features"].to(device)
    target = batch["target_range"].to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(features)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def _evaluate(
    model: SingleBsRangeCNN,
    loader: DataLoader,
    criterion: TargetRangeRmseLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target_range"].to(device)
        pred = model(features)
        bs = features.size(0)
        total_loss += float(criterion(pred, target).item()) * bs
        total_mae += float(TargetRangeRmseLoss.mean_abs_error_m(pred, target).item()) * bs
        n += bs
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_mae / n


def _plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_maes: list[float],
    out_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="train RMSE")
    axes[0].plot(epochs, val_losses, label="val RMSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RMSE (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_maes, label="val MAE", color="C2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MAE (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train USRP OFDM single-BS range CNN"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=DEFAULT_XIAOMI_SINGLE_BS_RANGE_H5,
        help="input HDF5 dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_XIAOMI_SINGLE_BS_RANGE_CNN_MODEL.parent,
        help="directory for best_model.pth and training curve",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="real_imag",
        choices=FEATURE_MODES,
    )
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=list(DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
        help="range ROI in meters (default: 0 8)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-cache-features",
        action="store_true",
        help="disable in-memory feature cache",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    set_random_seed(args.seed)
    device = torch.device(args.device)
    range_roi = _parse_range_roi(list(args.range_roi))
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    curve_path = output_dir / "training_curve.png"

    full_ds = SingleBsRangeTorchDataset(
        h5_path,
        range_roi=range_roi,
        feature_mode=args.feature_mode,
        cache_features=not args.no_cache_features,
    )
    train_loader, val_loader, n_train, n_val = _build_dataloaders(
        full_ds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"dataset={h5_path.relative_to(PROJECT_ROOT) if h5_path.is_relative_to(PROJECT_ROOT) else h5_path} "
        f"N={len(full_ds)} train={n_train} val={n_val} "
        f"roi={range_roi} mode={args.feature_mode} device={device}"
    )

    model = SingleBsRangeCNN(
        in_channels=feature_in_channels(args.feature_mode),
        base_channels=args.base_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = TargetRangeRmseLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_maes: list[float] = []
    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            loss = _train_step(batch, model, criterion, optimizer, device)
            bs = batch["features"].size(0)
            running += loss * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{loss:.4f}")
        train_rmse = running / max(n_seen, 1)
        val_rmse, val_mae = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_rmse)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        val_maes.append(val_mae)
        print(
            f"epoch {epoch:03d}: train_rmse={train_rmse:.4f} m  "
            f"val_rmse={val_rmse:.4f} m  val_mae={val_mae:.4f} m  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            save_single_bs_range_cnn_checkpoint(
                best_path,
                model,
                feature_mode=args.feature_mode,
                range_roi=range_roi,
                range_bin_step=full_ds.range_bin_step,
                extra={
                    "epoch": epoch,
                    "val_rmse": best_val,
                    "val_mae": val_mae,
                    "h5_path": str(h5_path),
                    "val_ratio": args.val_ratio,
                    "seed": args.seed,
                },
            )
            print(f"  saved best → {best_path} (val_rmse={best_val:.4f} m)")

    _plot_curves(train_losses, val_losses, val_maes, curve_path)
    print(f"done. best_val_rmse={best_val:.4f} m  curve={curve_path}")


if __name__ == "__main__":
    main()
