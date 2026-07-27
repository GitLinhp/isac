"""训练 Cooperative Monostatic CNN：双站 ROI 距离谱 → 目标 (x, y)。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/model_training/run_train_cooperative_monostatic_cnn.py

Smoke test::

    python script/model_training/run_train_cooperative_monostatic_cnn.py \\
        --epochs 2 --batch-size 32 --max-samples 512
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from isac.models import (
    CooperativeMonostaticCNN,
    TargetPositionRmseLoss,
)
from isac.utils import set_random_seed
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    DEFAULT_LABEL_JITTER_M,
    CooperativeMonostaticRangeProfileDataset,
    session_train_val_split_by_region,
)
from isac_imp.record_target_metadata import REGION_COUNT, target_region_name

DEFAULT_H5 = Path("data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5")
DEFAULT_OUTPUT_DIR = Path("models/cooperative_monostatic_cnn")


@dataclass(frozen=True)
class TrainPaths:
    checkpoint_dir: Path
    best_model: Path
    training_curve: Path


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"range-roi 须满足 min < max，收到 {lo} {hi}")
    return lo, hi


def _print_region_split_summary(
    split_info: dict[int, dict[str, int]],
    *,
    val_ratio: float,
    seed: int,
) -> None:
    rows = [
        [
            target_region_name(region_id),
            info.get("train", 0),
            info.get("val", 0),
        ]
        for region_id in range(REGION_COUNT)
        for info in [split_info.get(region_id, {"train": 0, "val": 0})]
    ]
    print(
        f"\nTrain/val split: region-stratified (9 regions), "
        f"val_ratio={val_ratio:.2f}, seed={seed}"
    )
    print(
        tabulate(
            rows,
            headers=["Region", "Train sessions", "Val sessions"],
            tablefmt="simple",
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 Cooperative Monostatic CNN（双站 ROI 距离谱 → xy）"
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=list(DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="调试用：限制参与划分的总帧数",
    )
    parser.add_argument(
        "--label-jitter-m",
        type=float,
        default=DEFAULT_LABEL_JITTER_M,
        help="训练集 target_xy 各轴均匀抖动半幅 (m)，验证集始终为 0",
    )
    parser.add_argument(
        "--no-label-jitter",
        action="store_true",
        help="禁用训练标签抖动（等价于 --label-jitter-m 0）",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=10,
        help="每隔多少 epoch 刷新 training_curve.png（默认 10）",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


def _resolve_paths(output_dir: Path) -> TrainPaths:
    checkpoint_dir = output_dir
    return TrainPaths(
        checkpoint_dir=checkpoint_dir,
        best_model=checkpoint_dir / "best_model.pth",
        training_curve=checkpoint_dir / "training_curve.png",
    )


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "dual_profiles": torch.stack([item["dual_profiles"] for item in batch]),
        "target_xy": torch.stack([item["target_xy"] for item in batch]),
        "session_index": torch.stack([item["session_index"] for item in batch]),
    }


def _checkpoint_payload(
    model: CooperativeMonostaticCNN,
    *,
    in_channels: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }


def _plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    if not history["epoch"]:
        return

    fig, (ax_loss, ax_err) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    epochs = history["epoch"]

    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_ylabel("Loss (RMSE m)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_err.plot(epochs, history["val_mean_euclidean_m"], label="Val mean Euclidean (m)")
    ax_err.set_xlabel("Epoch")
    ax_err.set_ylabel("Error (m)")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _evaluate(
    loader: DataLoader,
    model: CooperativeMonostaticCNN,
    criterion: TargetPositionRmseLoss,
    device: torch.device | str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_euclidean = 0.0
    n_batches = 0
    for batch in loader:
        dual_profiles = batch["dual_profiles"].to(device)
        target_xy = batch["target_xy"].to(device)
        pred_xy = model(dual_profiles)
        loss = criterion(pred_xy, target_xy)
        total_loss += float(loss.item())
        total_euclidean += float(
            TargetPositionRmseLoss.mean_euclidean_error_m(pred_xy, target_xy).item()
        )
        n_batches += 1
    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_euclidean / n_batches


def _train_one_epoch(
    loader: DataLoader,
    model: CooperativeMonostaticCNN,
    criterion: TargetPositionRmseLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train", unit="batch", leave=False):
        dual_profiles = batch["dual_profiles"].to(device)
        target_xy = batch["target_xy"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_xy = model(dual_profiles)
        loss = criterion(pred_xy, target_xy)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 不存在: {h5_path}")
    if args.plot_every < 1:
        raise ValueError(f"--plot-every 须 >= 1，收到 {args.plot_every}")

    range_roi = _parse_range_roi(list(args.range_roi))
    proc_params = grc_cooperative_processing_params()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths = _resolve_paths(args.output_dir.resolve())
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        session_indices = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        target_position = np.asarray(
            f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
        )

    if args.max_samples is not None:
        max_samples = min(int(args.max_samples), session_indices.size)
        session_indices = session_indices[:max_samples]
        target_position = target_position[:max_samples]

    train_idx, val_idx, split_info = session_train_val_split_by_region(
        session_indices,
        target_position,
        args.val_ratio,
        seed=args.seed,
    )
    _print_region_split_summary(
        split_info,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    label_jitter_m = 0.0 if args.no_label_jitter else float(args.label_jitter_m)
    if label_jitter_m < 0.0:
        raise ValueError(f"--label-jitter-m 须 >= 0，收到 {label_jitter_m}")

    train_dataset = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        train_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        transform_on_load=True,
        label_jitter_m=label_jitter_m,
    )
    val_dataset = CooperativeMonostaticRangeProfileDataset(
        h5_path,
        val_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        transform_on_load=True,
        label_jitter_m=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )

    in_channels = 4
    model = CooperativeMonostaticCNN(
        in_channels=in_channels,
        base_channels=args.base_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = TargetPositionRmseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"数据集: {h5_path}\n"
        f"训练 {len(train_dataset)} / 验证 {len(val_dataset)} 帧 | "
        f"ROI {range_roi[0]:.1f}–{range_roi[1]:.1f} m | "
        f"label_jitter_m={label_jitter_m:.3f} (train only)\n"
        f"模型 base_channels={args.base_channels}, num_layers={args.num_layers}, "
        f"dropout={args.dropout}, plot_every={args.plot_every}\n"
        f"检查点: {paths.best_model} | 曲线: {paths.training_curve} | device={device}"
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mean_euclidean_m": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )
        val_loss, val_mean_euclidean = _evaluate(
            val_loader, model, criterion, device
        )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mean_euclidean_m"].append(val_mean_euclidean)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mean_euclidean={val_mean_euclidean:.4f} m"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                _checkpoint_payload(model, in_channels=in_channels),
                paths.best_model,
            )

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            _plot_training_history(history, paths.training_curve)

    summary_rows = [
        ["Train frames", len(train_dataset)],
        ["Val frames", len(val_dataset)],
        ["Train label jitter (m)", f"{label_jitter_m:.3f}"],
        ["Best val RMSE (m)", f"{best_val_loss:.4f}"],
        [
            "Final val mean Euclidean (m)",
            f"{history['val_mean_euclidean_m'][-1]:.4f}",
        ],
        ["Checkpoint", str(paths.best_model)],
        ["Training curve", str(paths.training_curve)],
    ]
    print("\n" + tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="simple"))


if __name__ == "__main__":
    main()
