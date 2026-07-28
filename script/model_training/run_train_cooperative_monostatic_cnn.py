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
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from isac.models import (
    COOPERATIVE_FEATURE_MODES,
    CooperativeMonostaticCNN,
    TargetPositionRmseLoss,
    compute_logmag_norm_stats_from_h5,
    cooperative_feature_in_channels,
    load_cooperative_monostatic_cnn_checkpoint,
    save_cooperative_norm_stats,
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
    open_cooperative_monostatic_training_dataset,
    session_train_val_split_by_region,
)
from isac_imp.record_target_metadata import REGION_COUNT, is_inner_target_xy_m, target_region_name

DEFAULT_H5 = Path(
    "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
)
DEFAULT_TEST_H5 = Path(
    "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
DEFAULT_OUTPUT_DIR = Path("models/cooperative_monostatic_cnn")


@dataclass(frozen=True)
class TrainPaths:
    checkpoint_dir: Path
    best_model: Path
    training_curve: Path
    norm_stats: Path


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
    parser.add_argument(
        "--extra-train-h5-path",
        type=Path,
        default=None,
        help="optional second HDF5 concatenated into training set (e.g. Run2 finetune train)",
    )
    parser.add_argument(
        "--test-h5-path",
        type=Path,
        default=DEFAULT_TEST_H5,
        help="external test HDF5 (opt-in via --use-test-h5; not for deployment validation)",
    )
    parser.add_argument(
        "--use-test-h5",
        action="store_true",
        help="use --test-h5-path for checkpoint selection (legacy cross-domain mode)",
    )
    parser.add_argument(
        "--no-test-h5",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="train/val split on --h5-path (typically Run2) instead of Run1-only split",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume / finetune from existing checkpoint",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="override --lr when --resume is set",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (0 to disable)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="stop if val loss does not improve for N epochs (0 to disable)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="val split ratio on --h5-path when not using external test",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=list(COOPERATIVE_FEATURE_MODES),
        default="real_imag",
        help="model input feature mode (default: real_imag for cross-domain)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="fixed normalization npz for logmag_fixed_norm (default: auto-compute on train)",
    )
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
        "--outer-ring-weight",
        type=float,
        default=1.0,
        help="loss weight for outer-ring targets (|x|>0.6 or |y|>0.6); default 1.0",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std on float training features",
    )
    parser.add_argument(
        "--spec-augment-prob",
        type=float,
        default=0.0,
        help="probability of range-bin SpecAugment on training features",
    )
    parser.add_argument(
        "--spec-augment-max-bins",
        type=int,
        default=3,
        help="max masked range bins for SpecAugment",
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
        norm_stats=checkpoint_dir / "norm_stats.npz",
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
    feature_mode: str,
    norm_stats_path: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "feature_mode": feature_mode,
    }
    if norm_stats_path is not None:
        payload["norm_stats_path"] = norm_stats_path
    return payload


def _outer_ring_sample_weights(
    target_xy: torch.Tensor,
    *,
    outer_weight: float,
) -> torch.Tensor | None:
    if outer_weight == 1.0:
        return None
    weights = torch.ones(target_xy.shape[0], dtype=target_xy.dtype, device=target_xy.device)
    for i in range(target_xy.shape[0]):
        x = float(target_xy[i, 0])
        y = float(target_xy[i, 1])
        if not is_inner_target_xy_m(x, y):
            weights[i] = outer_weight
    return weights


def _plot_training_history(
    history: dict[str, list[float]],
    path: Path,
    *,
    eval_label: str = "Val",
) -> None:
    if not history["epoch"]:
        return

    eval_loss_key = "test_loss" if "test_loss" in history else "val_loss"
    eval_err_key = (
        "test_mean_euclidean_m"
        if "test_mean_euclidean_m" in history
        else "val_mean_euclidean_m"
    )

    fig, (ax_loss, ax_err) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    epochs = history["epoch"]

    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history[eval_loss_key], label=f"{eval_label} loss")
    ax_loss.set_ylabel("Loss (RMSE m)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_err.plot(
        epochs,
        history[eval_err_key],
        label=f"{eval_label} mean Euclidean (m)",
    )
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
    *,
    outer_ring_weight: float = 1.0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_euclidean = 0.0
    n_batches = 0
    for batch in loader:
        dual_profiles = batch["dual_profiles"].to(device)
        target_xy = batch["target_xy"].to(device)
        pred_xy = model(dual_profiles)
        sample_weight = _outer_ring_sample_weights(
            target_xy, outer_weight=outer_ring_weight
        )
        loss = criterion(pred_xy, target_xy, sample_weight=sample_weight)
        total_loss += float(loss.item())
        total_euclidean += float(
            TargetPositionRmseLoss.mean_euclidean_error_m(
                pred_xy, target_xy, sample_weight=sample_weight
            ).item()
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
    *,
    outer_ring_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train", unit="batch", leave=False):
        dual_profiles = batch["dual_profiles"].to(device)
        target_xy = batch["target_xy"].to(device)
        sample_weight = _outer_ring_sample_weights(
            target_xy, outer_weight=outer_ring_weight
        )
        optimizer.zero_grad(set_to_none=True)
        pred_xy = model(dual_profiles)
        loss = criterion(pred_xy, target_xy, sample_weight=sample_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _resolve_norm_stats(
    *,
    feature_mode: str,
    norm_stats_arg: Path | None,
    paths: TrainPaths,
    h5_path: Path,
    train_idx: np.ndarray,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float],
) -> tuple[np.ndarray | None, np.ndarray | None, Path | None]:
    if feature_mode != "logmag_fixed_norm":
        return None, None, None

    if norm_stats_arg is not None:
        from isac.models.preprocess import load_cooperative_norm_stats

        means, stds, _ = load_cooperative_norm_stats(norm_stats_arg)
        return means, stds, norm_stats_arg.resolve()

    if paths.norm_stats.is_file():
        from isac.models.preprocess import load_cooperative_norm_stats

        means, stds, _ = load_cooperative_norm_stats(paths.norm_stats)
        return means, stds, paths.norm_stats.resolve()

    print("Computing Run1-only log-mag norm stats on training frames...")
    means, stds = compute_logmag_norm_stats_from_h5(
        h5_path,
        train_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        show_progress=True,
    )
    save_cooperative_norm_stats(
        paths.norm_stats,
        means=means,
        stds=stds,
        feature_mode="logmag_fixed_norm",
    )
    print(f"Saved norm stats: {paths.norm_stats}")
    return means, stds, paths.norm_stats.resolve()


def _make_dataset(
    h5_path: Path,
    frame_indices: np.ndarray,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float],
    label_jitter_m: float,
    feature_mode: str,
    norm_means: np.ndarray | None,
    norm_stds: np.ndarray | None,
    feature_noise_std: float,
    spec_augment_prob: float,
    spec_augment_max_bins: int,
    augment: bool,
):
    return open_cooperative_monostatic_training_dataset(
        h5_path,
        frame_indices,
        proc_params=proc_params,
        range_roi=range_roi,
        transform_on_load=True,
        label_jitter_m=label_jitter_m,
        feature_mode=feature_mode,
        norm_means=norm_means,
        norm_stds=norm_stds,
        feature_noise_std=feature_noise_std,
        spec_augment_prob=spec_augment_prob,
        spec_augment_max_bins=spec_augment_max_bins,
        augment=augment,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 不存在: {h5_path}")
    if args.plot_every < 1:
        raise ValueError(f"--plot-every 须 >= 1，收到 {args.plot_every}")
    if args.outer_ring_weight <= 0.0:
        raise ValueError(f"--outer-ring-weight 须 > 0，收到 {args.outer_ring_weight}")

    use_external_test = args.use_test_h5 and not args.no_test_h5
    if args.finetune and use_external_test:
        raise ValueError("--finetune 与 --use-test-h5 不能同时使用")
    if args.extra_train_h5_path is not None and use_external_test:
        raise ValueError("--extra-train-h5-path 与 --use-test-h5 不能同时使用")

    test_h5_path = args.test_h5_path.resolve() if use_external_test else None
    if test_h5_path is not None and not test_h5_path.is_file():
        raise FileNotFoundError(f"测试 HDF5 不存在: {test_h5_path}")

    feature_mode = args.feature_mode
    range_roi = _parse_range_roi(list(args.range_roi))
    proc_params = grc_cooperative_processing_params()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths = _resolve_paths(args.output_dir.resolve())
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_frames = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
        session_indices = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        target_position = np.asarray(
            f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
        )

    if args.max_samples is not None:
        max_samples = min(int(args.max_samples), n_frames)
        n_frames = max_samples
        session_indices = session_indices[:max_samples]
        target_position = target_position[:max_samples]

    if use_external_test:
        train_idx = np.arange(n_frames, dtype=np.int64)
        with h5py.File(test_h5_path, "r") as f:
            n_test_frames = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
        eval_idx = np.arange(n_test_frames, dtype=np.int64)
        eval_h5_path = test_h5_path
        eval_label = "Test"
    else:
        train_idx, eval_idx, split_info = session_train_val_split_by_region(
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
        eval_h5_path = h5_path
        eval_label = "Val"

    label_jitter_m = 0.0 if args.no_label_jitter else float(args.label_jitter_m)
    if label_jitter_m < 0.0:
        raise ValueError(f"--label-jitter-m 须 >= 0，收到 {label_jitter_m}")

    norm_means, norm_stds, norm_stats_path = _resolve_norm_stats(
        feature_mode=feature_mode,
        norm_stats_arg=args.norm_stats,
        paths=paths,
        h5_path=h5_path,
        train_idx=train_idx,
        proc_params=proc_params,
        range_roi=range_roi,
    )

    train_datasets = [
        _make_dataset(
            h5_path,
            train_idx,
            proc_params=proc_params,
            range_roi=range_roi,
            label_jitter_m=label_jitter_m,
            feature_mode=feature_mode,
            norm_means=norm_means,
            norm_stds=norm_stds,
            feature_noise_std=args.feature_noise_std,
            spec_augment_prob=args.spec_augment_prob,
            spec_augment_max_bins=args.spec_augment_max_bins,
            augment=True,
        )
    ]
    if args.extra_train_h5_path is not None:
        extra_path = args.extra_train_h5_path.resolve()
        if not extra_path.is_file():
            raise FileNotFoundError(f"extra train HDF5 不存在: {extra_path}")
        with h5py.File(extra_path, "r") as f:
            extra_n = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
            extra_sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
            extra_targets = np.asarray(f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64)
        if args.finetune:
            extra_train_idx, _, extra_split = session_train_val_split_by_region(
                extra_sessions,
                extra_targets,
                args.val_ratio,
                seed=args.seed,
            )
            _print_region_split_summary(
                extra_split,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
        else:
            extra_train_idx = np.arange(extra_n, dtype=np.int64)
        train_datasets.append(
            _make_dataset(
                extra_path,
                extra_train_idx,
                proc_params=proc_params,
                range_roi=range_roi,
                label_jitter_m=label_jitter_m,
                feature_mode=feature_mode,
                norm_means=norm_means,
                norm_stds=norm_stds,
                feature_noise_std=args.feature_noise_std,
                spec_augment_prob=args.spec_augment_prob,
                spec_augment_max_bins=args.spec_augment_max_bins,
                augment=True,
            )
        )

    train_dataset: ConcatDataset | Any
    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )

    eval_dataset = _make_dataset(
        eval_h5_path,
        eval_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        label_jitter_m=0.0,
        feature_mode=feature_mode,
        norm_means=norm_means,
        norm_stds=norm_stds,
        feature_noise_std=0.0,
        spec_augment_prob=0.0,
        spec_augment_max_bins=args.spec_augment_max_bins,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )

    in_channels = cooperative_feature_in_channels(feature_mode)  # type: ignore[arg-type]
    lr = float(args.finetune_lr if args.resume is not None and args.finetune_lr else args.lr)

    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
        model = load_cooperative_monostatic_cnn_checkpoint(resume_path, device)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckpt_mode = ckpt.get("feature_mode", feature_mode)
        if ckpt_mode != feature_mode:
            print(
                f"Warning: checkpoint feature_mode={ckpt_mode!r} "
                f"!= CLI {feature_mode!r}; using CLI settings for in_channels"
            )
        if int(ckpt.get("in_channels", in_channels)) != in_channels:
            raise ValueError(
                f"resume in_channels={ckpt.get('in_channels')} "
                f"与当前 feature_mode 需要 {in_channels} 不一致"
            )
    else:
        model = CooperativeMonostaticCNN(
            in_channels=in_channels,
            base_channels=args.base_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    criterion = TargetPositionRmseLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(args.weight_decay),
    )
    scheduler = None
    if args.lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.lr_scheduler_patience,
            factor=0.5,
        )

    if use_external_test:
        dataset_msg = (
            f"训练集: {h5_path} ({len(train_dataset)} 帧)\n"
            f"测试集: {test_h5_path} ({len(eval_dataset)} 帧) | "
            f"checkpoint 依据 test_loss [legacy mode]"
        )
    elif args.finetune:
        dataset_msg = (
            f"Finetune 数据集: {h5_path}\n"
            f"训练 {len(train_dataset)} / 验证 {len(eval_dataset)} 帧"
        )
    else:
        dataset_msg = (
            f"部署验证训练 (Run1 only): {h5_path}\n"
            f"训练 {len(train_dataset)} / 验证 {len(eval_dataset)} 帧 | "
            f"checkpoint 依据 val_loss"
        )

    print(
        f"{dataset_msg}\n"
        f"feature_mode={feature_mode}, in_channels={in_channels}\n"
        f"ROI {range_roi[0]:.1f}–{range_roi[1]:.1f} m | "
        f"label_jitter_m={label_jitter_m:.3f} (train only)\n"
        f"weight_decay={args.weight_decay}, outer_ring_weight={args.outer_ring_weight}, "
        f"feature_noise_std={args.feature_noise_std}, spec_augment_prob={args.spec_augment_prob}\n"
        f"模型 base_channels={args.base_channels}, num_layers={args.num_layers}, "
        f"dropout={args.dropout}, lr={lr}, plot_every={args.plot_every}\n"
        f"检查点: {paths.best_model} | 曲线: {paths.training_curve} | device={device}"
    )
    if norm_stats_path is not None:
        print(f"Norm stats: {norm_stats_path}")

    eval_loss_key = "test_loss" if use_external_test else "val_loss"
    eval_err_key = (
        "test_mean_euclidean_m" if use_external_test else "val_mean_euclidean_m"
    )
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        eval_loss_key: [],
        eval_err_key: [],
    }
    best_eval_loss = float("inf")
    epochs_without_improve = 0
    norm_stats_rel: str | None
    if norm_stats_path is not None:
        try:
            norm_stats_rel = str(norm_stats_path.relative_to(paths.checkpoint_dir))
        except ValueError:
            norm_stats_rel = str(norm_stats_path)
    else:
        norm_stats_rel = None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            outer_ring_weight=args.outer_ring_weight,
        )
        eval_loss, eval_mean_euclidean = _evaluate(
            eval_loader,
            model,
            criterion,
            device,
            outer_ring_weight=args.outer_ring_weight,
        )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history[eval_loss_key].append(eval_loss)
        history[eval_err_key].append(eval_mean_euclidean)

        if scheduler is not None:
            scheduler.step(eval_loss)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | {eval_label.lower()}_loss={eval_loss:.4f} | "
            f"{eval_label.lower()}_mean_euclidean={eval_mean_euclidean:.4f} m | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_without_improve = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    in_channels=in_channels,
                    feature_mode=feature_mode,
                    norm_stats_path=norm_stats_rel,
                ),
                paths.best_model,
            )
        else:
            epochs_without_improve += 1

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            _plot_training_history(history, paths.training_curve, eval_label=eval_label)

        if (
            args.early_stop_patience > 0
            and epochs_without_improve >= args.early_stop_patience
        ):
            print(
                f"Early stop: no {eval_label.lower()} improvement for "
                f"{args.early_stop_patience} epochs."
            )
            break

    if use_external_test:
        summary_rows = [
            ["Train frames", len(train_dataset)],
            ["Test frames", len(eval_dataset)],
            ["Feature mode", feature_mode],
            ["Train label jitter (m)", f"{label_jitter_m:.3f}"],
            ["Best test RMSE (m)", f"{best_eval_loss:.4f}"],
            [
                "Final test mean Euclidean (m)",
                f"{history[eval_err_key][-1]:.4f}",
            ],
            ["Checkpoint", str(paths.best_model)],
            ["Training curve", str(paths.training_curve)],
        ]
    else:
        summary_rows = [
            ["Train frames", len(train_dataset)],
            ["Val frames", len(eval_dataset)],
            ["Feature mode", feature_mode],
            ["Train label jitter (m)", f"{label_jitter_m:.3f}"],
            ["Best val RMSE (m)", f"{best_eval_loss:.4f}"],
            [
                "Final val mean Euclidean (m)",
                f"{history[eval_err_key][-1]:.4f}",
            ],
            ["Checkpoint", str(paths.best_model)],
            ["Training curve", str(paths.training_curve)],
        ]
    print("\n" + tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="simple"))


if __name__ == "__main__":
    main()
