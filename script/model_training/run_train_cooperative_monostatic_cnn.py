"""训练 Cooperative Monostatic CNN：双站 ROI 距离谱 → 目标 (x, y)。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/model_training/run_train_cooperative_monostatic_cnn.py

默认已对齐部署配置（ROI 0–4 m、batch 128、增强与 outlier 过滤等）。
覆盖示例::

    python script/model_training/run_train_cooperative_monostatic_cnn.py \\
        --epochs 2 --batch-size 32 --max-samples 512 --no-filter-outliers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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

from isac import PROJECT_ROOT
from isac.models import (
    COOPERATIVE_FEATURE_MODES,
    COOPERATIVE_POOL_MODES,
    CooperativeMonostatic2DCNN,
    CooperativeMonostaticCNN,
    TargetPositionRmseLoss,
    compute_logmag_norm_stats_from_h5,
    cooperative_feature_in_channels,
    cooperative_model_type,
    load_cooperative_monostatic_cnn_checkpoint,
    save_cooperative_norm_stats,
)
from isac.models.loss import session_aggregated_target_rmse_loss
from isac.utils import set_random_seed
from isac_imp.cooperative_monostatic_pipeline import (
    grc_cooperative_processing_params,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    cooperative_frame_cpi_energy,
    filter_cooperative_frames_energy_mad,
    filter_cooperative_frames_hard,
    is_cooperative_monostatic_features_h5,
    load_cooperative_frame_energy,
    open_cooperative_monostatic_training_dataset,
    resolve_cooperative_features_h5,
    session_train_val_split_by_region,
)
from isac_imp.record_target_metadata import (
    REGION_COUNT,
    is_inner_target_xy_m,
    target_region_name,
)

DEFAULT_H5 = Path(
    "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
)
DEFAULT_TEST_H5 = Path(
    "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
DEFAULT_OUTPUT_DIR = Path("models/cnn_deploy_strict_roi4")
# 训练脚本默认 ROI（部署配置）；pipeline 全局 DEFAULT_RANGE_ROI 仍为 (0, 3.5)
TRAIN_DEFAULT_RANGE_ROI = (0.0, 4.0)
TRAIN_DEFAULT_LABEL_JITTER_M = 0.05
TRAIN_DEFAULT_OUTER_RING_WEIGHT = 2.0
TRAIN_DEFAULT_FEATURE_NOISE_STD = 0.02
TRAIN_DEFAULT_SPEC_AUGMENT_PROB = 0.3
EVAL_SCRIPT = PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_cnn_rmse.py"


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


def _resolve_dataset_h5_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    """解析训练 / 训练后评估 / legacy test HDF5 路径。

    ``--swap-train-eval-h5`` 时交换 train 与 eval；若 ``test`` 原与 ``eval`` 相同则随 eval 一起换。
    """
    train_h5 = args.h5_path.resolve()
    eval_h5 = args.eval_h5_path.resolve()
    test_h5 = args.test_h5_path.resolve()
    if not args.swap_train_eval_h5:
        return train_h5, eval_h5, test_h5

    orig_train, orig_eval, orig_test = train_h5, eval_h5, test_h5
    train_h5, eval_h5 = orig_eval, orig_train
    if orig_test == orig_eval:
        test_h5 = eval_h5
    return train_h5, eval_h5, test_h5


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
        "--swap-train-eval-h5",
        action="store_true",
        help="交换训练集与训练后评估集（Run1↔Run2）；等价于互换 --h5-path 与 --eval-h5-path",
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
    parser.add_argument("--batch-size", type=int, default=128)
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
        default=0.4,
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
        default=list(TRAIN_DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
        help="range ROI in meters (default: 0 4)",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--pool-mode",
        type=str,
        choices=list(COOPERATIVE_POOL_MODES),
        default="gap",
        help=(
            "1D CNN range pooling (S1): gap | attention | multiscale | "
            "gap_gmp | soft_argmax (default: gap)"
        ),
    )
    parser.add_argument(
        "--multiscale-bins",
        type=int,
        default=8,
        help="AdaptiveAvgPool1d bins when --pool-mode multiscale (default: 8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4; set 0 to disable)",
    )
    parser.add_argument(
        "--features-h5",
        type=Path,
        default=None,
        help="预计算 features sidecar（默认按 --h5-path / ROI / feature-mode 自动查找）",
    )
    parser.add_argument(
        "--eval-features-h5",
        type=Path,
        default=None,
        help="post-train eval 用 features sidecar（默认按 --eval-h5-path 自动查找）",
    )
    parser.add_argument(
        "--require-features-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="找不到 features sidecar 时直接报错（默认开启；--no-require-features-h5 关闭）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="调试用：限制参与划分的总帧数",
    )
    parser.add_argument(
        "--filter-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练前剔除异常帧（默认开启；--no-filter-outliers 关闭）",
    )
    parser.add_argument(
        "--xy-max-m",
        type=float,
        default=1.0,
        help="硬过滤：|x|,|y| 超过该值视为越界 (m)，默认 1.0",
    )
    parser.add_argument(
        "--outlier-energy-eps",
        type=float,
        default=1e-8,
        help="硬过滤：任一站 CPI 平均幅度 <= eps 视为近零",
    )
    parser.add_argument(
        "--outlier-energy-mad-z",
        type=float,
        default=5.0,
        help="软剔除：session 内能量 MAD z-score 阈值，默认 5.0",
    )
    parser.add_argument(
        "--label-jitter-m",
        type=float,
        default=TRAIN_DEFAULT_LABEL_JITTER_M,
        help="训练集 target_xy 各轴均匀抖动半幅 (m)，验证集始终为 0（default: 0.05）",
    )
    parser.add_argument(
        "--no-label-jitter",
        action="store_true",
        help="禁用训练标签抖动（等价于 --label-jitter-m 0）",
    )
    parser.add_argument(
        "--outer-ring-weight",
        type=float,
        default=TRAIN_DEFAULT_OUTER_RING_WEIGHT,
        help="loss weight for outer-ring targets (|x|>0.6 or |y|>0.6); default 2.0",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=TRAIN_DEFAULT_FEATURE_NOISE_STD,
        help="Gaussian noise std on float training features (default: 0.02)",
    )
    parser.add_argument(
        "--spec-augment-prob",
        type=float,
        default=TRAIN_DEFAULT_SPEC_AUGMENT_PROB,
        help="probability of range-bin SpecAugment on training features (default: 0.3)",
    )
    parser.add_argument(
        "--spec-augment-max-bins",
        type=int,
        default=3,
        help="max masked range bins for SpecAugment",
    )
    parser.add_argument(
        "--session-aggregated-loss",
        action="store_true",
        help="train with session-level aggregated RMSE loss instead of frame-level",
    )
    parser.add_argument(
        "--eval-h5-path",
        type=Path,
        default=DEFAULT_TEST_H5,
        help="训练结束后自动评估用的 HDF5（默认 Run2）",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=None,
        help="自动评估输出目录（默认 out/cooperative_monostatic/<output-dir 名>）",
    )
    parser.add_argument(
        "--no-eval-after-train",
        action="store_true",
        help="跳过训练结束后的 Run2 RMSE 自动评估",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
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
    model: CooperativeMonostaticCNN | CooperativeMonostatic2DCNN,
    *,
    in_channels: int,
    feature_mode: str,
    model_type: str,
    norm_stats_path: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "base_channels": model.base_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "feature_mode": feature_mode,
        "model_type": model_type,
    }
    if isinstance(model, CooperativeMonostaticCNN):
        payload["pool_mode"] = model.pool_mode
        payload["multiscale_bins"] = model.multiscale_bins
        payload["soft_argmax_temp"] = model.soft_argmax_temp
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
    weights = torch.ones(
        target_xy.shape[0], dtype=target_xy.dtype, device=target_xy.device
    )
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


def _compute_batch_loss(
    criterion: TargetPositionRmseLoss,
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    session_index: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None,
    session_aggregated_loss: bool,
) -> torch.Tensor:
    if session_aggregated_loss:
        return session_aggregated_target_rmse_loss(
            pred_xy,
            target_xy,
            session_index,
            sample_weight=sample_weight,
        )
    return criterion(pred_xy, target_xy, sample_weight=sample_weight)


@torch.no_grad()
def _evaluate(
    loader: DataLoader,
    model: CooperativeMonostaticCNN | CooperativeMonostatic2DCNN,
    criterion: TargetPositionRmseLoss,
    device: torch.device | str,
    *,
    outer_ring_weight: float = 1.0,
    session_aggregated_loss: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_euclidean = 0.0
    n_batches = 0
    for batch in loader:
        dual_profiles = batch["dual_profiles"].to(device, non_blocking=True)
        target_xy = batch["target_xy"].to(device, non_blocking=True)
        session_index = batch["session_index"].to(device, non_blocking=True)
        pred_xy = model(dual_profiles)
        sample_weight = _outer_ring_sample_weights(
            target_xy, outer_weight=outer_ring_weight
        )
        loss = _compute_batch_loss(
            criterion,
            pred_xy,
            target_xy,
            session_index,
            sample_weight=sample_weight,
            session_aggregated_loss=session_aggregated_loss,
        )
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
    model: CooperativeMonostaticCNN | CooperativeMonostatic2DCNN,
    criterion: TargetPositionRmseLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    outer_ring_weight: float = 1.0,
    session_aggregated_loss: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train", unit="batch", leave=False):
        dual_profiles = batch["dual_profiles"].to(device, non_blocking=True)
        target_xy = batch["target_xy"].to(device, non_blocking=True)
        session_index = batch["session_index"].to(device, non_blocking=True)
        sample_weight = _outer_ring_sample_weights(
            target_xy, outer_weight=outer_ring_weight
        )
        optimizer.zero_grad(set_to_none=True)
        pred_xy = model(dual_profiles)
        loss = _compute_batch_loss(
            criterion,
            pred_xy,
            target_xy,
            session_index,
            sample_weight=sample_weight,
            session_aggregated_loss=session_aggregated_loss,
        )
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
    seed: int,
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
        seed=seed,
    )


def _reseed_dataset_rngs(dataset: Any, seed: int) -> None:
    """递归对 Dataset / ConcatDataset 调用 ``reseed``。"""
    if isinstance(dataset, ConcatDataset):
        for i, sub in enumerate(dataset.datasets):
            _reseed_dataset_rngs(sub, int(seed) + i * 1009)
        return
    reseed = getattr(dataset, "reseed", None)
    if callable(reseed):
        reseed(int(seed))


def _dataloader_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker：按 info.seed 播种 numpy/torch 与 Dataset RNG。"""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed) % (2**32)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    _reseed_dataset_rngs(info.dataset, worker_seed)


def _load_profiles_for_frames(
    h5_path: Path,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """读取前 ``n_frames`` 帧的双站 divide CPI。"""
    with h5py.File(h5_path, "r") as f:
        profiles_dev0 = np.asarray(f[DATASET_KEY_PROFILES_DEV0][:n_frames])
        profiles_dev1 = np.asarray(f[DATASET_KEY_PROFILES_DEV1][:n_frames])
    return profiles_dev0, profiles_dev1


def _apply_outlier_filters(
    *,
    h5_path: Path,
    session_indices: np.ndarray,
    target_position: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray | None,
    xy_max_m: float,
    energy_eps: float,
    energy_mad_z: float,
    soft_filter: bool = True,
    require_nonempty_train: bool = True,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray | None]:
    """硬过滤全体候选帧，可选对 train/val 做能量 MAD 软剔除。

    ``session_indices`` / ``target_position`` 与 H5 前 N 帧对齐；
    ``train_idx`` / ``eval_idx`` 为该范围内的全局帧索引。
    MAD 统计使用全库能量，再分别过滤 train/val 子集。
    若 H5 含 ``frame_energy``（features sidecar），则跳过 raw CPI 全库读取。
    """
    n_frames = int(session_indices.shape[0])
    energy = load_cooperative_frame_energy(h5_path)
    profiles_dev0: np.ndarray | None = None
    profiles_dev1: np.ndarray | None = None
    if energy is not None:
        energy = np.asarray(energy[:n_frames], dtype=np.float64)
        if energy.shape[0] != n_frames:
            raise ValueError(
                f"frame_energy 长度 {energy.shape[0]} 与 n_frames={n_frames} 不一致"
            )
        keep_hard, drop_counts = filter_cooperative_frames_hard(
            target_position[:, :2],
            xy_max_m=xy_max_m,
            energy_eps=energy_eps,
        )
    else:
        if is_cooperative_monostatic_features_h5(h5_path):
            raise RuntimeError(
                f"features sidecar 缺少 frame_energy，无法做 outlier 过滤: {h5_path}"
            )
        profiles_dev0, profiles_dev1 = _load_profiles_for_frames(h5_path, n_frames)
        energy = cooperative_frame_cpi_energy(profiles_dev0, profiles_dev1)
        keep_hard, drop_counts = filter_cooperative_frames_hard(
            target_position[:, :2],
            profiles_dev0=profiles_dev0,
            profiles_dev1=profiles_dev1,
            xy_max_m=xy_max_m,
            energy_eps=energy_eps,
        )
    hard_set = set(int(i) for i in keep_hard)
    if keep_hard.size == 0:
        raise RuntimeError(
            f"outlier 硬过滤后无剩余帧"
            + (f" ({label})" if label else "")
            + f": {drop_counts}"
        )

    train_idx = np.asarray([i for i in train_idx if int(i) in hard_set], dtype=np.int64)
    eval_kept: np.ndarray | None
    if eval_idx is None:
        eval_kept = None
    else:
        eval_kept = np.asarray(
            [i for i in eval_idx if int(i) in hard_set], dtype=np.int64
        )

    soft_dropped_train = 0
    soft_dropped_val = 0
    if soft_filter:
        if train_idx.size > 0:
            train_idx, soft_dropped_train = filter_cooperative_frames_energy_mad(
                train_idx,
                session_indices,
                energy,
                z_thresh=energy_mad_z,
            )
        if eval_kept is not None and eval_kept.size > 0:
            eval_kept, soft_dropped_val = filter_cooperative_frames_energy_mad(
                eval_kept,
                session_indices,
                energy,
                z_thresh=energy_mad_z,
            )

    hard_dropped = n_frames - int(keep_hard.size)
    prefix = f"Outlier filter{f' [{label}]' if label else ''}:"
    eval_n = "n/a" if eval_kept is None else str(int(eval_kept.size))
    train_n = int(train_idx.size)
    print(
        f"{prefix} hard dropped {hard_dropped} "
        f"(nan_label={drop_counts['nan_label']}, oob_xy={drop_counts['oob_xy']}, "
        f"nan_cpi={drop_counts['nan_cpi']}, near_zero={drop_counts['near_zero']}); "
        f"soft dropped {soft_dropped_train} from train, {soft_dropped_val} from val; "
        f"train {train_n} / eval {eval_n}",
        flush=True,
    )
    if require_nonempty_train and train_idx.size == 0:
        raise RuntimeError(
            "outlier 过滤后训练集为空" + (f" ({label})" if label else "")
        )
    if eval_kept is not None and eval_kept.size == 0:
        raise RuntimeError(
            "outlier 过滤后验证/测试集为空" + (f" ({label})" if label else "")
        )
    return train_idx, eval_kept


def _run_post_train_eval(
    *,
    checkpoint: Path,
    eval_h5: Path,
    range_roi: tuple[float, float],
    output_dir: Path,
    device: str | None = None,
    features_h5: Path | None = None,
) -> None:
    """训练结束后调用 cnn_rmse 评估脚本（Run2 等外部集）。"""
    if not checkpoint.is_file():
        print(
            f"Warning: skip post-train eval, checkpoint missing: {checkpoint}",
            flush=True,
        )
        return
    if not eval_h5.is_file():
        print(
            f"Warning: skip post-train eval, eval HDF5 missing: {eval_h5}", flush=True
        )
        return
    if not EVAL_SCRIPT.is_file():
        print(
            f"Warning: skip post-train eval, script missing: {EVAL_SCRIPT}", flush=True
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cnn_rmse.csv"
    heatmap_path = output_dir / "cnn_rmse_heatmap.png"
    cdf_path = output_dir / "cnn_rmse_cdf.png"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--h5-path",
        str(eval_h5),
        "--checkpoint",
        str(checkpoint),
        "--range-roi",
        str(range_roi[0]),
        str(range_roi[1]),
        "--output-csv",
        str(csv_path),
        "--output-heatmap",
        str(heatmap_path),
        "--output-cdf",
        str(cdf_path),
    ]
    if features_h5 is not None:
        cmd.extend(["--features-h5", str(features_h5)])
    if device:
        cmd.extend(["--device", str(device)])

    print("\n>>> post-train eval:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    h5_path, post_eval_h5, resolved_test_h5 = _resolve_dataset_h5_paths(args)
    if args.swap_train_eval_h5:
        print(
            f"Dataset swap: train={h5_path} | eval={post_eval_h5}",
            flush=True,
        )
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 不存在: {h5_path}")
    if args.outer_ring_weight <= 0.0:
        raise ValueError(f"--outer-ring-weight 须 > 0，收到 {args.outer_ring_weight}")
    if args.filter_outliers:
        if args.xy_max_m <= 0.0:
            raise ValueError(f"--xy-max-m 须 > 0，收到 {args.xy_max_m}")
        if args.outlier_energy_eps < 0.0:
            raise ValueError(
                f"--outlier-energy-eps 须 >= 0，收到 {args.outlier_energy_eps}"
            )
        if args.outlier_energy_mad_z <= 0.0:
            raise ValueError(
                f"--outlier-energy-mad-z 须 > 0，收到 {args.outlier_energy_mad_z}"
            )

    use_external_test = args.use_test_h5 and not args.no_test_h5
    if args.finetune and use_external_test:
        raise ValueError("--finetune 与 --use-test-h5 不能同时使用")
    if args.extra_train_h5_path is not None and use_external_test:
        raise ValueError("--extra-train-h5-path 与 --use-test-h5 不能同时使用")

    test_h5_path = resolved_test_h5 if use_external_test else None
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

    raw_h5_path = h5_path
    h5_path = resolve_cooperative_features_h5(
        raw_h5_path,
        range_roi=range_roi,
        feature_mode=feature_mode,
        features_h5=args.features_h5,
        require=bool(args.require_features_h5),
    )
    if h5_path != raw_h5_path.resolve():
        print(f"Using features sidecar for train: {h5_path}", flush=True)

    post_eval_features: Path | None = None
    try:
        resolved_post_eval = resolve_cooperative_features_h5(
            post_eval_h5,
            range_roi=range_roi,
            feature_mode=feature_mode,
            features_h5=args.eval_features_h5,
            require=False,
        )
        if is_cooperative_monostatic_features_h5(resolved_post_eval):
            post_eval_features = resolved_post_eval
    except (FileNotFoundError, ValueError):
        post_eval_features = None

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
        assert test_h5_path is not None
        raw_test_h5 = test_h5_path
        test_h5_path = resolve_cooperative_features_h5(
            raw_test_h5,
            range_roi=range_roi,
            feature_mode=feature_mode,
            features_h5=None,
            require=bool(args.require_features_h5),
        )
        if test_h5_path != raw_test_h5.resolve():
            print(f"Using features sidecar for test: {test_h5_path}", flush=True)
        with h5py.File(test_h5_path, "r") as f:
            n_test_frames = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
            test_sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
            test_targets = np.asarray(
                f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
            )
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
        test_sessions = None
        test_targets = None

    if args.filter_outliers:
        if use_external_test:
            train_idx, _ = _apply_outlier_filters(
                h5_path=h5_path,
                session_indices=session_indices,
                target_position=target_position,
                train_idx=train_idx,
                eval_idx=None,
                xy_max_m=args.xy_max_m,
                energy_eps=args.outlier_energy_eps,
                energy_mad_z=args.outlier_energy_mad_z,
                soft_filter=True,
                require_nonempty_train=True,
                label="train-h5",
            )
            assert test_h5_path is not None
            assert test_sessions is not None and test_targets is not None
            _, eval_idx = _apply_outlier_filters(
                h5_path=test_h5_path,
                session_indices=test_sessions,
                target_position=test_targets,
                train_idx=np.array([], dtype=np.int64),
                eval_idx=eval_idx,
                xy_max_m=args.xy_max_m,
                energy_eps=args.outlier_energy_eps,
                energy_mad_z=args.outlier_energy_mad_z,
                soft_filter=True,
                require_nonempty_train=False,
                label="test-h5",
            )
        else:
            train_idx, eval_idx = _apply_outlier_filters(
                h5_path=h5_path,
                session_indices=session_indices,
                target_position=target_position,
                train_idx=train_idx,
                eval_idx=eval_idx,
                xy_max_m=args.xy_max_m,
                energy_eps=args.outlier_energy_eps,
                energy_mad_z=args.outlier_energy_mad_z,
                soft_filter=True,
                require_nonempty_train=True,
            )

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
            seed=args.seed,
        )
    ]
    if args.extra_train_h5_path is not None:
        extra_path = args.extra_train_h5_path.resolve()
        if not extra_path.is_file():
            raise FileNotFoundError(f"extra train HDF5 不存在: {extra_path}")
        extra_path = resolve_cooperative_features_h5(
            extra_path,
            range_roi=range_roi,
            feature_mode=feature_mode,
            features_h5=None,
            require=bool(args.require_features_h5),
        )
        with h5py.File(extra_path, "r") as f:
            extra_n = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
            extra_sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
            extra_targets = np.asarray(
                f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
            )
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
        if args.filter_outliers:
            extra_train_idx, _ = _apply_outlier_filters(
                h5_path=extra_path,
                session_indices=extra_sessions,
                target_position=extra_targets,
                train_idx=extra_train_idx,
                eval_idx=None,
                xy_max_m=args.xy_max_m,
                energy_eps=args.outlier_energy_eps,
                energy_mad_z=args.outlier_energy_mad_z,
                soft_filter=True,
                require_nonempty_train=True,
                label="extra-train-h5",
            )
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
                seed=args.seed + 17,
            )
        )

    train_dataset: ConcatDataset | Any
    train_dataset = (
        train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
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
        seed=args.seed,
    )

    pin_memory = device.type == "cuda"
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed))
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "collate_fn": _collate_batch,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = _dataloader_worker_init_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_gen,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    in_channels = cooperative_feature_in_channels(feature_mode)  # type: ignore[arg-type]
    model_type = cooperative_model_type(feature_mode)  # type: ignore[arg-type]
    lr = float(
        args.finetune_lr if args.resume is not None and args.finetune_lr else args.lr
    )

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
    elif model_type == "2d":
        model = CooperativeMonostatic2DCNN(
            in_channels=in_channels,
            base_channels=args.base_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = CooperativeMonostaticCNN(
            in_channels=in_channels,
            base_channels=args.base_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pool_mode=args.pool_mode,
            multiscale_bins=args.multiscale_bins,
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
        train_label = (
            "部署验证训练 (swapped)" if args.swap_train_eval_h5 else "部署验证训练"
        )
        dataset_msg = (
            f"{train_label}: {h5_path}\n"
            f"训练 {len(train_dataset)} / 验证 {len(eval_dataset)} 帧 | "
            f"checkpoint 依据 val_loss"
        )

    print(
        f"{dataset_msg}\n"
        f"feature_mode={feature_mode}, model_type={model_type}, in_channels={in_channels}\n"
        f"ROI {range_roi[0]:.1f}–{range_roi[1]:.1f} m | "
        f"label_jitter_m={label_jitter_m:.3f} (train only)\n"
        f"weight_decay={args.weight_decay}, outer_ring_weight={args.outer_ring_weight}, "
        f"session_aggregated_loss={args.session_aggregated_loss}, "
        f"feature_noise_std={args.feature_noise_std}, spec_augment_prob={args.spec_augment_prob}\n"
        f"模型 base_channels={args.base_channels}, num_layers={args.num_layers}, "
        f"dropout={args.dropout}, pool_mode={args.pool_mode}, "
        f"multiscale_bins={args.multiscale_bins}, lr={lr}\n"
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
            session_aggregated_loss=args.session_aggregated_loss,
        )
        eval_loss, eval_mean_euclidean = _evaluate(
            eval_loader,
            model,
            criterion,
            device,
            outer_ring_weight=args.outer_ring_weight,
            session_aggregated_loss=args.session_aggregated_loss,
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
                    model_type=model_type,
                    norm_stats_path=norm_stats_rel,
                ),
                paths.best_model,
            )
        else:
            epochs_without_improve += 1

        if (
            args.early_stop_patience > 0
            and epochs_without_improve >= args.early_stop_patience
        ):
            print(
                f"Early stop: no {eval_label.lower()} improvement for "
                f"{args.early_stop_patience} epochs."
            )
            break

    _plot_training_history(history, paths.training_curve, eval_label=eval_label)

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

    if not args.no_eval_after_train:
        eval_out = (
            args.eval_output_dir.resolve()
            if args.eval_output_dir is not None
            else (
                PROJECT_ROOT
                / "out"
                / "cooperative_monostatic"
                / paths.checkpoint_dir.name
            )
        )
        _run_post_train_eval(
            checkpoint=paths.best_model,
            eval_h5=post_eval_h5,
            range_roi=range_roi,
            output_dir=eval_out,
            device=args.device,
            features_h5=post_eval_features,
        )


if __name__ == "__main__":
    main()
