#!/usr/bin/env python3
"""训练 Cooperative Monostatic 概率条件化 Fine CNN（串联 Region→Fine）。

标签为全局 ``(x, y)`` RMSE。Region 由 ``--region-checkpoint`` 冻结提供
``softmax`` 概率；Fine 不独立使用。可选 ``--oracle-region-probs`` 用真值
one-hot 做上界 ablation。

示例::

    python script/model_training/run_train_cooperative_monostatic_fine_cnn.py \\
        --region-checkpoint models/cooperative_monostatic_region_cnn/best_model.pth \\
        --epochs 2 --batch-size 32 --max-samples 512
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from isac.models import (
    COOPERATIVE_FEATURE_MODES,
    COOPERATIVE_POOL_MODES,
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    CooperativeMonostaticTwoStageCNN,
    TargetPositionRmseLoss,
    cooperative_feature_in_channels,
    load_cooperative_monostatic_region_cnn_checkpoint,
    subregion_id_to_one_hot,
)
from isac.utils import set_random_seed
from isac_imp.cooperative_monostatic_pipeline import grc_cooperative_processing_params
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    filter_cooperative_frames_energy_mad,
    filter_cooperative_frames_hard,
    is_cooperative_monostatic_features_h5,
    load_cooperative_frame_energy,
    open_cooperative_monostatic_training_dataset,
    resolve_cooperative_features_h5,
    maybe_exclude_subregion_corner_frames,
    session_train_val_split_by_subregion,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT

DEFAULT_H5 = Path(
    "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
)
DEFAULT_OUTPUT_DIR = Path("models/cooperative_monostatic_fine_cnn")
TRAIN_DEFAULT_RANGE_ROI = (0.0, 4.0)
TRAIN_DEFAULT_LR = 5e-5
TRAIN_DEFAULT_BASE_CHANNELS = 64
TRAIN_DEFAULT_NUM_LAYERS = 3
TRAIN_DEFAULT_DROPOUT = 0.3
TRAIN_DEFAULT_POOL_MODE = "attention"
TRAIN_DEFAULT_EPOCHS = 100
TRAIN_DEFAULT_EARLY_STOP_PATIENCE = 15
TRAIN_DEFAULT_SPEC_AUGMENT_PROB = 0.5


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train cooperative monostatic Fine CNN (serial Region→Fine)"
    )
    parser.add_argument(
        "--region-checkpoint",
        type=Path,
        required=True,
        help="冻结 Region checkpoint（必填，串联提供 region_probs）",
    )
    parser.add_argument(
        "--oracle-region-probs",
        action="store_true",
        help="用真值 one-hot 覆盖 Region 概率（oracle 上界 ablation）",
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULT_LR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=list(TRAIN_DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="real_imag",
        choices=list(COOPERATIVE_FEATURE_MODES),
    )
    parser.add_argument("--feature-norm", type=str, default="none")
    parser.add_argument("--base-channels", type=int, default=TRAIN_DEFAULT_BASE_CHANNELS)
    parser.add_argument("--num-layers", type=int, default=TRAIN_DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=TRAIN_DEFAULT_DROPOUT)
    parser.add_argument(
        "--pool-mode",
        type=str,
        default=TRAIN_DEFAULT_POOL_MODE,
        choices=list(COOPERATIVE_POOL_MODES),
    )
    parser.add_argument(
        "--spec-augment-prob",
        type=float,
        default=TRAIN_DEFAULT_SPEC_AUGMENT_PROB,
    )
    parser.add_argument("--spec-augment-max-bins", type=int, default=3)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=TRAIN_DEFAULT_EARLY_STOP_PATIENCE,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--features-h5", type=Path, default=None)
    parser.add_argument(
        "--require-features-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-raw-cpi", action="store_true")
    parser.add_argument("--filter-outliers", action="store_true")
    parser.add_argument(
        "--exclude-subregion-corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "剔除 4x4 四角子区（默认关闭；"
            "--exclude-subregion-corners 开启）"
        ),
    )
    parser.add_argument("--xy-max-m", type=float, default=1.0)
    parser.add_argument("--outlier-energy-eps", type=float, default=1e-8)
    parser.add_argument("--outlier-energy-mad-z", type=float, default=5.0)
    return parser


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "dual_profiles": torch.stack([item["dual_profiles"] for item in batch]),
        "target_xy": torch.stack([item["target_xy"] for item in batch]),
        "target_subregion_id": torch.stack(
            [item["target_subregion_id"] for item in batch]
        ),
        "session_index": torch.stack([item["session_index"] for item in batch]),
    }


def _make_dataset(
    h5_path: Path,
    frame_indices: np.ndarray,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float],
    feature_mode: str,
    feature_norm: str,
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
        label_jitter_m=0.0,
        feature_mode=feature_mode,
        feature_norm=feature_norm,
        feature_noise_std=feature_noise_std,
        spec_augment_prob=spec_augment_prob,
        spec_augment_max_bins=spec_augment_max_bins,
        augment=augment,
        seed=seed,
        return_subregion=True,
    )


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
) -> tuple[np.ndarray, np.ndarray | None]:
    n_frames = int(session_indices.shape[0])
    energy = load_cooperative_frame_energy(h5_path)
    if energy is not None:
        energy = np.asarray(energy[:n_frames], dtype=np.float64)
    keep_hard, _ = filter_cooperative_frames_hard(
        target_position[:, :2],
        xy_max_m=xy_max_m,
        energy_eps=energy_eps,
    )
    keep_set = set(int(i) for i in keep_hard.tolist())
    train_idx = np.asarray(
        [i for i in train_idx if int(i) in keep_set], dtype=np.int64
    )
    if eval_idx is not None:
        eval_idx = np.asarray(
            [i for i in eval_idx if int(i) in keep_set], dtype=np.int64
        )
    if energy is not None and train_idx.size > 0:
        train_idx, _ = filter_cooperative_frames_energy_mad(
            session_indices,
            energy,
            train_idx,
            z_thresh=energy_mad_z,
        )
    if energy is not None and eval_idx is not None and eval_idx.size > 0:
        eval_idx, _ = filter_cooperative_frames_energy_mad(
            session_indices,
            energy,
            eval_idx,
            z_thresh=energy_mad_z,
        )
    if train_idx.size == 0:
        raise RuntimeError("过滤后训练集为空")
    return train_idx, eval_idx


def _run_epoch(
    two_stage: CooperativeMonostaticTwoStageCNN,
    loader: DataLoader,
    criterion: TargetPositionRmseLoss,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    oracle_region_probs: bool,
) -> dict[str, float]:
    # Region 始终冻结；仅 Fine 参与训练
    two_stage.region_model.eval()
    if train:
        two_stage.fine_model.train()
    else:
        two_stage.fine_model.eval()

    total_loss = 0.0
    total_n = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            dual = batch["dual_profiles"].to(device)
            global_tgt = batch["target_xy"].to(device)
            sid = batch["target_subregion_id"].to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            if oracle_region_probs:
                override = subregion_id_to_one_hot(sid, two_stage.num_classes).to(
                    device=device, dtype=dual.dtype
                )
                pred_xy, _logits = two_stage(
                    dual, region_probs_override=override
                )
            else:
                with torch.no_grad():
                    logits = two_stage.region_model(dual)
                    probs = F.softmax(logits, dim=-1)
                pred_xy = two_stage.fine_model(dual, probs)

            loss = criterion(pred_xy, global_tgt)
            if train:
                loss.backward()
                optimizer.step()

            bs = int(global_tgt.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs

    return {
        "global_rmse_m": total_loss / max(total_n, 1),
        "n": float(total_n),
    }


def _plot_history(history: dict[str, list[float]], path: Path) -> None:
    if not history["epoch"]:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_global_rmse"], label="Train global RMSE")
    ax.plot(epochs, history["val_global_rmse"], label="Val global RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Global RMSE (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    if not args.region_checkpoint.is_file():
        raise FileNotFoundError(f"Region checkpoint 不存在: {args.region_checkpoint}")

    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 不存在: {h5_path}")

    range_roi = _parse_range_roi(list(args.range_roi))
    feature_mode = args.feature_mode
    feature_norm = str(args.feature_norm)
    proc_params = grc_cooperative_processing_params()
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA 不可用，回退到 CPU", flush=True)

    paths = TrainPaths(
        checkpoint_dir=args.output_dir.resolve(),
        best_model=args.output_dir.resolve() / "best_model.pth",
        training_curve=args.output_dir.resolve() / "training_curve.png",
    )
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_h5 = h5_path
    if args.force_raw_cpi:
        if is_cooperative_monostatic_features_h5(raw_h5):
            raise ValueError(f"--force-raw-cpi 需要 raw CPI HDF5: {raw_h5}")
        h5_path = raw_h5
    else:
        h5_path = resolve_cooperative_features_h5(
            raw_h5,
            range_roi=range_roi,
            feature_mode=feature_mode,
            features_h5=args.features_h5,
            require=bool(args.require_features_h5),
        )
        if h5_path != raw_h5.resolve():
            print(f"Using features sidecar: {h5_path}", flush=True)

    with h5py.File(h5_path, "r") as f:
        n_frames = int(f[DATASET_KEY_SESSION_INDEX].shape[0])
        session_indices = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        target_position = np.asarray(f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64)

    if args.max_samples is not None:
        n_frames = min(int(args.max_samples), n_frames)
        session_indices = session_indices[:n_frames]
        target_position = target_position[:n_frames]

    train_idx, val_idx, split_info = session_train_val_split_by_subregion(
        session_indices,
        target_position,
        args.val_ratio,
        seed=args.seed,
        exclude_corner_subregions=bool(args.exclude_subregion_corners),
    )
    nonempty = sum(1 for info in split_info.values() if info["train"] + info["val"] > 0)
    print(
        f"Subregion split: {nonempty}/{SUBREGION_COUNT} cells nonempty; "
        f"train={len(train_idx)} val={len(val_idx)}",
        flush=True,
    )

    if args.filter_outliers:
        train_idx, val_idx = _apply_outlier_filters(
            h5_path=h5_path,
            session_indices=session_indices,
            target_position=target_position,
            train_idx=train_idx,
            eval_idx=val_idx,
            xy_max_m=args.xy_max_m,
            energy_eps=args.outlier_energy_eps,
            energy_mad_z=args.outlier_energy_mad_z,
        )
        assert val_idx is not None

    train_idx = maybe_exclude_subregion_corner_frames(
        train_idx,
        target_position,
        enabled=bool(args.exclude_subregion_corners),
        label="train",
    )
    val_idx = maybe_exclude_subregion_corner_frames(
        val_idx,
        target_position,
        enabled=bool(args.exclude_subregion_corners),
        label="val",
    )
    if train_idx.size == 0:
        raise ValueError("exclude-subregion-corners 后训练集为空")

    train_ds = _make_dataset(
        h5_path,
        train_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        feature_mode=feature_mode,
        feature_norm=feature_norm,
        feature_noise_std=args.feature_noise_std,
        spec_augment_prob=args.spec_augment_prob,
        spec_augment_max_bins=args.spec_augment_max_bins,
        augment=True,
        seed=args.seed,
    )
    val_ds = _make_dataset(
        h5_path,
        val_idx,
        proc_params=proc_params,
        range_roi=range_roi,
        feature_mode=feature_mode,
        feature_norm=feature_norm,
        feature_noise_std=0.0,
        spec_augment_prob=0.0,
        spec_augment_max_bins=args.spec_augment_max_bins,
        augment=False,
        seed=args.seed + 1,
    )

    def _worker_init(worker_id: int) -> None:
        if hasattr(train_ds, "reseed"):
            train_ds.reseed(args.seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
        worker_init_fn=_worker_init if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
        pin_memory=device.type == "cuda",
    )

    region_model: CooperativeMonostaticRegionCNN = (
        load_cooperative_monostatic_region_cnn_checkpoint(
            args.region_checkpoint, device
        )
    )
    for p in region_model.parameters():
        p.requires_grad_(False)
    region_model.eval()

    in_channels = cooperative_feature_in_channels(feature_mode)  # type: ignore[arg-type]
    fine_model = CooperativeMonostaticFineCNN(
        in_channels=in_channels,
        base_channels=args.base_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
        num_classes=SUBREGION_COUNT,
    ).to(device)
    two_stage = CooperativeMonostaticTwoStageCNN(region_model, fine_model).to(device)
    criterion = TargetPositionRmseLoss()
    optimizer = torch.optim.AdamW(fine_model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_global_rmse": [],
        "val_global_rmse": [],
    }
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.early_stop_patience

    oracle_tag = "oracle_onehot" if args.oracle_region_probs else "region_softmax"
    print(
        f"Fine CNN (serial) | region={args.region_checkpoint} | "
        f"probs={oracle_tag} | device={device} | checkpoint={paths.best_model}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            two_stage,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            oracle_region_probs=bool(args.oracle_region_probs),
        )
        val_metrics = _run_epoch(
            two_stage,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            oracle_region_probs=bool(args.oracle_region_probs),
        )
        history["epoch"].append(float(epoch))
        history["train_global_rmse"].append(float(train_metrics["global_rmse_m"]))
        history["val_global_rmse"].append(float(val_metrics["global_rmse_m"]))
        print(
            f"Epoch {epoch:03d} | "
            f"train global={train_metrics['global_rmse_m']:.4f} | "
            f"val global={val_metrics['global_rmse_m']:.4f}",
            flush=True,
        )

        if float(val_metrics["global_rmse_m"]) < best_val:
            best_val = float(val_metrics["global_rmse_m"])
            best_epoch = epoch
            patience_left = args.early_stop_patience
            torch.save(
                {
                    "model_state_dict": fine_model.state_dict(),
                    "in_channels": in_channels,
                    "base_channels": args.base_channels,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "model_kind": "fine",
                    "pool_mode": args.pool_mode,
                    "num_classes": SUBREGION_COUNT,
                    "feature_mode": feature_mode,
                    "feature_norm": feature_norm,
                    "range_roi": list(range_roi),
                    "region_checkpoint": str(args.region_checkpoint.resolve()),
                    "oracle_region_probs": bool(args.oracle_region_probs),
                    "val_global_rmse": best_val,
                    "epoch": epoch,
                },
                paths.best_model,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best={best_epoch})", flush=True)
                break

    _plot_history(history, paths.training_curve)
    print(
        tabulate(
            [
                ["Best epoch", best_epoch],
                ["Best val global RMSE", f"{best_val:.4f}"],
                ["Checkpoint", str(paths.best_model)],
            ],
            tablefmt="github",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
