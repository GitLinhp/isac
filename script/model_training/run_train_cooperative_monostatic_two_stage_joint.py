#!/usr/bin/env python3
"""两阶段端到端联合训练：Region → Fine 串联，损失=全局 RMSE。

支持两种启动方式：

1. **热启动联合**（默认历史路径）：加载已训 Region（及可选 Fine）再联合微调
2. **直接联合**（``--region-checkpoint`` 省略）：Region+Fine 均随机初始化，
   从零串联同训；建议加 ``--region-ce-weight`` 辅助区域分类

推理路径为 ``CooperativeMonostaticTwoStageCNN`` 串联前向。

示例::

    # 热启动
    python script/model_training/run_train_cooperative_monostatic_two_stage_joint.py \\
        --region-checkpoint models/two_stage_tune/region/region_drop01/best_model.pth \\
        --fine-checkpoint models/two_stage_tune/fine/fine_lr1e4/best_model.pth \\
        --epochs 2 --batch-size 32 --max-samples 512

    # 直接联合（从零）
    python script/model_training/run_train_cooperative_monostatic_two_stage_joint.py \\
        --output-dir models/two_stage_joint/direct \\
        --region-ce-weight 1.0 --region-lr 1e-4 --lr 1e-4 \\
        --epochs 2 --batch-size 32 --max-samples 512

    # Run2 10% session 并入 train（val 仍为主 H5）
    python script/model_training/run_train_cooperative_monostatic_two_stage_joint.py \\
        --region-checkpoint models/two_stage_joint/joint_direct/best_region.pth \\
        --fine-checkpoint models/two_stage_joint/joint_direct/best_fine.pth \\
        --extra-h5 data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5 \\
        --extra-session-list models/region_run2_10pct/split/run2_aug_sessions.txt \\
        --output-dir models/two_stage_joint_run2_10pct/joint_ft \\
        --lr 1e-5 --region-lr 1e-5 --epochs 50
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
import torch.nn.functional as F
from tabulate import tabulate
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from isac.models import (
    COOPERATIVE_FEATURE_MODES,
    COOPERATIVE_POOL_MODES,
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    CooperativeMonostaticTwoStageCNN,
    TargetPositionRmseLoss,
    TargetSubregionCrossEntropyLoss,
    cooperative_feature_in_channels,
    load_cooperative_monostatic_fine_cnn_checkpoint,
    load_cooperative_monostatic_region_cnn_checkpoint,
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
DEFAULT_OUTPUT_DIR = Path("models/cooperative_monostatic_two_stage_joint")
TRAIN_DEFAULT_RANGE_ROI = (0.0, 4.0)
TRAIN_DEFAULT_LR = 1e-4
TRAIN_DEFAULT_REGION_LR = 1e-5
TRAIN_DEFAULT_BASE_CHANNELS = 64
TRAIN_DEFAULT_NUM_LAYERS = 3
TRAIN_DEFAULT_DROPOUT = 0.1
TRAIN_DEFAULT_POOL_MODE = "attention"
TRAIN_DEFAULT_EPOCHS = 100
TRAIN_DEFAULT_EARLY_STOP_PATIENCE = 12
TRAIN_DEFAULT_SPEC_AUGMENT_PROB = 0.5
TRAIN_DEFAULT_REGION_TOPK = 3
TRAIN_DEFAULT_REGION_CE_WEIGHT = 0.0


@dataclass(frozen=True)
class TrainPaths:
    checkpoint_dir: Path
    best_region: Path
    best_fine: Path
    joint_meta: Path
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
        description=(
            "Joint train Region→Fine serial pipeline with global RMSE "
            "(warm-start or direct from-scratch)"
        )
    )
    parser.add_argument(
        "--region-checkpoint",
        type=Path,
        default=None,
        help="Region 热启动 checkpoint；省略则为直接联合（Region 随机初始化）",
    )
    parser.add_argument(
        "--fine-checkpoint",
        type=Path,
        default=None,
        help="Fine 热启动 checkpoint（缺省则新建 Fine）",
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULT_LR, help="Fine lr")
    parser.add_argument(
        "--region-lr",
        type=float,
        default=None,
        help=(
            "Region lr；缺省：热启动用 1e-5，直接联合用与 --lr 相同"
        ),
    )
    parser.add_argument(
        "--region-ce-weight",
        type=float,
        default=TRAIN_DEFAULT_REGION_CE_WEIGHT,
        help="Region CE 辅助损失权重（直接联合建议 >0，如 1.0）",
    )
    parser.add_argument(
        "--neighbor-smooth",
        type=float,
        default=0.0,
        help="Region CE 邻域软标签 α（0=标准硬标签）",
    )
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
        "--region-topk",
        type=int,
        default=TRAIN_DEFAULT_REGION_TOPK,
        help="Region top-k 指标（不参与 xy 融合）",
    )
    parser.add_argument(
        "--spec-augment-prob",
        type=float,
        default=TRAIN_DEFAULT_SPEC_AUGMENT_PROB,
    )
    parser.add_argument("--spec-augment-max-bins", type=int, default=3)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    parser.add_argument(
        "--extra-h5",
        type=Path,
        default=None,
        help="额外训练 H5（仅并入 train，不进 val）",
    )
    parser.add_argument(
        "--extra-session-frac",
        type=float,
        default=0.0,
        help="从 extra-h5 按 session 抽样比例并入 train（0=关闭；与 extra-session-list 互斥优先 list）",
    )
    parser.add_argument(
        "--extra-session-seed",
        type=int,
        default=42,
        help="extra session 抽样 seed",
    )
    parser.add_argument(
        "--extra-session-list",
        type=Path,
        default=None,
        help="显式 session id 列表文件（每行一个 int）；若给定则忽略 frac",
    )
    parser.add_argument(
        "--extra-val-session-list",
        type=Path,
        default=None,
        help="extra H5 中仅用于验证的 session 列表（与 train list 须无交）",
    )
    parser.add_argument(
        "--extra-features-h5",
        type=Path,
        default=None,
        help="extra-h5 的 features sidecar（可选）",
    )
    parser.add_argument(
        "--early-stop-on",
        type=str,
        default="run1_val",
        choices=("run1_val", "extra_val"),
        help="early-stop / 存 best 的指标：run1_val（默认）或 extra_val",
    )
    parser.add_argument(
        "--freeze-fine",
        action="store_true",
        help="冻结 Fine，仅更新 Region",
    )
    parser.add_argument(
        "--extra-oversample",
        type=int,
        default=1,
        help="将 extra train dataset 在 Concat 中重复 N 次（N>=1）",
    )
    return parser


def split_session_list_train_val(
    sessions: np.ndarray | list[int],
    *,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将 session 列表拆成 (train, val)，无交且覆盖全部。"""
    unique = np.unique(np.asarray(sessions, dtype=np.int64))
    if unique.size == 0:
        raise ValueError("sessions 为空")
    n_val_i = int(n_val)
    if n_val_i < 1 or n_val_i >= int(unique.size):
        raise ValueError(
            f"n_val 须在 [1, n_sessions-1]，收到 n_val={n_val_i} n={unique.size}"
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(unique)
    val = np.sort(perm[:n_val_i])
    train = np.sort(perm[n_val_i:])
    return train, val


def _read_session_list(path: Path) -> list[int]:
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(int(s))
    if not ids:
        raise ValueError(f"session 列表为空: {path}")
    return ids


def _write_session_list(path: Path, sessions: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in sessions:
            f.write(f"{int(sid)}\n")


def sample_sessions_by_frac(
    session_indices: np.ndarray,
    *,
    frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 session 抽 ``frac``，返回 (aug_sessions, holdout_sessions)。"""
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac 须在 (0, 1]，收到 {frac}")
    unique = np.unique(np.asarray(session_indices, dtype=np.int64))
    if unique.size == 0:
        raise ValueError("session_indices 为空")
    rng = np.random.default_rng(int(seed))
    n_aug = max(1, int(round(float(frac) * int(unique.size))))
    n_aug = min(n_aug, int(unique.size))
    perm = rng.permutation(unique)
    aug = np.sort(perm[:n_aug])
    holdout = np.sort(perm[n_aug:])
    return aug, holdout


def frames_for_sessions(
    session_indices: np.ndarray,
    sessions: np.ndarray | list[int],
) -> np.ndarray:
    want = set(int(s) for s in np.asarray(sessions, dtype=np.int64).tolist())
    return np.asarray(
        [i for i, s in enumerate(session_indices.tolist()) if int(s) in want],
        dtype=np.int64,
    )


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
    region_topk: int,
    region_ce_weight: float = 0.0,
    region_ce_criterion: TargetSubregionCrossEntropyLoss | None = None,
    freeze_fine: bool = False,
) -> dict[str, float]:
    if train:
        two_stage.train()
        if freeze_fine:
            two_stage.fine_model.eval()
    else:
        two_stage.eval()

    total_rmse = 0.0
    total_ce = 0.0
    total_top1 = 0
    total_topk_hit = 0
    total_n = 0
    k = max(1, int(region_topk))
    ce_w = float(region_ce_weight)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            dual = batch["dual_profiles"].to(device)
            global_tgt = batch["target_xy"].to(device)
            true_sid = batch["target_subregion_id"].to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            pred_xy, logits = two_stage(dual)
            rmse_loss = criterion(pred_xy, global_tgt)
            loss = rmse_loss
            ce_val = 0.0
            if ce_w > 0.0:
                assert region_ce_criterion is not None
                ce_loss = region_ce_criterion(logits, true_sid)
                loss = loss + ce_w * ce_loss
                ce_val = float(ce_loss.item())
            if train:
                loss.backward()
                optimizer.step()

            probs = F.softmax(logits.detach(), dim=-1)
            topk_probs_raw, topk_ids = torch.topk(probs, k=k, dim=-1)
            pred_top1 = topk_ids[:, 0]
            hit = (topk_ids == true_sid.unsqueeze(-1)).any(dim=-1)
            bs = int(global_tgt.shape[0])
            total_rmse += float(rmse_loss.item()) * bs
            total_ce += ce_val * bs
            total_top1 += int((pred_top1 == true_sid).sum().item())
            total_topk_hit += int(hit.sum().item())
            total_n += bs
            _ = topk_probs_raw  # silence unused

    return {
        "global_rmse_m": total_rmse / max(total_n, 1),
        "region_ce": total_ce / max(total_n, 1),
        "region_top1_acc": total_top1 / max(total_n, 1),
        "region_topk_hit": total_topk_hit / max(total_n, 1),
        "n": float(total_n),
    }


def _plot_history(history: dict[str, list[float]], path: Path) -> None:
    if not history["epoch"]:
        return
    fig, (ax_rmse, ax_hit) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    epochs = history["epoch"]
    ax_rmse.plot(epochs, history["train_global_rmse"], label="Train global RMSE")
    ax_rmse.plot(epochs, history["val_global_rmse"], label="Val global RMSE")
    ax_rmse.set_ylabel("Global RMSE (m)")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)
    ax_hit.plot(epochs, history["train_top1"], label="Train top-1")
    ax_hit.plot(epochs, history["val_top1"], label="Val top-1")
    ax_hit.plot(epochs, history["train_topk_hit"], label="Train top-k hit")
    ax_hit.plot(epochs, history["val_topk_hit"], label="Val top-k hit")
    ax_hit.set_xlabel("Epoch")
    ax_hit.set_ylabel("Region acc / hit")
    ax_hit.legend()
    ax_hit.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_region_ckpt(
    path: Path,
    *,
    model: CooperativeMonostaticRegionCNN,
    meta: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": "region",
            **meta,
        },
        path,
    )


def _save_fine_ckpt(
    path: Path,
    *,
    model: CooperativeMonostaticFineCNN,
    meta: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": "fine",
            **meta,
        },
        path,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    if int(args.region_topk) < 1:
        raise ValueError(f"--region-topk 须 >= 1，收到 {args.region_topk}")
    if float(args.region_ce_weight) < 0.0:
        raise ValueError(
            f"--region-ce-weight 须 >= 0，收到 {args.region_ce_weight}"
        )
    if float(args.neighbor_smooth) < 0.0 or float(args.neighbor_smooth) >= 1.0:
        raise ValueError(
            f"--neighbor-smooth 须在 [0, 1)，收到 {args.neighbor_smooth}"
        )
    if int(args.extra_oversample) < 1:
        raise ValueError(f"--extra-oversample 须 >= 1，收到 {args.extra_oversample}")
    early_stop_on = str(args.early_stop_on)
    if early_stop_on == "extra_val" and args.extra_val_session_list is None:
        raise ValueError("--early-stop-on extra_val 时须提供 --extra-val-session-list")

    direct_joint = args.region_checkpoint is None
    if args.region_checkpoint is not None and not args.region_checkpoint.is_file():
        raise FileNotFoundError(f"Region checkpoint 不存在: {args.region_checkpoint}")
    if direct_joint and args.fine_checkpoint is not None:
        raise ValueError(
            "直接联合（无 --region-checkpoint）时不支持单独热启动 Fine；"
            "请省略 --fine-checkpoint，或同时提供 --region-checkpoint"
        )

    # Region lr：直接联合默认与 Fine 相同；热启动默认更小
    if args.region_lr is None:
        region_lr = float(args.lr) if direct_joint else float(TRAIN_DEFAULT_REGION_LR)
    else:
        region_lr = float(args.region_lr)

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
        best_region=args.output_dir.resolve() / "best_region.pth",
        best_fine=args.output_dir.resolve() / "best_fine.pth",
        joint_meta=args.output_dir.resolve() / "joint_meta.pth",
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

    # 可选：extra H5 并入 train；可选 extra_val 作跨域选模
    extra_val_loader: DataLoader | None = None
    if args.extra_h5 is not None:
        if args.extra_session_list is None and float(args.extra_session_frac) <= 0.0:
            raise ValueError(
                "指定 --extra-h5 时须提供 --extra-session-list 或 --extra-session-frac > 0"
            )
        extra_raw = args.extra_h5.resolve()
        if not extra_raw.is_file():
            raise FileNotFoundError(f"extra HDF5 不存在: {extra_raw}")
        if args.force_raw_cpi:
            extra_h5 = extra_raw
        else:
            extra_h5 = resolve_cooperative_features_h5(
                extra_raw,
                range_roi=range_roi,
                feature_mode=feature_mode,
                features_h5=args.extra_features_h5,
                require=bool(args.require_features_h5),
            )
            if extra_h5 != extra_raw:
                print(f"Using extra features sidecar: {extra_h5}", flush=True)
        with h5py.File(extra_h5, "r") as f:
            extra_sessions_all = np.asarray(
                f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64
            )
            extra_targets_all = np.asarray(
                f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
            )
        unique_extra = set(int(s) for s in np.unique(extra_sessions_all).tolist())
        if args.extra_session_list is not None:
            aug_sessions = np.asarray(
                _read_session_list(args.extra_session_list.resolve()),
                dtype=np.int64,
            )
            missing = [s for s in aug_sessions.tolist() if int(s) not in unique_extra]
            if missing:
                raise ValueError(
                    f"--extra-session-list 含不在 extra-h5 中的 session: {missing[:10]}"
                )
        else:
            aug_sessions, _ = sample_sessions_by_frac(
                extra_sessions_all,
                frac=float(args.extra_session_frac),
                seed=int(args.extra_session_seed),
            )

        val_extra_sessions = np.asarray([], dtype=np.int64)
        if args.extra_val_session_list is not None:
            val_extra_sessions = np.asarray(
                _read_session_list(args.extra_val_session_list.resolve()),
                dtype=np.int64,
            )
            missing_v = [
                s for s in val_extra_sessions.tolist() if int(s) not in unique_extra
            ]
            if missing_v:
                raise ValueError(
                    f"--extra-val-session-list 含不在 extra-h5 中的 session: "
                    f"{missing_v[:10]}"
                )
            train_set = set(int(s) for s in aug_sessions.tolist())
            val_set = set(int(s) for s in val_extra_sessions.tolist())
            inter = train_set & val_set
            if inter:
                raise ValueError(
                    f"extra train/val session 有交集: {sorted(inter)[:10]}"
                )

        holdout_sessions = np.asarray(
            sorted(
                unique_extra
                - set(int(s) for s in aug_sessions.tolist())
                - set(int(s) for s in val_extra_sessions.tolist())
            ),
            dtype=np.int64,
        )
        extra_train_idx = frames_for_sessions(extra_sessions_all, aug_sessions)
        if extra_train_idx.size == 0:
            raise RuntimeError("extra session 抽样后无训练帧")
        extra_train_idx = maybe_exclude_subregion_corner_frames(
            extra_train_idx,
            extra_targets_all,
            enabled=bool(args.exclude_subregion_corners),
            label="extra-train",
        )
        if extra_train_idx.size == 0:
            raise RuntimeError("exclude-subregion-corners 后 extra 训练帧为空")
        _write_session_list(
            paths.checkpoint_dir / "extra_train_sessions.txt", aug_sessions
        )
        _write_session_list(paths.checkpoint_dir / "extra_sessions.txt", aug_sessions)
        _write_session_list(
            paths.checkpoint_dir / "extra_holdout_sessions.txt", holdout_sessions
        )
        if val_extra_sessions.size > 0:
            _write_session_list(
                paths.checkpoint_dir / "extra_val_sessions.txt", val_extra_sessions
            )
        oversample = int(args.extra_oversample)
        print(
            f"Extra train: h5={extra_h5} sessions={len(aug_sessions)} "
            f"frames={len(extra_train_idx)} oversample={oversample} "
            f"extra_val_sessions={len(val_extra_sessions)} "
            f"holdout_sessions={len(holdout_sessions)}",
            flush=True,
        )
        extra_ds = _make_dataset(
            extra_h5,
            extra_train_idx,
            proc_params=proc_params,
            range_roi=range_roi,
            feature_mode=feature_mode,
            feature_norm=feature_norm,
            feature_noise_std=args.feature_noise_std,
            spec_augment_prob=args.spec_augment_prob,
            spec_augment_max_bins=args.spec_augment_max_bins,
            augment=True,
            seed=args.seed + 7,
        )
        train_parts = [train_ds] + [extra_ds] * oversample
        train_ds = ConcatDataset(train_parts)

        if val_extra_sessions.size > 0:
            extra_val_idx = frames_for_sessions(
                extra_sessions_all, val_extra_sessions
            )
            if extra_val_idx.size == 0:
                raise RuntimeError("extra val session 无帧")
            extra_val_idx = maybe_exclude_subregion_corner_frames(
                extra_val_idx,
                extra_targets_all,
                enabled=bool(args.exclude_subregion_corners),
                label="extra-val",
            )
            if extra_val_idx.size == 0:
                raise RuntimeError("exclude-subregion-corners 后 extra val 为空")
            extra_val_ds = _make_dataset(
                extra_h5,
                extra_val_idx,
                proc_params=proc_params,
                range_roi=range_roi,
                feature_mode=feature_mode,
                feature_norm=feature_norm,
                feature_noise_std=0.0,
                spec_augment_prob=0.0,
                spec_augment_max_bins=args.spec_augment_max_bins,
                augment=False,
                seed=args.seed + 11,
            )
            extra_val_loader = DataLoader(
                extra_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=_collate_batch,
                pin_memory=device.type == "cuda",
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
        worker_seed = args.seed + worker_id
        datasets = getattr(train_ds, "datasets", [train_ds])
        for ds in datasets:
            if hasattr(ds, "reseed"):
                ds.reseed(worker_seed)

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

    in_channels = cooperative_feature_in_channels(feature_mode)  # type: ignore[arg-type]
    region_base_channels = args.base_channels
    region_num_layers = args.num_layers
    region_dropout = args.dropout
    region_pool = args.pool_mode

    if direct_joint:
        region_model = CooperativeMonostaticRegionCNN(
            in_channels=in_channels,
            base_channels=region_base_channels,
            num_layers=region_num_layers,
            dropout=region_dropout,
            pool_mode=region_pool,
            num_classes=SUBREGION_COUNT,
        ).to(device)
    else:
        assert args.region_checkpoint is not None
        region_model = load_cooperative_monostatic_region_cnn_checkpoint(
            args.region_checkpoint, device
        )
        region_model.train()
        region_init = torch.load(
            args.region_checkpoint, map_location="cpu", weights_only=False
        )
        in_channels = int(region_init.get("in_channels", in_channels))
        region_base_channels = int(
            region_init.get("base_channels", args.base_channels)
        )
        region_num_layers = int(region_init.get("num_layers", args.num_layers))
        region_dropout = float(region_init.get("dropout", args.dropout))
        region_pool = str(region_init.get("pool_mode", args.pool_mode))

    if args.fine_checkpoint is not None:
        if not args.fine_checkpoint.is_file():
            raise FileNotFoundError(f"Fine checkpoint 不存在: {args.fine_checkpoint}")
        try:
            fine_model = load_cooperative_monostatic_fine_cnn_checkpoint(
                args.fine_checkpoint, device
            )
            fine_model.train()
            fine_init = torch.load(
                args.fine_checkpoint, map_location="cpu", weights_only=False
            )
            fine_base_channels = int(fine_init.get("base_channels", args.base_channels))
            fine_num_layers = int(fine_init.get("num_layers", args.num_layers))
            fine_dropout = float(fine_init.get("dropout", args.dropout))
            fine_pool = str(fine_init.get("pool_mode", args.pool_mode))
        except ValueError as exc:
            print(
                f"警告: Fine 热启动失败（{exc}），改为新建 Fine",
                flush=True,
            )
            fine_base_channels = args.base_channels
            fine_num_layers = args.num_layers
            fine_dropout = args.dropout
            fine_pool = args.pool_mode
            fine_model = CooperativeMonostaticFineCNN(
                in_channels=in_channels,
                base_channels=fine_base_channels,
                num_layers=fine_num_layers,
                dropout=fine_dropout,
                pool_mode=fine_pool,
                num_classes=SUBREGION_COUNT,
            ).to(device)
    else:
        fine_base_channels = args.base_channels
        fine_num_layers = args.num_layers
        fine_dropout = args.dropout
        fine_pool = args.pool_mode
        fine_model = CooperativeMonostaticFineCNN(
            in_channels=in_channels,
            base_channels=fine_base_channels,
            num_layers=fine_num_layers,
            dropout=fine_dropout,
            pool_mode=fine_pool,
            num_classes=SUBREGION_COUNT,
        ).to(device)

    two_stage = CooperativeMonostaticTwoStageCNN(region_model, fine_model).to(device)
    freeze_fine = bool(args.freeze_fine)
    if freeze_fine:
        for p in fine_model.parameters():
            p.requires_grad = False
        fine_model.eval()
    criterion = TargetPositionRmseLoss()
    region_ce_weight = float(args.region_ce_weight)
    region_ce_criterion: TargetSubregionCrossEntropyLoss | None = None
    if region_ce_weight > 0.0:
        region_ce_criterion = TargetSubregionCrossEntropyLoss(
            num_classes=SUBREGION_COUNT,
            neighbor_smooth=float(args.neighbor_smooth),
        )
    if freeze_fine:
        optimizer = torch.optim.AdamW(
            [{"params": region_model.parameters(), "lr": region_lr}],
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": region_model.parameters(), "lr": region_lr},
                {"params": fine_model.parameters(), "lr": float(args.lr)},
            ],
            weight_decay=float(args.weight_decay),
        )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_global_rmse": [],
        "val_global_rmse": [],
        "val_extra_rmse": [],
        "train_top1": [],
        "val_top1": [],
        "train_topk_hit": [],
        "val_topk_hit": [],
    }
    best_select_rmse = float("inf")
    best_epoch = 0
    best_val_top1 = -1.0
    best_val_topk = -1.0
    best_run1_val_rmse = float("inf")
    best_extra_val_rmse = float("nan")
    patience_left = args.early_stop_patience
    region_topk = int(args.region_topk)
    mode_tag = "direct" if direct_joint else "warm-start"

    print(
        f"Joint two-stage ({mode_tag}) | topk_metrics={region_topk} | "
        f"device={device} | region_lr={region_lr} fine_lr={args.lr} | "
        f"freeze_fine={freeze_fine} | early_stop_on={early_stop_on} | "
        f"region_ce_weight={region_ce_weight} | "
        f"region_init={args.region_checkpoint} | "
        f"fine_init={args.fine_checkpoint} | "
        f"out={paths.checkpoint_dir}",
        flush=True,
    )

    region_meta_base = {
        "in_channels": in_channels,
        "base_channels": region_base_channels,
        "num_layers": region_num_layers,
        "dropout": region_dropout,
        "pool_mode": region_pool,
        "num_classes": SUBREGION_COUNT,
        "feature_mode": feature_mode,
        "feature_norm": feature_norm,
        "range_roi": list(range_roi),
        "region_topk": region_topk,
        "joint_train": True,
        "direct_joint": direct_joint,
        "region_ce_weight": region_ce_weight,
        "early_stop_on": early_stop_on,
        "freeze_fine": freeze_fine,
    }
    fine_meta_base = {
        "in_channels": in_channels,
        "base_channels": fine_base_channels,
        "num_layers": fine_num_layers,
        "dropout": fine_dropout,
        "pool_mode": fine_pool,
        "num_classes": SUBREGION_COUNT,
        "feature_mode": feature_mode,
        "feature_norm": feature_norm,
        "range_roi": list(range_roi),
        "region_topk": region_topk,
        "joint_train": True,
        "direct_joint": direct_joint,
        "freeze_fine": freeze_fine,
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            two_stage,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            region_topk=region_topk,
            region_ce_weight=region_ce_weight,
            region_ce_criterion=region_ce_criterion,
            freeze_fine=freeze_fine,
        )
        val_metrics = _run_epoch(
            two_stage,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            region_topk=region_topk,
            region_ce_weight=region_ce_weight,
            region_ce_criterion=region_ce_criterion,
            freeze_fine=freeze_fine,
        )
        extra_val_metrics: dict[str, float] | None = None
        if extra_val_loader is not None:
            extra_val_metrics = _run_epoch(
                two_stage,
                extra_val_loader,
                criterion,
                optimizer=None,
                device=device,
                train=False,
                region_topk=region_topk,
                region_ce_weight=region_ce_weight,
                region_ce_criterion=region_ce_criterion,
                freeze_fine=freeze_fine,
            )
        history["epoch"].append(float(epoch))
        history["train_global_rmse"].append(float(train_metrics["global_rmse_m"]))
        history["val_global_rmse"].append(float(val_metrics["global_rmse_m"]))
        history["val_extra_rmse"].append(
            float(extra_val_metrics["global_rmse_m"])
            if extra_val_metrics is not None
            else float("nan")
        )
        history["train_top1"].append(float(train_metrics["region_top1_acc"]))
        history["val_top1"].append(float(val_metrics["region_top1_acc"]))
        history["train_topk_hit"].append(float(train_metrics["region_topk_hit"]))
        history["val_topk_hit"].append(float(val_metrics["region_topk_hit"]))
        ce_msg = ""
        if region_ce_weight > 0.0:
            ce_msg = (
                f" ce={train_metrics['region_ce']:.4f}/{val_metrics['region_ce']:.4f}"
            )
        extra_msg = ""
        if extra_val_metrics is not None:
            extra_msg = (
                f" | extra_val rmse={extra_val_metrics['global_rmse_m']:.4f} "
                f"top1={extra_val_metrics['region_top1_acc']:.4f}"
            )
        print(
            f"Epoch {epoch:03d} | "
            f"train rmse={train_metrics['global_rmse_m']:.4f} "
            f"top1={train_metrics['region_top1_acc']:.4f} "
            f"top{region_topk}={train_metrics['region_topk_hit']:.4f} | "
            f"val rmse={val_metrics['global_rmse_m']:.4f} "
            f"top1={val_metrics['region_top1_acc']:.4f} "
            f"top{region_topk}={val_metrics['region_topk_hit']:.4f}"
            f"{extra_msg}{ce_msg}",
            flush=True,
        )

        run1_val_rmse = float(val_metrics["global_rmse_m"])
        if early_stop_on == "extra_val":
            assert extra_val_metrics is not None
            select_rmse = float(extra_val_metrics["global_rmse_m"])
            select_top1 = float(extra_val_metrics["region_top1_acc"])
            select_topk = float(extra_val_metrics["region_topk_hit"])
        else:
            select_rmse = run1_val_rmse
            select_top1 = float(val_metrics["region_top1_acc"])
            select_topk = float(val_metrics["region_topk_hit"])

        if select_rmse < best_select_rmse:
            best_select_rmse = select_rmse
            best_epoch = epoch
            best_val_top1 = select_top1
            best_val_topk = select_topk
            best_run1_val_rmse = run1_val_rmse
            best_extra_val_rmse = (
                float(extra_val_metrics["global_rmse_m"])
                if extra_val_metrics is not None
                else float("nan")
            )
            patience_left = args.early_stop_patience
            region_meta = {
                **region_meta_base,
                "val_global_rmse": best_run1_val_rmse,
                "val_extra_rmse": best_extra_val_rmse,
                "select_rmse": best_select_rmse,
                "val_acc": best_val_top1,
                "val_topk_hit": best_val_topk,
                "epoch": epoch,
            }
            fine_meta = {
                **fine_meta_base,
                "val_global_rmse": best_run1_val_rmse,
                "val_extra_rmse": best_extra_val_rmse,
                "select_rmse": best_select_rmse,
                "epoch": epoch,
            }
            _save_region_ckpt(paths.best_region, model=region_model, meta=region_meta)
            _save_fine_ckpt(paths.best_fine, model=fine_model, meta=fine_meta)
            torch.save(
                {
                    "epoch": epoch,
                    "val_global_rmse": best_run1_val_rmse,
                    "val_extra_rmse": best_extra_val_rmse,
                    "select_rmse": best_select_rmse,
                    "early_stop_on": early_stop_on,
                    "freeze_fine": freeze_fine,
                    "val_region_top1": best_val_top1,
                    "val_region_topk_hit": best_val_topk,
                    "region_topk": region_topk,
                    "direct_joint": direct_joint,
                    "region_ce_weight": region_ce_weight,
                    "region_checkpoint": str(paths.best_region),
                    "fine_checkpoint": str(paths.best_fine),
                    "region_init": str(args.region_checkpoint.resolve())
                    if args.region_checkpoint
                    else "",
                    "fine_init": str(args.fine_checkpoint.resolve())
                    if args.fine_checkpoint
                    else "",
                },
                paths.joint_meta,
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
                ["Mode", mode_tag],
                ["Early-stop on", early_stop_on],
                ["Freeze fine", freeze_fine],
                ["Best epoch", best_epoch],
                ["Best select RMSE", f"{best_select_rmse:.4f}"],
                ["Best Run1 val RMSE", f"{best_run1_val_rmse:.4f}"],
                [
                    "Best extra val RMSE",
                    f"{best_extra_val_rmse:.4f}"
                    if best_extra_val_rmse == best_extra_val_rmse
                    else "n/a",
                ],
                ["Best val region top-1", f"{best_val_top1:.4f}"],
                [f"Best val region top-{region_topk}", f"{best_val_topk:.4f}"],
                ["Region ckpt", str(paths.best_region)],
                ["Fine ckpt", str(paths.best_fine)],
            ],
            tablefmt="github",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
