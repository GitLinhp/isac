#!/usr/bin/env python3
"""训练 Cooperative Monostatic 子区域分类 CNN（16 类，0.5 m 4×4）。

独立于现有 ``run_train_cooperative_monostatic_cnn.py``，不覆盖单阶段 (x,y) 方案。

示例::

    python script/model_training/run_train_cooperative_monostatic_region_cnn.py \\
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
    COOPERATIVE_POOL_MODES,
    CooperativeMonostaticRegionCNN,
    TargetSubregionCrossEntropyLoss,
    TargetSubregionTopKSoftmaxCELoss,
    apply_feature_mixup,
    cooperative_feature_in_channels,
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
DEFAULT_OUTPUT_DIR = Path("models/cooperative_monostatic_region_cnn")
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
    best_by_topk_hit: Path
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
        description="Train cooperative monostatic 16-subregion (0.5m 4x4) classifier CNN"
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
        "--label-smoothing",
        type=float,
        default=0.0,
        help="均匀 label smoothing（可与邻域软标签叠加）",
    )
    parser.add_argument(
        "--neighbor-smooth",
        type=float,
        default=0.0,
        help="邻域软标签 α：真值 1-α，四邻接均分 α（默认关闭）",
    )
    parser.add_argument(
        "--topk-ce",
        action="store_true",
        help="启用 forced top-k softmax CE（与推理 top-k 对齐；与 neighbor/label-smooth 互斥）",
    )
    parser.add_argument(
        "--topk-ce-k",
        type=int,
        default=3,
        help="topk-ce 的 k（默认 3）",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="特征 mixup Beta(α,α)；0 关闭",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay（默认 0.01）",
    )
    parser.add_argument("--early-stop-patience", type=int, default=TRAIN_DEFAULT_EARLY_STOP_PATIENCE)
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
        "--class-weight",
        action="store_true",
        help="按训练集子区域频率反比加权 CE",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="加载已有 Region checkpoint 权重再训（结构须兼容）",
    )
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
        "--extra-features-h5",
        type=Path,
        default=None,
        help="extra-h5 的 features sidecar（可选）",
    )
    return parser


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
        label_jitter_m=0.0,  # 分类关闭 jitter
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


def _class_counts(target_position: np.ndarray, indices: np.ndarray) -> np.ndarray:
    from isac_imp.record_target_metadata import target_subregion_index_xy_m

    counts = np.zeros(SUBREGION_COUNT, dtype=np.int64)
    for i in indices:
        sid = target_subregion_index_xy_m(
            float(target_position[i, 0]), float(target_position[i, 1])
        )
        counts[sid] += 1
    return counts


def _sample_mixup_lambda(alpha: float, *, device: torch.device) -> float:
    if alpha <= 0.0:
        return 1.0
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(alpha, device=device),
    )
    return float(dist.sample().item())


def _run_epoch(
    model: CooperativeMonostaticRegionCNN,
    loader: DataLoader,
    criterion: TargetSubregionCrossEntropyLoss | TargetSubregionTopKSoftmaxCELoss,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    mixup_alpha: float = 0.0,
    eval_topk: int = 3,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_topk_hit = 0
    total_n = 0
    conf = np.zeros((SUBREGION_COUNT, SUBREGION_COUNT), dtype=np.int64)
    use_topk_ce = isinstance(criterion, TargetSubregionTopKSoftmaxCELoss)
    k_eval = max(1, int(eval_topk))

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            dual = batch["dual_profiles"].to(device)
            target = batch["target_subregion_id"].to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            use_mixup = train and mixup_alpha > 0.0 and int(dual.shape[0]) >= 2
            if use_mixup:
                lam = _sample_mixup_lambda(mixup_alpha, device=device)
                perm = torch.randperm(dual.shape[0], device=device)
                dual_in = lam * dual + (1.0 - lam) * dual[perm]
                logits = model(dual_in)
                if use_topk_ce:
                    loss = lam * criterion(logits, target) + (1.0 - lam) * criterion(
                        logits, target[perm]
                    )
                else:
                    soft_a = criterion.soft_targets_from_ids(
                        target, dtype=dual.dtype, device=device
                    )
                    soft_b = criterion.soft_targets_from_ids(
                        target[perm], dtype=dual.dtype, device=device
                    )
                    _, soft = apply_feature_mixup(
                        dual, dual[perm], soft_a, soft_b, lam=lam
                    )
                    loss = criterion(logits, target, soft_targets=soft)
            else:
                logits = model(dual)
                loss = criterion(logits, target)

            if train:
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=-1)
            topk_ids = logits.topk(k=min(k_eval, logits.shape[-1]), dim=-1).indices
            hit = (topk_ids == target.unsqueeze(-1)).any(dim=-1)
            bs = int(target.shape[0])
            total_loss += float(loss.item()) * bs
            total_correct += int((pred == target).sum().item())
            total_topk_hit += int(hit.sum().item())
            total_n += bs
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                conf[int(t), int(p)] += 1

    acc = total_correct / max(total_n, 1)
    topk_hit = total_topk_hit / max(total_n, 1)
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": acc,
        "topk_hit": topk_hit,
        "n": float(total_n),
        "confusion": conf,  # type: ignore[dict-item]
    }


def _plot_history(history: dict[str, list[float]], path: Path) -> None:
    if not history["epoch"]:
        return
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    epochs = history["epoch"]
    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_acc.plot(epochs, history["train_acc"], label="Train top-1")
    ax_acc.plot(epochs, history["val_acc"], label="Val top-1")
    if history.get("train_topk_hit") and history.get("val_topk_hit"):
        ax_acc.plot(epochs, history["train_topk_hit"], label="Train top-k hit")
        ax_acc.plot(epochs, history["val_topk_hit"], label="Val top-k hit")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy / hit")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_arg_parser().parse_args()
    set_random_seed(args.seed)

    if args.topk_ce and (
        float(args.neighbor_smooth) > 0.0 or float(args.label_smoothing) > 0.0
    ):
        raise ValueError(
            "--topk-ce 与 --neighbor-smooth / --label-smoothing 互斥"
        )
    if args.topk_ce and int(args.topk_ce_k) < 1:
        raise ValueError(f"--topk-ce-k 须 >= 1，收到 {args.topk_ce_k}")

    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 不存在: {h5_path}")

    range_roi = _parse_range_roi(list(args.range_roi))
    feature_mode = args.feature_mode
    feature_norm = str(args.feature_norm)
    proc_params = grc_cooperative_processing_params()
    device = torch.device(
        args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA 不可用，回退到 CPU", flush=True)

    paths = TrainPaths(
        checkpoint_dir=args.output_dir.resolve(),
        best_model=args.output_dir.resolve() / "best_model.pth",
        best_by_topk_hit=args.output_dir.resolve() / "best_by_topk_hit.pth",
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

    class_weight = None
    if args.class_weight:
        if args.topk_ce:
            raise ValueError("--class-weight 暂不支持与 --topk-ce 同时使用")
        counts = _class_counts(target_position, train_idx)
        # 反频率；空类给 0 权重
        inv = np.zeros(SUBREGION_COUNT, dtype=np.float64)
        mask = counts > 0
        inv[mask] = counts[mask].sum() / (mask.sum() * counts[mask].astype(np.float64))
        class_weight = torch.tensor(inv, dtype=torch.float32)
        print(f"Class weights (nonzero): {(inv > 0).sum()}/{SUBREGION_COUNT}", flush=True)

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

    # 可选：extra H5 仅并入 train
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
        if args.extra_session_list is not None:
            aug_sessions = np.asarray(
                _read_session_list(args.extra_session_list.resolve()),
                dtype=np.int64,
            )
            unique_extra = set(int(s) for s in np.unique(extra_sessions_all).tolist())
            missing = [s for s in aug_sessions.tolist() if int(s) not in unique_extra]
            if missing:
                raise ValueError(
                    f"--extra-session-list 含不在 extra-h5 中的 session: {missing[:10]}"
                )
            holdout_sessions = np.asarray(
                sorted(unique_extra - set(int(s) for s in aug_sessions.tolist())),
                dtype=np.int64,
            )
        else:
            aug_sessions, holdout_sessions = sample_sessions_by_frac(
                extra_sessions_all,
                frac=float(args.extra_session_frac),
                seed=int(args.extra_session_seed),
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
        _write_session_list(paths.checkpoint_dir / "extra_sessions.txt", aug_sessions)
        _write_session_list(
            paths.checkpoint_dir / "extra_holdout_sessions.txt", holdout_sessions
        )
        print(
            f"Extra train: h5={extra_h5} sessions={len(aug_sessions)} "
            f"frames={len(extra_train_idx)} holdout_sessions={len(holdout_sessions)} "
            f"(val 仍仅为主 H5)",
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
        train_ds = ConcatDataset([train_ds, extra_ds])

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
        # ConcatDataset 时尝试 reseed 各子数据集
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
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
        model = load_cooperative_monostatic_region_cnn_checkpoint(resume_path, device)
        model.train()
        resume_meta = torch.load(resume_path, map_location="cpu", weights_only=False)
        # 覆盖超参元数据以匹配 ckpt 结构
        in_channels = int(resume_meta.get("in_channels", in_channels))
        if int(resume_meta.get("base_channels", args.base_channels)) != args.base_channels:
            print(
                f"警告: resume base_channels={resume_meta.get('base_channels')} "
                f"与 CLI {args.base_channels} 不一致，以 checkpoint 为准",
                flush=True,
            )
        print(f"Resumed Region weights from {resume_path}", flush=True)
    else:
        model = CooperativeMonostaticRegionCNN(
            in_channels=in_channels,
            base_channels=args.base_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pool_mode=args.pool_mode,
            num_classes=SUBREGION_COUNT,
        ).to(device)
    topk_ce_k = int(args.topk_ce_k)
    if args.topk_ce:
        criterion: TargetSubregionCrossEntropyLoss | TargetSubregionTopKSoftmaxCELoss = (
            TargetSubregionTopKSoftmaxCELoss(
                num_classes=SUBREGION_COUNT,
                topk=topk_ce_k,
            ).to(device)
        )
        select_metric = "topk_hit"
    else:
        criterion = TargetSubregionCrossEntropyLoss(
            num_classes=SUBREGION_COUNT,
            class_weight=class_weight,
            label_smoothing=float(args.label_smoothing),
            neighbor_smooth=float(args.neighbor_smooth),
        ).to(device)
        select_metric = "acc"
    eval_topk = topk_ce_k if args.topk_ce else 3
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(args.weight_decay),
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_topk_hit": [],
        "val_topk_hit": [],
    }
    best_select = -1.0
    best_epoch = 0
    best_val_acc = -1.0
    best_val_topk_hit = -1.0
    best_topk_select = -1.0
    best_topk_epoch = 0
    patience_left = args.early_stop_patience

    print(
        f"Region CNN | classes={SUBREGION_COUNT} | device={device} | "
        f"topk_ce={args.topk_ce} k={topk_ce_k} select={select_metric} | "
        f"neighbor_smooth={args.neighbor_smooth} label_smoothing={args.label_smoothing} "
        f"mixup_alpha={args.mixup_alpha} noise={args.feature_noise_std} "
        f"spec_aug={args.spec_augment_prob} wd={args.weight_decay} | "
        f"checkpoint={paths.best_model}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            mixup_alpha=float(args.mixup_alpha),
            eval_topk=eval_topk,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
            mixup_alpha=0.0,
            eval_topk=eval_topk,
        )
        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["acc"]))
        history["val_acc"].append(float(val_metrics["acc"]))
        history["train_topk_hit"].append(float(train_metrics["topk_hit"]))
        history["val_topk_hit"].append(float(val_metrics["topk_hit"]))
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['acc']:.4f} top{eval_topk}={train_metrics['topk_hit']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} top{eval_topk}={val_metrics['topk_hit']:.4f}",
            flush=True,
        )

        select_val = float(val_metrics[select_metric])
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "model_kind": "region",
            "pool_mode": args.pool_mode,
            "num_classes": SUBREGION_COUNT,
            "feature_mode": feature_mode,
            "feature_norm": feature_norm,
            "range_roi": list(range_roi),
            "label_smoothing": float(args.label_smoothing),
            "neighbor_smooth": float(args.neighbor_smooth),
            "topk_ce": bool(args.topk_ce),
            "topk_ce_k": topk_ce_k,
            "mixup_alpha": float(args.mixup_alpha),
            "feature_noise_std": float(args.feature_noise_std),
            "spec_augment_prob": float(args.spec_augment_prob),
            "weight_decay": float(args.weight_decay),
            "val_acc": float(val_metrics["acc"]),
            "val_topk_hit": float(val_metrics["topk_hit"]),
            "epoch": epoch,
        }
        # 另存按 Run1 val top-k hit 选的对照 ckpt（主 early-stop 仍看 select_metric）
        topk_val = float(val_metrics["topk_hit"])
        if topk_val > best_topk_select:
            best_topk_select = topk_val
            best_topk_epoch = epoch
            torch.save(
                {**ckpt_payload, "select_metric": "topk_hit"},
                paths.best_by_topk_hit,
            )
        if select_val > best_select:
            best_select = select_val
            best_epoch = epoch
            best_val_acc = float(val_metrics["acc"])
            best_val_topk_hit = float(val_metrics["topk_hit"])
            patience_left = args.early_stop_patience
            torch.save(
                {**ckpt_payload, "select_metric": select_metric},
                paths.best_model,
            )
            conf = val_metrics["confusion"]
            np.save(paths.checkpoint_dir / "best_val_confusion.npy", conf)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best={best_epoch})", flush=True)
                break

    _plot_history(history, paths.training_curve)
    print(
        tabulate(
            [
                ["Best epoch (select)", best_epoch],
                ["Select metric", select_metric],
                ["Best val select", f"{best_select:.4f}"],
                ["Best val top-1", f"{best_val_acc:.4f}"],
                [f"Best val top-{eval_topk} hit", f"{best_val_topk_hit:.4f}"],
                ["Best topk-hit epoch", best_topk_epoch],
                ["Best topk-hit", f"{best_topk_select:.4f}"],
                ["Checkpoint", str(paths.best_model)],
                ["Checkpoint (by topk hit)", str(paths.best_by_topk_hit)],
            ],
            tablefmt="github",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
