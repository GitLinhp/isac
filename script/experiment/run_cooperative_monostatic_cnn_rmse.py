#!/usr/bin/env python3
"""Cooperative monostatic HDF5 数据集：CNN 双站定位 RMSE 评估。

divide CPI → ROI 复数距离谱 → CooperativeMonostaticCNN → (x, y) 回归。

示例::

    python script/experiment/run_cooperative_monostatic_cnn_rmse.py \\
        --checkpoint models/cooperative_monostatic_cnn/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.models import load_cooperative_monostatic_cnn_checkpoint
from isac.models.preprocess import (
    cooperative_uses_slowtime_input,
    divide_cpi_dual_to_roi_range_profiles_np,
    divide_cpi_dual_to_roi_range_slowtime_np,
    dual_roi_to_model_input,
    dual_slowtime_to_model_input,
    load_cooperative_norm_stats,
)
from isac.sensing.localization import position_rmse_xy
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FEATURES,
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
    cooperative_frame_cpi_energy,
    filter_cooperative_frames_energy_mad,
    filter_cooperative_frames_hard,
    is_cooperative_monostatic_features_h5,
    load_cooperative_frame_energy,
    resolve_cooperative_features_h5,
    session_train_val_split_by_region,
)
from isac_imp.record_target_metadata import is_inner_target_xy_m

CSV_COLUMNS = (
    "sample_idx",
    "session_index",
    "frame_index",
    "true_x_m",
    "true_y_m",
    "est_x_m",
    "est_y_m",
    "rmse_xy_m",
)

DEFAULT_VAL_RATIO = 0.2
DEFAULT_VAL_SEED = 42
DEFAULT_XY_MAX_M = 1.0
DEFAULT_OUTLIER_ENERGY_EPS = 1e-8
DEFAULT_OUTLIER_ENERGY_MAD_Z = 5.0


def _default_h5_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "experiment"
        / "cooperative_monostatic_measurement0"
        / "cooperative_monostatic_dataset.h5"
    )


def _default_checkpoint() -> Path:
    return PROJECT_ROOT / "models" / "cooperative_monostatic_cnn" / "best_model.pth"


def _default_output_csv() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "cnn_rmse.csv"


def _default_output_heatmap() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "cnn_rmse_heatmap.png"


def _default_output_cdf() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "cnn_rmse_cdf.png"


def _load_plot_heatmap_module():
    plot_path = Path(__file__).resolve().with_name(
        "plot_cooperative_monostatic_music_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"range-roi 须满足 min < max，收到 {lo} {hi}")
    return lo, hi


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cooperative monostatic CNN localization RMSE from HDF5"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=_default_h5_path(),
        help="input cooperative monostatic HDF5 dataset (raw or features)",
    )
    parser.add_argument(
        "--features-h5",
        type=Path,
        default=None,
        help="预计算 features sidecar（默认按 --h5-path / ROI / checkpoint feature_mode 自动查找）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint(),
        help="CooperativeMonostaticCNN checkpoint path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=_default_output_csv(),
        help="output CSV path for per-sample metrics",
    )
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=list(DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
        help="range ROI in meters (default: 0 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="inference batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit number of samples for debugging",
    )
    parser.add_argument(
        "--session-index",
        type=int,
        default=None,
        help="evaluate only one session index",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="evaluate only validation frames (session split, ratio=0.2, seed=42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="validation session ratio when --val-only (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_VAL_SEED,
        help="random seed for session split when --val-only (default: 42)",
    )
    parser.add_argument(
        "--aggregate-session",
        action="store_true",
        help="average CNN xy predictions over frames per session before RMSE",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip heatmap/CDF PNG outputs after evaluation",
    )
    parser.add_argument(
        "--plot-heatmap",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-heatmap",
        type=Path,
        default=_default_output_heatmap(),
        help="output heatmap PNG path (default: under out/cooperative_monostatic/)",
    )
    parser.add_argument(
        "--plot-cdf",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-cdf",
        type=Path,
        default=_default_output_cdf(),
        help="output CDF PNG path (default: under out/cooperative_monostatic/)",
    )
    parser.add_argument(
        "--filter-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="硬过滤 + session 能量 MAD 软剔除（默认开启；--no-filter-outliers 关闭）",
    )
    parser.add_argument(
        "--xy-max-m",
        type=float,
        default=DEFAULT_XY_MAX_M,
        help=f"硬过滤：|x|,|y| 超过该值视为越界 (m)，默认 {DEFAULT_XY_MAX_M}",
    )
    parser.add_argument(
        "--outlier-energy-eps",
        type=float,
        default=DEFAULT_OUTLIER_ENERGY_EPS,
        help="硬过滤：任一站 CPI 平均幅度 <= eps 视为近零",
    )
    parser.add_argument(
        "--outlier-energy-mad-z",
        type=float,
        default=DEFAULT_OUTLIER_ENERGY_MAD_Z,
        help=f"软剔除：session 内能量 MAD z-score 阈值，默认 {DEFAULT_OUTLIER_ENERGY_MAD_Z}",
    )
    return parser.parse_args()


def _resolve_frame_indices(
    h5_path: Path,
    *,
    max_samples: int | None,
    session_index: int | None,
    val_only: bool,
    val_ratio: float,
    seed: int,
) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        if DATASET_KEY_FEATURES in f:
            total = int(f[DATASET_KEY_FEATURES].shape[0])
        else:
            total = int(f[DATASET_KEY_PROFILES_DEV0].shape[0])
        session_arr = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        target_position = np.asarray(
            f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
        )

    indices = np.arange(total, dtype=np.int64)
    if val_only:
        _, val_idx, _ = session_train_val_split_by_region(
            session_arr,
            target_position,
            val_ratio,
            seed=seed,
        )
        indices = val_idx

    if session_index is not None:
        mask = session_arr[indices] == int(session_index)
        indices = indices[mask]

    indices_list = [int(i) for i in indices]
    if max_samples is not None:
        indices_list = indices_list[: max_samples]
    return indices_list


def _apply_eval_outlier_filters(
    h5_path: Path,
    frame_indices: list[int],
    *,
    xy_max_m: float = DEFAULT_XY_MAX_M,
    energy_eps: float = DEFAULT_OUTLIER_ENERGY_EPS,
    energy_mad_z: float = DEFAULT_OUTLIER_ENERGY_MAD_Z,
) -> list[int]:
    """对评估候选帧做硬过滤 + session 能量 MAD 软剔除（与训练口径一致）。"""
    cand = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    n_before = int(cand.size)
    if n_before == 0:
        return []

    with h5py.File(h5_path, "r") as f:
        n_frames = (
            int(f[DATASET_KEY_FEATURES].shape[0])
            if DATASET_KEY_FEATURES in f
            else int(f[DATASET_KEY_PROFILES_DEV0].shape[0])
        )
        session_indices = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        target_position = np.asarray(
            f[DATASET_KEY_TARGET_POSITION][:], dtype=np.float64
        )

    if session_indices.shape[0] != n_frames or target_position.shape[0] != n_frames:
        raise ValueError(
            "session_index / target_position 与帧数不一致: "
            f"{session_indices.shape[0]}, {target_position.shape[0]} vs {n_frames}"
        )
    if np.any(cand < 0) or np.any(cand >= n_frames):
        raise ValueError("frame_indices 超出 HDF5 帧范围")

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
        with h5py.File(h5_path, "r") as f:
            profiles_dev0 = np.asarray(f[DATASET_KEY_PROFILES_DEV0][:n_frames])
            profiles_dev1 = np.asarray(f[DATASET_KEY_PROFILES_DEV1][:n_frames])
        energy = cooperative_frame_cpi_energy(profiles_dev0, profiles_dev1)
        keep_hard, drop_counts = filter_cooperative_frames_hard(
            target_position[:, :2],
            profiles_dev0=profiles_dev0,
            profiles_dev1=profiles_dev1,
            xy_max_m=xy_max_m,
            energy_eps=energy_eps,
        )

    hard_set = {int(i) for i in keep_hard}
    after_hard = np.asarray([i for i in cand if int(i) in hard_set], dtype=np.int64)
    hard_dropped_from_cand = n_before - int(after_hard.size)

    soft_dropped = 0
    kept = after_hard
    if after_hard.size > 0:
        kept, soft_dropped = filter_cooperative_frames_energy_mad(
            after_hard,
            session_indices,
            energy,
            z_thresh=energy_mad_z,
        )

    print(
        f"Outlier filter [eval]: hard dropped {hard_dropped_from_cand} from candidates "
        f"(nan_label={drop_counts['nan_label']}, oob_xy={drop_counts['oob_xy']}, "
        f"nan_cpi={drop_counts['nan_cpi']}, near_zero={drop_counts['near_zero']}); "
        f"soft dropped {soft_dropped}; kept {kept.size} / {n_before}",
        flush=True,
    )
    if kept.size == 0:
        raise RuntimeError("outlier 过滤后评估集为空")
    return [int(i) for i in kept]

def _predict_xy_batch(
    model: torch.nn.Module,
    device: torch.device | str,
    dual_profiles: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        pred = model(dual_profiles.to(device))
    return pred.detach().cpu().numpy()


def _load_checkpoint_inference_config(
    checkpoint: Path,
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_mode = str(ckpt.get("feature_mode", "legacy_4ch"))
    norm_means: np.ndarray | None = None
    norm_stds: np.ndarray | None = None
    norm_stats_path = ckpt.get("norm_stats_path")
    if norm_stats_path is not None:
        stats_path = Path(norm_stats_path)
        if not stats_path.is_file():
            stats_path = checkpoint.parent / norm_stats_path
        if stats_path.is_file():
            norm_means, norm_stds, stats_mode = load_cooperative_norm_stats(stats_path)
            if stats_mode != feature_mode and feature_mode == "logmag_fixed_norm":
                pass
    return feature_mode, norm_means, norm_stds


def _evaluate_per_frame_from_features(
    h5_path: Path,
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    frame_indices: list[int],
    batch_size: int,
    show_progress: bool,
) -> list[dict[str, float | int]]:
    """从预计算 features sidecar 评估（跳过 CPI→ROI FFT）。"""
    rows: list[dict[str, float | int]] = []
    if not frame_indices:
        return rows

    with h5py.File(h5_path, "r") as f:
        feat_ds = f[DATASET_KEY_FEATURES]
        target_ds = f[DATASET_KEY_TARGET_POSITION]
        session_ds = f[DATASET_KEY_SESSION_INDEX]
        frame_ds = f[DATASET_KEY_FRAME_INDEX]

        batch_bar = tqdm(
            range(0, len(frame_indices), batch_size),
            desc="CNN RMSE",
            unit="batch",
            disable=not show_progress,
        )
        for start in batch_bar:
            chunk = frame_indices[start : start + batch_size]
            feats = np.stack(
                [np.asarray(feat_ds[i], dtype=np.float32) for i in chunk],
                axis=0,
            )
            model_input = torch.from_numpy(feats)
            pred_xy = _predict_xy_batch(model, device, model_input)

            for i, sample_idx in enumerate(chunk):
                true_x = float(target_ds[sample_idx, 0])
                true_y = float(target_ds[sample_idx, 1])
                est_x = float(pred_xy[i, 0])
                est_y = float(pred_xy[i, 1])
                rmse = position_rmse_xy((est_x, est_y), (true_x, true_y))
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "session_index": int(session_ds[sample_idx]),
                        "frame_index": int(frame_ds[sample_idx]),
                        "true_x_m": true_x,
                        "true_y_m": true_y,
                        "est_x_m": est_x,
                        "est_y_m": est_y,
                        "rmse_xy_m": rmse,
                    }
                )
    return rows


def _evaluate_per_frame(
    h5_path: Path,
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
    frame_indices: list[int],
    batch_size: int,
    show_progress: bool,
    feature_mode: str = "legacy_4ch",
    norm_means: np.ndarray | None = None,
    norm_stds: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    if is_cooperative_monostatic_features_h5(h5_path):
        return _evaluate_per_frame_from_features(
            h5_path,
            model,
            device,
            frame_indices=frame_indices,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    rows: list[dict[str, float | int]] = []
    if not frame_indices:
        return rows

    with h5py.File(h5_path, "r") as f:
        dev0_ds = f[DATASET_KEY_PROFILES_DEV0]
        dev1_ds = f[DATASET_KEY_PROFILES_DEV1]
        target_ds = f[DATASET_KEY_TARGET_POSITION]
        session_ds = f[DATASET_KEY_SESSION_INDEX]
        frame_ds = f[DATASET_KEY_FRAME_INDEX]

        batch_bar = tqdm(
            range(0, len(frame_indices), batch_size),
            desc="CNN RMSE",
            unit="batch",
            disable=not show_progress,
        )
        for start in batch_bar:
            chunk = frame_indices[start : start + batch_size]
            dual_list: list[np.ndarray] = []
            meta: list[tuple[int, int, int, float, float]] = []

            for sample_idx in chunk:
                divide_dev0 = dev0_ds[sample_idx]
                divide_dev1 = dev1_ds[sample_idx]
                if cooperative_uses_slowtime_input(feature_mode):  # type: ignore[arg-type]
                    dual_arr = divide_cpi_dual_to_roi_range_slowtime_np(
                        divide_dev0,
                        divide_dev1,
                        proc_params=proc_params,
                        range_roi=range_roi,
                    )
                    dual_list.append(dual_arr)
                else:
                    roi0, roi1 = divide_cpi_dual_to_roi_range_profiles_np(
                        divide_dev0,
                        divide_dev1,
                        proc_params=proc_params,
                        range_roi=range_roi,
                    )
                    dual_list.append(np.stack([roi0, roi1], axis=0))
                true_x = float(target_ds[sample_idx, 0])
                true_y = float(target_ds[sample_idx, 1])
                meta.append(
                    (
                        sample_idx,
                        int(session_ds[sample_idx]),
                        int(frame_ds[sample_idx]),
                        true_x,
                        true_y,
                    )
                )

            dual_np = np.stack(dual_list, axis=0).astype(np.complex64, copy=False)
            dual_tensor = torch.from_numpy(dual_np)
            if cooperative_uses_slowtime_input(feature_mode):  # type: ignore[arg-type]
                model_input = dual_slowtime_to_model_input(
                    dual_tensor,
                    mode=feature_mode,  # type: ignore[arg-type]
                )
            else:
                model_input = dual_roi_to_model_input(
                    dual_tensor,
                    mode=feature_mode,  # type: ignore[arg-type]
                    norm_means=norm_means,
                    norm_stds=norm_stds,
                )
            pred_xy = _predict_xy_batch(model, device, model_input)

            for i, (sample_idx, sess, frame_idx, true_x, true_y) in enumerate(meta):
                est_x = float(pred_xy[i, 0])
                est_y = float(pred_xy[i, 1])
                rmse = position_rmse_xy((est_x, est_y), (true_x, true_y))
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "session_index": sess,
                        "frame_index": frame_idx,
                        "true_x_m": true_x,
                        "true_y_m": true_y,
                        "est_x_m": est_x,
                        "est_y_m": est_y,
                        "rmse_xy_m": rmse,
                    }
                )
    return rows


def _evaluate_aggregate_session(
    per_frame_rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    session_buckets: dict[int, dict[str, list[float | int]]] = defaultdict(
        lambda: {
            "est_x": [],
            "est_y": [],
            "true_x": [],
            "true_y": [],
            "frame_index": [],
        }
    )
    for row in per_frame_rows:
        sess = int(row["session_index"])
        bucket = session_buckets[sess]
        bucket["est_x"].append(float(row["est_x_m"]))
        bucket["est_y"].append(float(row["est_y_m"]))
        bucket["true_x"].append(float(row["true_x_m"]))
        bucket["true_y"].append(float(row["true_y_m"]))
        bucket["frame_index"].append(int(row["frame_index"]))

    rows: list[dict[str, float | int]] = []
    for sess in sorted(session_buckets):
        bucket = session_buckets[sess]
        true_x = float(np.mean(bucket["true_x"]))
        true_y = float(np.mean(bucket["true_y"]))
        est_x = float(np.nanmean(bucket["est_x"]))
        est_y = float(np.nanmean(bucket["est_y"]))
        rmse = position_rmse_xy((est_x, est_y), (true_x, true_y))
        rows.append(
            {
                "sample_idx": sess,
                "session_index": sess,
                "frame_index": int(np.mean(bucket["frame_index"])),
                "true_x_m": true_x,
                "true_y_m": true_y,
                "est_x_m": est_x,
                "est_y_m": est_y,
                "rmse_xy_m": rmse,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rmse_stats(rmses: np.ndarray) -> dict[str, float | int]:
    """计算 RMSE 数组的样本数与 mean/std/median。"""
    valid = rmses[np.isfinite(rmses)]
    stats: dict[str, float | int] = {
        "samples": int(rmses.size),
        "valid": int(valid.size),
        "nan": int(rmses.size - valid.size),
    }
    if valid.size:
        stats["mean"] = float(valid.mean())
        stats["std"] = float(valid.std())
        stats["median"] = float(np.median(valid))
    return stats


def _stats_table_row(
    region: str,
    stats: dict[str, float | int],
) -> list[str | int | float]:
    """将 ``_rmse_stats`` 结果转为 tabulate 行。"""
    if stats.get("valid", 0):
        return [
            region,
            stats["samples"],
            stats["valid"],
            stats["nan"],
            stats["mean"],
            stats["std"],
            stats["median"],
        ]
    return [
        region,
        stats["samples"],
        stats["valid"],
        stats["nan"],
        "-",
        "-",
        "-",
    ]


def _print_summary(rows: list[dict[str, float | int]]) -> None:
    rmses = np.asarray([row["rmse_xy_m"] for row in rows], dtype=np.float64)
    inner_mask = np.array(
        [
            is_inner_target_xy_m(float(row["true_x_m"]), float(row["true_y_m"]))
            for row in rows
        ],
        dtype=bool,
    )
    headers = [
        "Region",
        "Samples",
        "Valid",
        "NaN",
        "Mean (m)",
        "Std (m)",
        "Median (m)",
    ]
    table_rows = [
        _stats_table_row("global", _rmse_stats(rmses)),
        _stats_table_row(
            "inner (|x|,|y| <= 0.5 m)",
            _rmse_stats(rmses[inner_mask]),
        ),
        _stats_table_row("outer", _rmse_stats(rmses[~inner_mask])),
    ]
    print("\nCNN localization RMSE summary:")
    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="simple_grid",
            floatfmt=".4f",
        )
    )


def main() -> None:
    args = argument_parser()
    raw_h5_path = args.h5_path.resolve()
    if not raw_h5_path.is_file():
        raise FileNotFoundError(raw_h5_path)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    range_roi = _parse_range_roi(list(args.range_roi))
    proc_params = grc_cooperative_processing_params()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size = max(1, int(args.batch_size))

    model = load_cooperative_monostatic_cnn_checkpoint(checkpoint, device)
    feature_mode, norm_means, norm_stds = _load_checkpoint_inference_config(checkpoint)

    h5_path = resolve_cooperative_features_h5(
        raw_h5_path,
        range_roi=range_roi,
        feature_mode=feature_mode,
        features_h5=args.features_h5,
        require=False,
    )
    if h5_path != raw_h5_path:
        print(f"Using features sidecar: {h5_path}", flush=True)

    frame_indices = _resolve_frame_indices(
        h5_path,
        max_samples=args.max_samples,
        session_index=args.session_index,
        val_only=args.val_only,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    n_before_filter = len(frame_indices)
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
        frame_indices = _apply_eval_outlier_filters(
            h5_path,
            frame_indices,
            xy_max_m=float(args.xy_max_m),
            energy_eps=float(args.outlier_energy_eps),
            energy_mad_z=float(args.outlier_energy_mad_z),
        )

    split_label = (
        f"val-only region-stratified (9 regions, ratio={args.val_ratio}, seed={args.seed})"
        if args.val_only
        else "all frames"
    )
    if args.filter_outliers:
        frames_msg = (
            f"{n_before_filter} -> {len(frame_indices)} after outlier filter "
            f"({split_label})"
        )
    else:
        frames_msg = f"{len(frame_indices)} ({split_label})"
    print(
        f"HDF5: {h5_path}\n"
        f"Checkpoint: {checkpoint}\n"
        f"Feature mode: {feature_mode}\n"
        f"Frames: {frames_msg} | "
        f"ROI {range_roi[0]:.1f}–{range_roi[1]:.1f} m | "
        f"batch_size={batch_size} | device={device}"
    )

    per_frame_rows = _evaluate_per_frame(
        h5_path,
        model,
        device,
        proc_params=proc_params,
        range_roi=range_roi,
        frame_indices=frame_indices,
        batch_size=batch_size,
        show_progress=not args.no_progress,
        feature_mode=feature_mode,
        norm_means=norm_means,
        norm_stds=norm_stds,
    )

    if args.aggregate_session:
        rows = _evaluate_aggregate_session(per_frame_rows)
    else:
        rows = per_frame_rows

    output_csv = args.output_csv.resolve()
    _write_csv(output_csv, rows)
    print(f"output csv: {output_csv}")
    _print_summary(rows)

    if not args.no_plot:
        plot_mod = _load_plot_heatmap_module()
        plot_mod.plot_rmse_heatmap_combined_from_csv(
            output_csv,
            args.output_heatmap.resolve(),
        )
        print(f"output heatmap: {args.output_heatmap.resolve()}")
        plot_mod.plot_rmse_cdf_from_csv(
            output_csv,
            args.output_cdf.resolve(),
            title="CNN localization RMSE CDF",
        )
        print(f"output cdf: {args.output_cdf.resolve()}")


if __name__ == "__main__":
    main()
