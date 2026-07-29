#!/usr/bin/env python3
"""两阶段定位评估：RegionCNN → FineCNN 串联 → 全局 (x, y)。

主输出与单阶段 ``run_cooperative_monostatic_cnn_rmse.py`` 对齐（7 列 CSV、
global/inner/outer 汇总、heatmap/CDF/scatter）。Region 诊断写入 sidecar CSV。

示例::

    python script/experiment/run_cooperative_monostatic_two_stage_eval.py \\
        --region-checkpoint models/cooperative_monostatic_region_cnn/best_model.pth \\
        --fine-checkpoint models/cooperative_monostatic_fine_cnn/best_model.pth \\
        --region-topk 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.models import (
    CooperativeMonostaticTwoStageCNN,
    decode_xy_topk_region_probs,
    load_cooperative_monostatic_fine_cnn_checkpoint,
    load_cooperative_monostatic_region_cnn_checkpoint,
    subregion_id_to_one_hot,
)
from isac.models.preprocess import (
    apply_real_imag_rms_norm,
    divide_cpi_dual_to_roi_range_profiles_np,
    dual_roi_to_model_input,
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
    is_cooperative_monostatic_features_h5,
    resolve_cooperative_features_h5,
    maybe_exclude_subregion_corner_frames,
    session_train_val_split_by_subregion,
)
from isac_imp.record_target_metadata import (
    SUBREGION_COUNT,
    target_subregion_index_xy_m,
)

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))
from cooperative_monostatic_eval_report import (  # noqa: E402
    LOCALIZATION_CSV_COLUMNS,
    plot_localization_artifacts,
    print_localization_rmse_summary,
    to_localization_row,
    write_localization_csv,
)

REGION_DIAGNOSTICS_COLUMNS = (
    "sample_idx",
    "session_index",
    "frame_index",
    "true_x_m",
    "true_y_m",
    "true_subregion_id",
    "pred_subregion_id",
    "region_correct",
    "region_topk_hit",
    "topk_ids",
    "topk_probs",
    "est_x_m",
    "est_y_m",
    "est_x_top1_m",
    "est_y_top1_m",
    "est_x_oracle_m",
    "est_y_oracle_m",
    "rmse_xy_m",
    "rmse_xy_top1_m",
    "rmse_xy_oracle_m",
)

DEFAULT_H5 = (
    PROJECT_ROOT
    / "data"
    / "experiment"
    / "cooperative_monostatic_measurement0"
    / "cooperative_monostatic_dataset.h5"
)
DEFAULT_REGION_CKPT = (
    PROJECT_ROOT / "models" / "cooperative_monostatic_region_cnn" / "best_model.pth"
)
DEFAULT_FINE_CKPT = (
    PROJECT_ROOT / "models" / "cooperative_monostatic_fine_cnn" / "best_model.pth"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "cooperative_monostatic" / "two_stage"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage region+fine cooperative monostatic evaluation"
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--region-checkpoint", type=Path, default=DEFAULT_REGION_CKPT)
    parser.add_argument("--fine-checkpoint", type=Path, default=DEFAULT_FINE_CKPT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="评估输出目录（未显式指定 --output-* 时文件落于此）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="主定位 CSV（7 列，与单 CNN 一致；默认 <output-dir>/two_stage_rmse.csv）",
    )
    parser.add_argument(
        "--output-heatmap",
        type=Path,
        default=None,
        help="heatmap PNG（默认 <output-dir>/two_stage_rmse_heatmap.png）",
    )
    parser.add_argument(
        "--output-cdf",
        type=Path,
        default=None,
        help="CDF PNG（默认 <output-dir>/two_stage_rmse_cdf.png）",
    )
    parser.add_argument(
        "--output-scatter",
        type=Path,
        default=None,
        help="xy scatter PNG（默认 <output-dir>/two_stage_xy_scatter.png）",
    )
    parser.add_argument(
        "--output-region-diagnostics",
        type=Path,
        default=None,
        help=(
            "Region 诊断 sidecar CSV（默认 <output-dir>/two_stage_region_diagnostics.csv）"
        ),
    )
    parser.add_argument(
        "--print-region-metrics",
        action="store_true",
        help="额外打印 Region top-1/top-k/oracle/per-class 指标",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-subregion-corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "剔除 4x4 四角子区帧（默认关闭；"
            "--exclude-subregion-corners 开启）"
        ),
    )
    parser.add_argument("--features-h5", type=Path, default=None)
    parser.add_argument(
        "--require-features-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_M", "MAX_M"),
        help="覆盖 checkpoint 中的 ROI（默认读 region ckpt）",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default=None,
        help="覆盖 checkpoint feature_mode（默认读 region ckpt）",
    )
    parser.add_argument("--feature-norm", type=str, default="none")
    parser.add_argument(
        "--region-topk",
        type=int,
        default=3,
        help="Region top-k 指标（默认 3；不参与 xy 融合）",
    )
    parser.add_argument(
        "--session-list",
        type=Path,
        default=None,
        help="仅评估这些 session（每行一个 int）",
    )
    parser.add_argument(
        "--exclude-session-list",
        type=Path,
        default=None,
        help="排除这些 session（每行一个 int）；可与 --session-list 联用",
    )
    return parser.parse_args()


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = args.output_dir.resolve()
    return {
        "csv": (args.output_csv or (out_dir / "two_stage_rmse.csv")).resolve(),
        "heatmap": (
            args.output_heatmap or (out_dir / "two_stage_rmse_heatmap.png")
        ).resolve(),
        "cdf": (args.output_cdf or (out_dir / "two_stage_rmse_cdf.png")).resolve(),
        "scatter": (
            args.output_scatter or (out_dir / "two_stage_xy_scatter.png")
        ).resolve(),
        "diagnostics": (
            args.output_region_diagnostics
            or (out_dir / "two_stage_region_diagnostics.csv")
        ).resolve(),
        "confusion": (out_dir / "region_confusion.npy").resolve(),
        "out_dir": out_dir,
    }


def _read_session_id_file(path: Path) -> set[int]:
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(int(s))
    return ids


def _resolve_frame_indices(
    h5_path: Path,
    *,
    max_samples: int | None,
    val_only: bool,
    val_ratio: float,
    seed: int,
    session_list: Path | None = None,
    exclude_session_list: Path | None = None,
    exclude_corner_subregions: bool = False,
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
        _, val_idx, _ = session_train_val_split_by_subregion(
            session_arr,
            target_position,
            val_ratio,
            seed=seed,
            exclude_corner_subregions=exclude_corner_subregions,
        )
        indices = val_idx

    if session_list is not None:
        want = _read_session_id_file(session_list.resolve())
        if not want:
            raise ValueError(f"--session-list 为空: {session_list}")
        mask = np.asarray(
            [int(session_arr[i]) in want for i in indices.tolist()], dtype=bool
        )
        indices = indices[mask]
    if exclude_session_list is not None:
        ban = _read_session_id_file(exclude_session_list.resolve())
        if ban:
            mask = np.asarray(
                [int(session_arr[i]) not in ban for i in indices.tolist()],
                dtype=bool,
            )
            indices = indices[mask]

    indices = maybe_exclude_subregion_corner_frames(
        indices,
        target_position,
        enabled=exclude_corner_subregions,
        label="eval",
    )

    indices_list = [int(i) for i in indices]
    if max_samples is not None:
        indices_list = indices_list[:max_samples]
    return indices_list


def _load_model_input_batch(
    h5_path: Path,
    chunk: list[int],
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
    feature_mode: str,
    feature_norm: str,
) -> torch.Tensor:
    if is_cooperative_monostatic_features_h5(h5_path):
        with h5py.File(h5_path, "r") as f:
            feats = np.stack(
                [np.asarray(f[DATASET_KEY_FEATURES][i], dtype=np.float32) for i in chunk],
                axis=0,
            )
        model_input = torch.from_numpy(feats)
        if feature_norm == "rms":
            model_input = apply_real_imag_rms_norm(model_input)
        return model_input

    dual_list: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        for sample_idx in chunk:
            roi0, roi1 = divide_cpi_dual_to_roi_range_profiles_np(
                f[DATASET_KEY_PROFILES_DEV0][sample_idx],
                f[DATASET_KEY_PROFILES_DEV1][sample_idx],
                proc_params=proc_params,
                range_roi=range_roi,
            )
            dual_list.append(np.stack([roi0, roi1], axis=0))
    dual = torch.from_numpy(np.stack(dual_list, axis=0))
    return dual_roi_to_model_input(
        dual,
        mode=feature_mode,  # type: ignore[arg-type]
        feature_norm=feature_norm,  # type: ignore[arg-type]
    )


@torch.no_grad()
def _evaluate(
    h5_path: Path,
    two_stage: CooperativeMonostaticTwoStageCNN,
    device: torch.device,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
    frame_indices: list[int],
    batch_size: int,
    show_progress: bool,
    feature_mode: str,
    feature_norm: str,
    region_topk: int = 3,
) -> list[dict[str, float | int | str]]:
    """返回完整诊断行（含 Region 字段）；主 CSV 由调用方投影为 7 列。"""
    rows: list[dict[str, float | int | str]] = []
    if not frame_indices:
        return rows
    topk = int(region_topk)
    if topk < 1:
        raise ValueError(f"region_topk 须 >= 1，收到 {region_topk}")

    with h5py.File(h5_path, "r") as f:
        target_ds = f[DATASET_KEY_TARGET_POSITION]
        session_ds = f[DATASET_KEY_SESSION_INDEX]
        frame_ds = f[DATASET_KEY_FRAME_INDEX]
        metas = [
            (
                int(i),
                int(session_ds[i]),
                int(frame_ds[i]),
                float(target_ds[i, 0]),
                float(target_ds[i, 1]),
            )
            for i in frame_indices
        ]

    batch_bar = tqdm(
        range(0, len(frame_indices), batch_size),
        desc="Two-stage eval",
        unit="batch",
        disable=not show_progress,
    )
    for start in batch_bar:
        chunk = frame_indices[start : start + batch_size]
        meta_chunk = metas[start : start + batch_size]
        model_input = _load_model_input_batch(
            h5_path,
            chunk,
            proc_params=proc_params,
            range_roi=range_roi,
            feature_mode=feature_mode,
            feature_norm=feature_norm,
        ).to(device)

        est_xy, topk_ids, topk_probs = decode_xy_topk_region_probs(
            two_stage, model_input, topk=topk
        )
        top1_ids = topk_ids[:, 0]
        est_xy_top1 = est_xy

        true_sids = torch.tensor(
            [
                target_subregion_index_xy_m(true_x, true_y)
                for _, _, _, true_x, true_y in meta_chunk
            ],
            dtype=torch.int64,
            device=device,
        )
        override = subregion_id_to_one_hot(true_sids, two_stage.num_classes).to(
            device=device, dtype=model_input.dtype
        )
        oracle_xy, _ = two_stage(model_input, region_probs_override=override)

        for i, (sample_idx, sess, frame_i, true_x, true_y) in enumerate(meta_chunk):
            true_sid = int(true_sids[i].item())
            pred_id = int(top1_ids[i].item())
            ids_i = [int(topk_ids[i, j].item()) for j in range(topk)]
            probs_i = [float(topk_probs[i, j].item()) for j in range(topk)]
            est_x = float(est_xy[i, 0].item())
            est_y = float(est_xy[i, 1].item())
            t1x = float(est_xy_top1[i, 0].item())
            t1y = float(est_xy_top1[i, 1].item())
            ox = float(oracle_xy[i, 0].item())
            oy = float(oracle_xy[i, 1].item())
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "session_index": sess,
                    "frame_index": frame_i,
                    "true_x_m": true_x,
                    "true_y_m": true_y,
                    "true_subregion_id": true_sid,
                    "pred_subregion_id": pred_id,
                    "region_correct": int(pred_id == true_sid),
                    "region_topk_hit": int(true_sid in ids_i),
                    "topk_ids": ",".join(str(v) for v in ids_i),
                    "topk_probs": ",".join(f"{v:.6f}" for v in probs_i),
                    "est_x_m": est_x,
                    "est_y_m": est_y,
                    "est_x_top1_m": t1x,
                    "est_y_top1_m": t1y,
                    "est_x_oracle_m": ox,
                    "est_y_oracle_m": oy,
                    "rmse_xy_m": position_rmse_xy((est_x, est_y), (true_x, true_y)),
                    "rmse_xy_top1_m": position_rmse_xy((t1x, t1y), (true_x, true_y)),
                    "rmse_xy_oracle_m": position_rmse_xy((ox, oy), (true_x, true_y)),
                }
            )
    return rows


def _write_region_diagnostics_csv(
    path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REGION_DIAGNOSTICS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in REGION_DIAGNOSTICS_COLUMNS})


def _print_region_diagnostics_summary(
    rows: list[dict[str, float | int | str]],
    *,
    region_topk: int,
) -> None:
    if not rows:
        print("无评估样本")
        return
    n = len(rows)
    top1_acc = sum(int(r["region_correct"]) for r in rows) / n
    topk_hit = sum(int(r["region_topk_hit"]) for r in rows) / n
    rmse = np.asarray([float(r["rmse_xy_m"]) for r in rows], dtype=np.float64)
    rmse_oracle = np.asarray(
        [float(r["rmse_xy_oracle_m"]) for r in rows], dtype=np.float64
    )
    correct_mask = np.asarray(
        [int(r["region_correct"]) for r in rows], dtype=bool
    )
    conf = np.zeros((SUBREGION_COUNT, SUBREGION_COUNT), dtype=np.int64)
    for r in rows:
        conf[int(r["true_subregion_id"]), int(r["pred_subregion_id"])] += 1

    print("\nRegion diagnostics:")
    print(
        tabulate(
            [
                ["N", n],
                ["region_topk", region_topk],
                ["Region top-1 acc", f"{top1_acc:.4f}"],
                [f"Region top-{region_topk} hit", f"{topk_hit:.4f}"],
                ["Oracle-region RMSE mean", f"{rmse_oracle.mean():.4f}"],
                [
                    "RMSE when top-1 correct",
                    f"{rmse[correct_mask].mean():.4f}"
                    if correct_mask.any()
                    else "n/a",
                ],
                [
                    "RMSE when top-1 wrong",
                    f"{rmse[~correct_mask].mean():.4f}"
                    if (~correct_mask).any()
                    else "n/a",
                ],
            ],
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    per_class = []
    for sid in range(SUBREGION_COUNT):
        total = int(conf[sid].sum())
        if total == 0:
            continue
        hit = int(conf[sid, sid])
        per_class.append([sid, total, hit, f"{hit / total:.3f}"])
    if per_class:
        print("\nPer-class top-1 accuracy (nonempty):")
        print(
            tabulate(
                per_class,
                headers=["subregion_id", "n", "correct", "acc"],
                tablefmt="github",
            )
        )


def main() -> None:
    args = _parse_args()
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA 不可用，回退到 CPU", flush=True)

    paths = _resolve_output_paths(args)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    region_ckpt_path = args.region_checkpoint.resolve()
    fine_ckpt_path = args.fine_checkpoint.resolve()
    region_model = load_cooperative_monostatic_region_cnn_checkpoint(
        region_ckpt_path, device
    )
    fine_model = load_cooperative_monostatic_fine_cnn_checkpoint(
        fine_ckpt_path, device
    )
    two_stage = CooperativeMonostaticTwoStageCNN(region_model, fine_model).to(device)
    two_stage.eval()

    region_ckpt = torch.load(region_ckpt_path, map_location="cpu", weights_only=False)
    feature_mode = args.feature_mode or str(region_ckpt.get("feature_mode", "real_imag"))
    feature_norm = str(args.feature_norm)
    if args.range_roi is not None:
        range_roi = (float(args.range_roi[0]), float(args.range_roi[1]))
    elif "range_roi" in region_ckpt:
        rr = region_ckpt["range_roi"]
        range_roi = (float(rr[0]), float(rr[1]))
    else:
        range_roi = DEFAULT_RANGE_ROI

    raw_h5 = args.h5_path.resolve()
    if not raw_h5.is_file():
        raise FileNotFoundError(raw_h5)
    h5_path = resolve_cooperative_features_h5(
        raw_h5,
        range_roi=range_roi,
        feature_mode=feature_mode,
        features_h5=args.features_h5,
        require=bool(args.require_features_h5),
    )
    if h5_path != raw_h5:
        print(f"Using features sidecar: {h5_path}", flush=True)

    frame_indices = _resolve_frame_indices(
        h5_path,
        max_samples=args.max_samples,
        val_only=args.val_only,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        session_list=args.session_list,
        exclude_session_list=args.exclude_session_list,
        exclude_corner_subregions=bool(args.exclude_subregion_corners),
    )
    if not frame_indices:
        raise RuntimeError("过滤 session / 四角子区后无评估帧")
    proc_params = grc_cooperative_processing_params()
    region_topk = int(args.region_topk)
    if region_topk < 1:
        raise ValueError(f"--region-topk 须 >= 1，收到 {args.region_topk}")
    print(
        f"HDF5: {h5_path}\n"
        f"Region ckpt: {region_ckpt_path}\n"
        f"Fine ckpt: {fine_ckpt_path}\n"
        f"Frames: {len(frame_indices)} | ROI {range_roi[0]:.1f}–{range_roi[1]:.1f} m | "
        f"feature_mode={feature_mode} | region_topk={region_topk} | device={device}",
        flush=True,
    )

    rows = _evaluate(
        h5_path,
        two_stage,
        device,
        proc_params=proc_params,
        range_roi=range_roi,
        frame_indices=frame_indices,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        feature_mode=feature_mode,
        feature_norm=feature_norm,
        region_topk=region_topk,
    )

    localization_rows = [to_localization_row(r) for r in rows]
    write_localization_csv(paths["csv"], localization_rows)
    print(f"output csv: {paths['csv']}")
    print_localization_rmse_summary(
        localization_rows,
        title="Two-stage localization mean error summary",
    )

    _write_region_diagnostics_csv(paths["diagnostics"], rows)
    print(f"output region diagnostics: {paths['diagnostics']}")

    conf = np.zeros((SUBREGION_COUNT, SUBREGION_COUNT), dtype=np.int64)
    for r in rows:
        conf[int(r["true_subregion_id"]), int(r["pred_subregion_id"])] += 1
    np.save(paths["confusion"], conf)
    print(f"output confusion: {paths['confusion']}")

    if args.print_region_metrics:
        _print_region_diagnostics_summary(rows, region_topk=region_topk)

    if not args.no_plot and localization_rows:
        plot_localization_artifacts(
            paths["csv"],
            heatmap=paths["heatmap"],
            cdf=paths["cdf"],
            scatter=paths["scatter"],
            cdf_title="Two-stage localization RMSE CDF",
        )


if __name__ == "__main__":
    main()
