#!/usr/bin/env python3
"""Cooperative monostatic HDF5 数据集：MUSIC 双站定位 RMSE 评估。

复现 GRC 接收链中 divide CPI → CPI 复数距离谱 → 1D MUSIC → TX/RX 椭圆交会。

示例::

    python script/experiment/run_cooperative_monostatic_music_rmse.py \\
        --h5-path data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.sensing.detection.cfar import CFARDetector
from isac.sensing.localization import position_rmse_xy
from isac_imp.cfar_eval_cli import (
    add_cfar_arguments,
    build_cfar_detector_from_args,
    print_cfar_config,
)
from isac_imp.eval_timing import (
    run_algo_core_timed,
    timing_json_path_for_csv,
    write_eval_timing_json,
)
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_DEV0_RX_XY,
    DEFAULT_DEV0_TX_XY,
    DEFAULT_DEV0_XY,
    DEFAULT_DEV1_RX_XY,
    DEFAULT_DEV1_TX_XY,
    DEFAULT_DEV1_XY,
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
    localize_xy_from_two_ranges_with_bias,
    music_range_from_divide_cpi,
)
from isac_imp.cooperative_monostatic_range_calibration import (
    add_range_bias_calib_arguments,
    correct_monostatic_range_pair,
    resolve_and_apply_eval_row_calibration,
    resolve_loaded_range_biases,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_POSITION,
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
    "r_dev0_m",
    "r_dev1_m",
    "r_dev0_cal_m",
    "r_dev1_cal_m",
    "rmse_xy_m",
)


def _default_h5_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "experiment"
        # / "cooperative_monostatic_measurement0"
        / "cooperative_monostatic"
        / "cooperative_monostatic_dataset.h5"
    )


def _default_output_csv() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse.csv"


def _default_output_heatmap() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse_heatmap.png"


def _default_output_cdf() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse_cdf.png"


def _default_output_range_heatmap() -> Path:
    return (
        PROJECT_ROOT
        / "out"
        / "cooperative_monostatic"
        / "music_range_mae_heatmap_dev.png"
    )


def _default_output_range_cdf_dev0() -> Path:
    return (
        PROJECT_ROOT
        / "out"
        / "cooperative_monostatic"
        / "music_range_bs0_cdf.png"
    )


def _default_output_range_cdf_dev1() -> Path:
    return (
        PROJECT_ROOT
        / "out"
        / "cooperative_monostatic"
        / "music_range_bs1_cdf.png"
    )


def _default_output_scatter() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_xy_scatter.png"


def _load_plot_heatmap_module():
    plot_path = Path(__file__).resolve().with_name(
        "plot_cooperative_monostatic_music_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_plot_range_heatmap_module():
    plot_path = Path(__file__).resolve().with_name(
        "plot_cooperative_monostatic_range_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_range_mae_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load range heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cooperative monostatic MUSIC localization RMSE from HDF5"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=_default_h5_path(),
        help="input cooperative monostatic HDF5 dataset",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=_default_output_csv(),
        help="output CSV path for per-sample metrics",
    )
    parser.add_argument(
        "--dev0-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV0_XY,
        metavar=("X", "Y"),
        help="dev0 midpoint (heatmap / legacy); default 0 -2",
    )
    parser.add_argument(
        "--dev1-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV1_XY,
        metavar=("X", "Y"),
        help="dev1 midpoint (heatmap / legacy); default -2 0",
    )
    parser.add_argument(
        "--dev0-tx-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV0_TX_XY,
        metavar=("X", "Y"),
        help="dev0 TX antenna xy (default: from DEFAULT_DEV0_TX_XY)",
    )
    parser.add_argument(
        "--dev0-rx-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV0_RX_XY,
        metavar=("X", "Y"),
        help="dev0 RX antenna xy (default: from DEFAULT_DEV0_RX_XY)",
    )
    parser.add_argument(
        "--dev1-tx-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV1_TX_XY,
        metavar=("X", "Y"),
        help="dev1 TX antenna xy (default: from DEFAULT_DEV1_TX_XY)",
    )
    parser.add_argument(
        "--dev1-rx-xy",
        type=float,
        nargs=2,
        default=DEFAULT_DEV1_RX_XY,
        metavar=("X", "Y"),
        help="dev1 RX antenna xy (default: from DEFAULT_DEV1_RX_XY)",
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
        "--aggregate-session",
        action="store_true",
        help="average MUSIC ranges over frames per session before localization",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar",
    )
    parser.add_argument(
        "--plot-heatmap",
        action="store_true",
        help="after evaluation, plot RMSE heatmap by target position",
    )
    parser.add_argument(
        "--output-heatmap",
        type=Path,
        default=_default_output_heatmap(),
        help="output heatmap PNG when --plot-heatmap is set",
    )
    parser.add_argument(
        "--plot-cdf",
        action="store_true",
        help="after evaluation, plot RMSE CDF (global / inner / outer)",
    )
    parser.add_argument(
        "--output-cdf",
        type=Path,
        default=_default_output_cdf(),
        help="output CDF PNG when --plot-cdf is set",
    )
    parser.add_argument(
        "--plot-range-heatmap",
        action="store_true",
        help="after evaluation, plot per-device range MAE heatmap (dev0 + dev1)",
    )
    parser.add_argument(
        "--output-range-heatmap",
        type=Path,
        default=_default_output_range_heatmap(),
        help="output range MAE heatmap PNG when --plot-range-heatmap is set",
    )
    parser.add_argument(
        "--plot-range-cdf",
        action="store_true",
        help="after evaluation, plot per-device range absolute-error CDF (dev0 + dev1)",
    )
    parser.add_argument(
        "--output-range-cdf-dev0",
        type=Path,
        default=_default_output_range_cdf_dev0(),
        help="output range CDF PNG for dev0 when --plot-range-cdf is set",
    )
    parser.add_argument(
        "--output-range-cdf-dev1",
        type=Path,
        default=_default_output_range_cdf_dev1(),
        help="output range CDF PNG for dev1 when --plot-range-cdf is set",
    )
    parser.add_argument(
        "--plot-scatter",
        action="store_true",
        help="after evaluation, plot estimated xy scatter colored by RMSE",
    )
    parser.add_argument(
        "--output-scatter",
        type=Path,
        default=_default_output_scatter(),
        help="output xy scatter PNG when --plot-scatter is set",
    )
    add_cfar_arguments(parser, method_label="MUSIC peak selection")
    add_range_bias_calib_arguments(parser)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="compute device for MUSIC (default: cuda:0; use cpu for NumPy path)",
    )
    return parser.parse_args()


def _music_range_from_divide_cpi(
    divide_cpi: np.ndarray,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
    cfar_detector: CFARDetector | None = None,
    device: str | None = None,
) -> float:
    """Thin wrapper：divide CPI → 固定 ROI 谱 → MUSIC（兼容 tx_offset_sweep）。"""
    return music_range_from_divide_cpi(
        divide_cpi,
        proc_params=proc_params,
        range_roi=range_roi,
        cfar_detector=cfar_detector,
        device=device,
    )


def _xy_pair(values: tuple[float, float] | list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _antenna_kwargs_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "tx0_xy": _xy_pair(args.dev0_tx_xy),
        "rx0_xy": _xy_pair(args.dev0_rx_xy),
        "tx1_xy": _xy_pair(args.dev1_tx_xy),
        "rx1_xy": _xy_pair(args.dev1_rx_xy),
    }


def _localize_sample(
    r0_m: float,
    r1_m: float,
    true_xy: tuple[float, float],
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    bias_dev0_m: float = 0.0,
    bias_dev1_m: float = 0.0,
    tx0_xy: tuple[float, float] = DEFAULT_DEV0_TX_XY,
    rx0_xy: tuple[float, float] = DEFAULT_DEV0_RX_XY,
    tx1_xy: tuple[float, float] = DEFAULT_DEV1_TX_XY,
    rx1_xy: tuple[float, float] = DEFAULT_DEV1_RX_XY,
) -> tuple[float, float, float]:
    if not np.isfinite(r0_m) or not np.isfinite(r1_m):
        return float("nan"), float("nan"), float("nan")
    try:
        est_x, est_y = localize_xy_from_two_ranges_with_bias(
            dev0_xy,
            r0_m,
            dev1_xy,
            r1_m,
            bias_dev0_m=bias_dev0_m,
            bias_dev1_m=bias_dev1_m,
            tx0_xy=tx0_xy,
            rx0_xy=rx0_xy,
            tx1_xy=tx1_xy,
            rx1_xy=rx1_xy,
        )
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    rmse = position_rmse_xy((est_x, est_y), true_xy)
    return est_x, est_y, rmse


def _range_bias_tuple(
    range_biases: tuple[float, float] | None,
) -> tuple[float, float]:
    if range_biases is None:
        return 0.0, 0.0
    return float(range_biases[0]), float(range_biases[1])


def _build_eval_row(
    *,
    sample_idx: int,
    session_index: int,
    frame_index: int,
    true_x: float,
    true_y: float,
    r0: float,
    r1: float,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    range_biases: tuple[float, float] | None = None,
    tx0_xy: tuple[float, float] = DEFAULT_DEV0_TX_XY,
    rx0_xy: tuple[float, float] = DEFAULT_DEV0_RX_XY,
    tx1_xy: tuple[float, float] = DEFAULT_DEV1_TX_XY,
    rx1_xy: tuple[float, float] = DEFAULT_DEV1_RX_XY,
) -> dict[str, float | int]:
    bias_dev0_m, bias_dev1_m = _range_bias_tuple(range_biases)
    true_xy = (true_x, true_y)
    r0_cal, r1_cal = correct_monostatic_range_pair(
        r0,
        r1,
        bias_dev0_m=bias_dev0_m,
        bias_dev1_m=bias_dev1_m,
    )
    est_x, est_y, rmse = _localize_sample(
        r0,
        r1,
        true_xy,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        bias_dev0_m=bias_dev0_m,
        bias_dev1_m=bias_dev1_m,
        tx0_xy=tx0_xy,
        rx0_xy=rx0_xy,
        tx1_xy=tx1_xy,
        rx1_xy=rx1_xy,
    )
    return {
        "sample_idx": int(sample_idx),
        "session_index": int(session_index),
        "frame_index": int(frame_index),
        "true_x_m": true_x,
        "true_y_m": true_y,
        "est_x_m": est_x,
        "est_y_m": est_y,
        "r_dev0_m": r0,
        "r_dev1_m": r1,
        "r_dev0_cal_m": r0_cal,
        "r_dev1_cal_m": r1_cal,
        "rmse_xy_m": rmse,
    }


def _evaluate_per_frame(
    h5_path: Path,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    proc_params: dict,
    range_roi: tuple[float, float],
    max_samples: int | None,
    session_index: int | None,
    show_progress: bool,
    cfar_detector: CFARDetector | None = None,
    range_biases: tuple[float, float] | None = None,
    antenna_kwargs: dict[str, tuple[float, float]] | None = None,
    device: str = "cuda:0",
) -> tuple[list[dict[str, float | int]], float, int]:
    """预加载 CPI 后按算法核口径计时；返回 ``(rows, eval_s, n_timed)``。"""
    import torch

    ant = antenna_kwargs or {}
    with h5py.File(h5_path, "r") as f:
        dev0_ds = f[DATASET_KEY_PROFILES_DEV0]
        dev1_ds = f[DATASET_KEY_PROFILES_DEV1]
        target_ds = f[DATASET_KEY_TARGET_POSITION]
        session_ds = f[DATASET_KEY_SESSION_INDEX]
        frame_ds = f[DATASET_KEY_FRAME_INDEX]
        total = int(dev0_ds.shape[0])

        indices = list(range(total))
        if session_index is not None:
            session_arr = session_ds[:]
            indices = [i for i in indices if int(session_arr[i]) == session_index]
        if max_samples is not None:
            indices = indices[: max_samples]

        # preload (not timed)
        meta: list[tuple[int, int, int, float, float]] = []
        cpi0_list: list = []
        cpi1_list: list = []
        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        torch_device = torch.device(device if use_cuda else "cpu")
        algo_device = str(torch_device) if use_cuda else "cpu"

        for sample_idx in tqdm(
            indices,
            desc="MUSIC preload",
            unit="frame",
            disable=not show_progress,
        ):
            d0 = np.asarray(dev0_ds[sample_idx], dtype=np.complex64)
            d1 = np.asarray(dev1_ds[sample_idx], dtype=np.complex64)
            if use_cuda:
                cpi0_list.append(torch.as_tensor(d0, device=torch_device))
                cpi1_list.append(torch.as_tensor(d1, device=torch_device))
            else:
                cpi0_list.append(d0)
                cpi1_list.append(d1)
            true_x, true_y = (float(v) for v in target_ds[sample_idx, :2])
            meta.append(
                (
                    int(sample_idx),
                    int(session_ds[sample_idx]),
                    int(frame_ds[sample_idx]),
                    true_x,
                    true_y,
                )
            )

    n = len(meta)
    rows_slot: list[dict[str, float | int] | None] = [None] * n

    def run_one(i: int) -> None:
        sample_idx, sess, frame_i, true_x, true_y = meta[i]
        r0 = _music_range_from_divide_cpi(
            cpi0_list[i],
            proc_params=proc_params,
            range_roi=range_roi,
            cfar_detector=cfar_detector,
            device=algo_device if use_cuda else None,
        )
        r1 = _music_range_from_divide_cpi(
            cpi1_list[i],
            proc_params=proc_params,
            range_roi=range_roi,
            cfar_detector=cfar_detector,
            device=algo_device if use_cuda else None,
        )
        rows_slot[i] = _build_eval_row(
            sample_idx=sample_idx,
            session_index=sess,
            frame_index=frame_i,
            true_x=true_x,
            true_y=true_y,
            r0=r0,
            r1=r1,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
            range_biases=range_biases,
            **ant,
        )

    eval_s, n_timed = run_algo_core_timed(
        n,
        device=torch_device if use_cuda else None,
        run_one=run_one,
    )
    rows = [r for r in rows_slot if r is not None]
    return rows, eval_s, n_timed


def _evaluate_aggregate_session(
    h5_path: Path,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    proc_params: dict,
    range_roi: tuple[float, float],
    max_samples: int | None,
    session_index: int | None,
    show_progress: bool,
    cfar_detector: CFARDetector | None = None,
    range_biases: tuple[float, float] | None = None,
    antenna_kwargs: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, float | int]]:
    ant = antenna_kwargs or {}
    session_ranges: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"r0": [], "r1": [], "true_x": [], "true_y": [], "frame_index": []}
    )

    with h5py.File(h5_path, "r") as f:
        dev0_ds = f[DATASET_KEY_PROFILES_DEV0]
        dev1_ds = f[DATASET_KEY_PROFILES_DEV1]
        target_ds = f[DATASET_KEY_TARGET_POSITION]
        session_ds = f[DATASET_KEY_SESSION_INDEX]
        frame_ds = f[DATASET_KEY_FRAME_INDEX]
        total = int(dev0_ds.shape[0])

        indices = range(total)
        if session_index is not None:
            session_arr = session_ds[:]
            indices = [i for i in indices if int(session_arr[i]) == session_index]
        if max_samples is not None:
            indices = list(indices)[: max_samples]

        iterator = tqdm(
            indices,
            desc="MUSIC ranges",
            unit="frame",
            disable=not show_progress,
        )
        for sample_idx in iterator:
            sess = int(session_ds[sample_idx])
            r0 = _music_range_from_divide_cpi(
                dev0_ds[sample_idx],
                proc_params=proc_params,
                range_roi=range_roi,
                cfar_detector=cfar_detector,
            )
            r1 = _music_range_from_divide_cpi(
                dev1_ds[sample_idx],
                proc_params=proc_params,
                range_roi=range_roi,
                cfar_detector=cfar_detector,
            )
            bucket = session_ranges[sess]
            bucket["r0"].append(r0)
            bucket["r1"].append(r1)
            bucket["true_x"].append(float(target_ds[sample_idx, 0]))
            bucket["true_y"].append(float(target_ds[sample_idx, 1]))
            bucket["frame_index"].append(int(frame_ds[sample_idx]))

    rows: list[dict[str, float | int]] = []
    for sess in sorted(session_ranges):
        bucket = session_ranges[sess]
        true_x = float(np.mean(bucket["true_x"]))
        true_y = float(np.mean(bucket["true_y"]))
        r0 = float(np.nanmean(bucket["r0"]))
        r1 = float(np.nanmean(bucket["r1"]))
        rows.append(
            _build_eval_row(
                sample_idx=sess,
                session_index=sess,
                frame_index=int(np.mean(bucket["frame_index"])),
                true_x=true_x,
                true_y=true_y,
                r0=r0,
                r1=r1,
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
                range_biases=range_biases,
                **ant,
            )
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
    print("\nMUSIC localization mean error summary:")
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
    if args.calibrate_range and args.calib_json is not None:
        raise ValueError("--calibrate-range 与 --calib-json 不能同时使用")

    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    proc_params = grc_cooperative_processing_params()
    range_roi = DEFAULT_RANGE_ROI
    dev0_xy = _xy_pair(args.dev0_xy)
    dev1_xy = _xy_pair(args.dev1_xy)
    antenna_kwargs = _antenna_kwargs_from_args(args)
    localize_fn = partial(_localize_sample, **antenna_kwargs)
    range_biases = resolve_loaded_range_biases(args)
    cfar_detector = build_cfar_detector_from_args(args)
    proc_params["cfar_enabled"] = cfar_detector is not None
    print_cfar_config(cfar_detector)

    t0 = time.perf_counter()
    device = str(args.device)
    if args.aggregate_session:
        rows = _evaluate_aggregate_session(
            h5_path,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
            proc_params=proc_params,
            range_roi=range_roi,
            max_samples=args.max_samples,
            session_index=args.session_index,
            show_progress=not args.no_progress,
            cfar_detector=cfar_detector,
            range_biases=range_biases,
            antenna_kwargs=antenna_kwargs,
        )
        eval_s = time.perf_counter() - t0
        n_eval = len(rows)
        timing_device = "cpu"
    else:
        rows, eval_s, n_eval = _evaluate_per_frame(
            h5_path,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
            proc_params=proc_params,
            range_roi=range_roi,
            max_samples=args.max_samples,
            session_index=args.session_index,
            show_progress=not args.no_progress,
            cfar_detector=cfar_detector,
            range_biases=range_biases,
            antenna_kwargs=antenna_kwargs,
            device=device,
        )
        timing_device = device

    resolve_and_apply_eval_row_calibration(
        rows,
        args,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        localize_fn=localize_fn,
        calibration_preapplied=range_biases is not None,
        **antenna_kwargs,
    )

    output_csv = args.output_csv.resolve()
    _write_csv(output_csv, rows)
    print(f"output csv: {output_csv}")
    write_eval_timing_json(
        timing_json_path_for_csv(output_csv),
        method="music",
        eval_s=eval_s,
        n_samples=n_eval,
        device=timing_device,
    )
    _print_summary(rows)

    if (
        args.plot_heatmap
        or args.plot_cdf
        or args.plot_range_heatmap
        or args.plot_range_cdf
        or args.plot_scatter
    ):
        plot_mod = _load_plot_heatmap_module()
        if args.plot_heatmap:
            plot_mod.plot_rmse_heatmap_combined_from_csv(
                output_csv,
                args.output_heatmap.resolve(),
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
            )
            print(f"output heatmap: {args.output_heatmap.resolve()}")
        if args.plot_cdf:
            plot_mod.plot_rmse_cdf_from_csv(
                output_csv,
                args.output_cdf.resolve(),
                title="MUSIC localization mean error CDF",
            )
            print(f"output cdf: {args.output_cdf.resolve()}")
        if args.plot_scatter:
            plot_mod.plot_xy_estimate_scatter_from_csv(
                output_csv,
                args.output_scatter.resolve(),
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
            )
            print(f"output scatter: {args.output_scatter.resolve()}")
        if args.plot_range_heatmap or args.plot_range_cdf:
            range_plot_mod = _load_plot_range_heatmap_module()
            if args.plot_range_heatmap:
                import pandas as pd

                from isac_imp.cooperative_monostatic_range_calibration import (
                    dataframe_for_range_mae,
                )

                plot_df = dataframe_for_range_mae(pd.DataFrame(rows))
                range_plot_mod.plot_range_mae_heatmap_dual_dev_from_df(
                    plot_df,
                    args.output_range_heatmap.resolve(),
                    method="music",
                    dev0_xy=dev0_xy,
                    dev1_xy=dev1_xy,
                )
                print(f"output range heatmap: {args.output_range_heatmap.resolve()}")
            if args.plot_range_cdf:
                summaries = range_plot_mod.plot_range_abs_error_cdf_dual_dev_from_csv(
                    output_csv,
                    method="music",
                    output_dev0=args.output_range_cdf_dev0.resolve(),
                    output_dev1=args.output_range_cdf_dev1.resolve(),
                    tx0_xy=antenna_kwargs["tx0_xy"],
                    rx0_xy=antenna_kwargs["rx0_xy"],
                    tx1_xy=antenna_kwargs["tx1_xy"],
                    rx1_xy=antenna_kwargs["rx1_xy"],
                )
                print(f"output range cdf dev0: {summaries['dev0']['output_png']}")
                print(f"output range cdf dev1: {summaries['dev1']['output_png']}")


if __name__ == "__main__":
    main()
