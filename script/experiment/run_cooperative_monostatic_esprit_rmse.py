#!/usr/bin/env python3
"""Cooperative monostatic HDF5 数据集：ESPRIT 双站定位 RMSE 评估。

复现 GRC 接收链中 divide CPI → CPI 复数距离谱 → 1D ESPRIT → 双圆交会。

示例::

    python script/experiment/run_cooperative_monostatic_esprit_rmse.py \\
        --h5-path data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.sensing.localization import position_rmse_xy
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_ESPRIT_NUM_SOURCES,
    DEFAULT_ESPRIT_SUBARRAY_SIZE,
    DEFAULT_ESPRIT_WINDOW_SIZE,
    DEFAULT_RANGE_ROI,
    divide_cpi_to_complex_range_profile,
    estimate_monostatic_range_esprit_m,
    grc_cooperative_processing_params,
    localize_xy_from_two_ranges,
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
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "esprit_rmse.csv"


def _default_output_heatmap() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "esprit_rmse_heatmap.png"


def _default_output_cdf() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "esprit_rmse_cdf.png"


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


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cooperative monostatic ESPRIT localization RMSE from HDF5"
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
        default=(0.0, -2.0),
        metavar=("X", "Y"),
        help="dev0 sensor position in meters (default: 0 -2)",
    )
    parser.add_argument(
        "--dev1-xy",
        type=float,
        nargs=2,
        default=(-2.0, 0.0),
        metavar=("X", "Y"),
        help="dev1 sensor position in meters (default: -2 0)",
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
        help="average ESPRIT ranges over frames per session before localization",
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
        "--esprit-num-sources",
        type=int,
        default=DEFAULT_ESPRIT_NUM_SOURCES,
        help=f"ESPRIT num sources (default: {DEFAULT_ESPRIT_NUM_SOURCES})",
    )
    parser.add_argument(
        "--esprit-subarray-size",
        type=int,
        default=DEFAULT_ESPRIT_SUBARRAY_SIZE,
        help=f"ESPRIT subarray size (default: {DEFAULT_ESPRIT_SUBARRAY_SIZE})",
    )
    parser.add_argument(
        "--esprit-window-size",
        type=int,
        default=DEFAULT_ESPRIT_WINDOW_SIZE,
        help=f"ESPRIT refine window size (default: {DEFAULT_ESPRIT_WINDOW_SIZE})",
    )
    return parser.parse_args()


def _apply_esprit_cli_overrides(proc_params: dict, args: argparse.Namespace) -> None:
    proc_params["esprit_num_sources"] = int(args.esprit_num_sources)
    proc_params["esprit_subarray_size"] = int(args.esprit_subarray_size)
    proc_params["esprit_window_size"] = int(args.esprit_window_size)


def _esprit_range_from_divide_cpi(
    divide_cpi: np.ndarray,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
) -> float:
    profile = divide_cpi_to_complex_range_profile(
        divide_cpi,
        fft_len=proc_params["fft_len"],
        zeropadding_fac=proc_params["zeropadding_fac"],
        transpose_len=proc_params["transpose_len"],
    )
    return estimate_monostatic_range_esprit_m(
        profile,
        range_bin_step=proc_params["range_bin_step"],
        range_roi=range_roi,
        num_sources=proc_params["esprit_num_sources"],
        subarray_size=proc_params["esprit_subarray_size"],
        window_size=proc_params["esprit_window_size"],
    )


def _localize_sample(
    r0_m: float,
    r1_m: float,
    true_xy: tuple[float, float],
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
) -> tuple[float, float, float]:
    if not np.isfinite(r0_m) or not np.isfinite(r1_m):
        return float("nan"), float("nan"), float("nan")
    try:
        est_x, est_y = localize_xy_from_two_ranges(
            dev0_xy,
            r0_m,
            dev1_xy,
            r1_m,
            y_hint=true_xy[1],
        )
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    rmse = position_rmse_xy((est_x, est_y), true_xy)
    return est_x, est_y, rmse


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
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
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
            indices = (i for i in indices if int(session_arr[i]) == session_index)
            indices = list(indices)
        if max_samples is not None:
            indices = list(indices)[: max_samples]

        iterator = tqdm(
            indices,
            desc="ESPRIT RMSE",
            unit="frame",
            disable=not show_progress,
        )
        for sample_idx in iterator:
            divide_dev0 = dev0_ds[sample_idx]
            divide_dev1 = dev1_ds[sample_idx]
            true_x, true_y = (float(v) for v in target_ds[sample_idx, :2])
            true_xy = (true_x, true_y)

            r0 = _esprit_range_from_divide_cpi(
                divide_dev0,
                proc_params=proc_params,
                range_roi=range_roi,
            )
            r1 = _esprit_range_from_divide_cpi(
                divide_dev1,
                proc_params=proc_params,
                range_roi=range_roi,
            )
            est_x, est_y, rmse = _localize_sample(
                r0,
                r1,
                true_xy,
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
            )
            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "session_index": int(session_ds[sample_idx]),
                    "frame_index": int(frame_ds[sample_idx]),
                    "true_x_m": true_x,
                    "true_y_m": true_y,
                    "est_x_m": est_x,
                    "est_y_m": est_y,
                    "r_dev0_m": r0,
                    "r_dev1_m": r1,
                    "rmse_xy_m": rmse,
                }
            )
    return rows


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
) -> list[dict[str, float | int]]:
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
            desc="ESPRIT ranges",
            unit="frame",
            disable=not show_progress,
        )
        for sample_idx in iterator:
            sess = int(session_ds[sample_idx])
            r0 = _esprit_range_from_divide_cpi(
                dev0_ds[sample_idx],
                proc_params=proc_params,
                range_roi=range_roi,
            )
            r1 = _esprit_range_from_divide_cpi(
                dev1_ds[sample_idx],
                proc_params=proc_params,
                range_roi=range_roi,
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
        true_xy = (true_x, true_y)
        r0 = float(np.nanmean(bucket["r0"]))
        r1 = float(np.nanmean(bucket["r1"]))
        est_x, est_y, rmse = _localize_sample(
            r0,
            r1,
            true_xy,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
        )
        rows.append(
            {
                "sample_idx": sess,
                "session_index": sess,
                "frame_index": int(np.mean(bucket["frame_index"])),
                "true_x_m": true_x,
                "true_y_m": true_y,
                "est_x_m": est_x,
                "est_y_m": est_y,
                "r_dev0_m": r0,
                "r_dev1_m": r1,
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
    print("\nESPRIT localization RMSE summary:")
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
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    proc_params = grc_cooperative_processing_params()
    _apply_esprit_cli_overrides(proc_params, args)
    range_roi = DEFAULT_RANGE_ROI
    dev0_xy = (float(args.dev0_xy[0]), float(args.dev0_xy[1]))
    dev1_xy = (float(args.dev1_xy[0]), float(args.dev1_xy[1]))

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
        )
    else:
        rows = _evaluate_per_frame(
            h5_path,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
            proc_params=proc_params,
            range_roi=range_roi,
            max_samples=args.max_samples,
            session_index=args.session_index,
            show_progress=not args.no_progress,
        )

    output_csv = args.output_csv.resolve()
    _write_csv(output_csv, rows)
    print(f"output csv: {output_csv}")
    _print_summary(rows)

    if args.plot_heatmap or args.plot_cdf:
        plot_mod = _load_plot_heatmap_module()
        if args.plot_heatmap:
            plot_mod.plot_rmse_heatmap_combined_from_csv(
                output_csv,
                args.output_heatmap.resolve(),
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
                title_prefix="Cooperative Monostatic ESPRIT Localization RMSE",
            )
            print(f"output heatmap: {args.output_heatmap.resolve()}")
        if args.plot_cdf:
            plot_mod.plot_rmse_cdf_from_csv(
                output_csv,
                args.output_cdf.resolve(),
                title="ESPRIT localization RMSE CDF",
            )
            print(f"output cdf: {args.output_cdf.resolve()}")


if __name__ == "__main__":
    main()
