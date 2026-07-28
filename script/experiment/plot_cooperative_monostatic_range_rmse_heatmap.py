#!/usr/bin/env python3
"""Cooperative monostatic MUSIC/ESPRIT 单站距离 MAE 热力图：dev0 / dev1 双子图。

支持从 CSV（``run_cooperative_monostatic_*_rmse.py`` 输出）或 HDF5 直接评估并出图。

示例::

    python script/experiment/plot_cooperative_monostatic_range_rmse_heatmap.py \\
        --input-csv out/cooperative_monostatic/music_rmse.csv \\
        --method music

    python script/experiment/plot_cooperative_monostatic_range_rmse_heatmap.py \\
        --h5-path data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5 \\
        --method music
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap

from isac import PROJECT_ROOT
from isac.sensing.detection.cfar import CFARDetector
from isac_imp.cooperative_monostatic_range_calibration import (
    RangeBiasCalibResult,
    add_range_bias_calib_arguments,
    dataframe_for_range_mae,
    format_calib_summary,
    resolve_range_bias_calibration,
    true_monostatic_range_m,
)
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_CFAR_DETECTOR,
    DEFAULT_CFAR_GUARD,
    DEFAULT_CFAR_PFA,
    DEFAULT_CFAR_TRAILING,
    DEFAULT_CFAR_TYPE,
    DEFAULT_ESPRIT_NUM_SOURCES,
    DEFAULT_ESPRIT_SUBARRAY_SIZE,
    DEFAULT_ESPRIT_WINDOW_SIZE,
    DEFAULT_RANGE_ROI,
    default_range_cfar_detector,
    grc_cooperative_processing_params,
)

MethodName = Literal["music", "esprit"]

ABS_ERR_COL_DEV0 = "abs_err_r_dev0_m"
ABS_ERR_COL_DEV1 = "abs_err_r_dev1_m"

RANGE_EVAL_COLUMNS = (
    "sample_idx",
    "session_index",
    "frame_index",
    "true_x_m",
    "true_y_m",
    "r_dev0_m",
    "r_dev1_m",
)


@dataclass(frozen=True)
class RangeEvalOptions:
    """H5 距离估计评估选项（与 RMSE 评估脚本对齐）。"""

    dev0_xy: tuple[float, float] = (0.0, -2.0)
    dev1_xy: tuple[float, float] = (-2.0, 0.0)
    aggregate_session: bool = False
    max_samples: int | None = None
    session_index: int | None = None
    show_progress: bool = True
    enable_cfar: bool = False
    cfar_type: str = DEFAULT_CFAR_TYPE
    cfar_guard: int = DEFAULT_CFAR_GUARD
    cfar_trailing: int = DEFAULT_CFAR_TRAILING
    cfar_pfa: float = DEFAULT_CFAR_PFA
    cfar_detector: str = DEFAULT_CFAR_DETECTOR
    cfar_k: int | None = None
    cfar_offset: float | None = None
    esprit_num_sources: int = DEFAULT_ESPRIT_NUM_SOURCES
    esprit_subarray_size: int = DEFAULT_ESPRIT_SUBARRAY_SIZE
    esprit_window_size: int = DEFAULT_ESPRIT_WINDOW_SIZE


def _load_heatmap_module():
    plot_path = Path(__file__).resolve().with_name(
        "plot_cooperative_monostatic_music_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_module(method: MethodName):
    script_name = (
        "run_cooperative_monostatic_music_rmse.py"
        if method == "music"
        else "run_cooperative_monostatic_esprit_rmse.py"
    )
    eval_path = Path(__file__).resolve().with_name(script_name)
    module_name = f"range_eval_{method}"
    spec = importlib.util.spec_from_file_location(module_name, eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eval module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_range_csv_columns(df: pd.DataFrame) -> None:
    required = ("true_x_m", "true_y_m", "r_dev0_m", "r_dev1_m")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(missing)}")


def add_per_dev_range_abs_errors(
    df: pd.DataFrame,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
) -> pd.DataFrame:
    """为每条样本增加 ``abs_err_r_dev0_m`` / ``abs_err_r_dev1_m``。"""
    _validate_range_csv_columns(df)
    out = df.copy()
    true_r0 = out.apply(
        lambda row: true_monostatic_range_m(
            (float(row["true_x_m"]), float(row["true_y_m"])),
            dev0_xy,
        ),
        axis=1,
    )
    true_r1 = out.apply(
        lambda row: true_monostatic_range_m(
            (float(row["true_x_m"]), float(row["true_y_m"])),
            dev1_xy,
        ),
        axis=1,
    )
    err0 = np.abs(out["r_dev0_m"].to_numpy(dtype=np.float64) - true_r0.to_numpy())
    err1 = np.abs(out["r_dev1_m"].to_numpy(dtype=np.float64) - true_r1.to_numpy())
    err0[~np.isfinite(err0)] = np.nan
    err1[~np.isfinite(err1)] = np.nan
    out[ABS_ERR_COL_DEV0] = err0
    out[ABS_ERR_COL_DEV1] = err1
    return out


def _build_cfar_detector_from_options(options: RangeEvalOptions) -> CFARDetector | None:
    if not options.enable_cfar:
        return None
    cfar_type = str(options.cfar_type).strip().lower()
    if cfar_type == "os" and options.cfar_k is None:
        raise ValueError("--cfar-k is required when --cfar-type os")
    return default_range_cfar_detector(
        cfar_type=cfar_type,
        guard=int(options.cfar_guard),
        trailing=int(options.cfar_trailing),
        pfa=float(options.cfar_pfa),
        detector=str(options.cfar_detector),
        k=int(options.cfar_k) if options.cfar_k is not None else None,
        offset=float(options.cfar_offset) if options.cfar_offset is not None else None,
    )


def _rows_to_range_dataframe(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("evaluation produced no rows")
    df = pd.DataFrame(rows)
    missing = [c for c in RANGE_EVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"evaluation rows missing columns: {', '.join(missing)}")
    return df[list(RANGE_EVAL_COLUMNS)].copy()


def evaluate_range_estimates_to_dataframe(
    h5_path: Path,
    *,
    method: MethodName,
    options: RangeEvalOptions | None = None,
) -> pd.DataFrame:
    """在 HDF5 上运行 MUSIC/ESPRIT 单站距离估计，返回 per-sample DataFrame。"""
    opts = options or RangeEvalOptions()
    h5_path = h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    eval_mod = _load_eval_module(method)
    proc_params = grc_cooperative_processing_params()
    range_roi = DEFAULT_RANGE_ROI

    if method == "esprit":
        proc_params["esprit_num_sources"] = int(opts.esprit_num_sources)
        proc_params["esprit_subarray_size"] = int(opts.esprit_subarray_size)
        proc_params["esprit_window_size"] = int(opts.esprit_window_size)

    cfar_detector = None
    if method == "music":
        cfar_detector = _build_cfar_detector_from_options(opts)

    eval_kwargs = dict(
        dev0_xy=opts.dev0_xy,
        dev1_xy=opts.dev1_xy,
        proc_params=proc_params,
        range_roi=range_roi,
        max_samples=opts.max_samples,
        session_index=opts.session_index,
        show_progress=opts.show_progress,
    )
    if method == "music":
        if opts.aggregate_session:
            rows = eval_mod._evaluate_aggregate_session(
                h5_path,
                cfar_detector=cfar_detector,
                **eval_kwargs,
            )
        else:
            rows = eval_mod._evaluate_per_frame(
                h5_path,
                cfar_detector=cfar_detector,
                **eval_kwargs,
            )
    elif opts.aggregate_session:
        rows = eval_mod._evaluate_aggregate_session(h5_path, **eval_kwargs)
    else:
        rows = eval_mod._evaluate_per_frame(h5_path, **eval_kwargs)

    return _rows_to_range_dataframe(rows)


def _method_title_prefix(method: MethodName) -> str:
    label = "MUSIC" if method == "music" else "ESPRIT"
    return f"Cooperative Monostatic {label} Range MAE"


def _default_h5_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "experiment"
        / "cooperative_monostatic"
        / "cooperative_monostatic_dataset.h5"
    )


def _default_input_csv(method: MethodName) -> Path:
    name = "music_rmse.csv" if method == "music" else "esprit_rmse.csv"
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / name


def _default_output_png(method: MethodName, *, h5_path: Path | None = None) -> Path:
    if h5_path is not None:
        tag = h5_path.parent.name
        name = (
            f"{method}_range_mae_heatmap_dev_{tag}.png"
            if method == "music"
            else f"esprit_range_mae_heatmap_dev_{tag}.png"
        )
    else:
        name = (
            "music_range_mae_heatmap_dev.png"
            if method == "music"
            else "esprit_range_mae_heatmap_dev.png"
        )
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / name


def plot_range_mae_heatmap_dual_dev_from_df(
    df: pd.DataFrame,
    output_png: Path,
    *,
    method: MethodName = "music",
    dev0_xy: tuple[float, float] = (0.0, -2.0),
    dev1_xy: tuple[float, float] = (-2.0, 0.0),
    cmap: str | Colormap = "RdYlGn_r",
    dpi: int = 150,
    show: bool = False,
    data_source: str | None = None,
    calib_result: RangeBiasCalibResult | None = None,
) -> dict[str, float | int | str]:
    """从含 ``r_dev0_m`` / ``r_dev1_m`` 的 DataFrame 绘制 dev0/dev1 距离 MAE 1×2 热力图。"""
    heatmap_mod = _load_heatmap_module()
    work = add_per_dev_range_abs_errors(df, dev0_xy=dev0_xy, dev1_xy=dev1_xy)

    xs, ys, z0, interp0 = heatmap_mod.build_rmse_grid_10cm_interpolated(
        work, value_col=ABS_ERR_COL_DEV0
    )
    _, _, z1, interp1 = heatmap_mod.build_rmse_grid_10cm_interpolated(
        work, value_col=ABS_ERR_COL_DEV1
    )

    x_edges = heatmap_mod._uniform_axis_edges(xs, heatmap_mod.UNIFIED_GRID_STEP_M)
    y_edges = heatmap_mod._uniform_axis_edges(ys, heatmap_mod.UNIFIED_GRID_STEP_M)
    cmap_obj = heatmap_mod._cmap_with_bad(cmap)

    finite_vals = np.concatenate([z0[np.isfinite(z0)], z1[np.isfinite(z1)]])
    if finite_vals.size == 0:
        src = data_source or "input data"
        raise ValueError(f"no finite range MAE values in {src}")
    vmin = float(finite_vals.min())
    vmax = float(finite_vals.max())

    mean0 = float(z0[np.isfinite(z0)].mean()) if np.isfinite(z0).any() else float("nan")
    mean1 = float(z1[np.isfinite(z1)].mean()) if np.isfinite(z1).any() else float("nan")
    filled0 = int(np.isfinite(z0).sum())
    filled1 = int(np.isfinite(z1).sum())

    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    mesh0 = ax0.pcolormesh(
        x_edges,
        y_edges,
        z0,
        cmap=cmap_obj,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.pcolormesh(
        x_edges,
        y_edges,
        z1,
        cmap=cmap_obj,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )

    for ax, dev_xy, marker, dev_label in (
        (ax0, dev0_xy, "^", "dev0"),
        (ax1, dev1_xy, "s", "dev1"),
    ):
        ax.scatter(
            [dev_xy[0]],
            [dev_xy[1]],
            marker=marker,
            s=80 if marker == "^" else 70,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            label=f"{dev_label} ({dev_xy[0]:.1f}, {dev_xy[1]:.1f}) m",
            zorder=3,
        )
        heatmap_mod._apply_axis_ticks(ax, xs, ys)
        ax.set_xlabel("Target x (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=9)

    ax0.set_ylabel("Target y (m)")
    mean0_s = f"{mean0:.3f}" if np.isfinite(mean0) else "nan"
    mean1_s = f"{mean1:.3f}" if np.isfinite(mean1) else "nan"
    ax0.set_title(f"dev0 Range MAE\ncells={filled0}, mean MAE={mean0_s} m")
    ax1.set_title(f"dev1 Range MAE\ncells={filled1}, mean MAE={mean1_s} m")

    fig.subplots_adjust(right=0.88)
    cbar = fig.colorbar(mesh0, ax=[ax0, ax1], fraction=0.035, pad=0.02)
    cbar.set_label("MAE (m)")

    title_prefix = _method_title_prefix(method)
    source_line = f"\nsource: {data_source}" if data_source else ""
    calib_line = ""
    if calib_result is not None:
        calib_line = (
            f"\nbias_dev0={calib_result.bias_dev0_m:.3f} m, "
            f"bias_dev1={calib_result.bias_dev1_m:.3f} m"
        )
    fig.suptitle(
        f"{title_prefix} (uniform 10 cm grid, outer interpolated){source_line}{calib_line}\n"
        f"dev0 mean={mean0_s} m, dev1 mean={mean1_s} m, "
        f"interpolated cells dev0/dev1={interp0}/{interp1}",
        y=0.98,
        fontsize=11,
    )
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "total_rows": int(len(work)),
        "filled_cells_dev0": filled0,
        "filled_cells_dev1": filled1,
        "mean_mae_dev0_m": mean0,
        "mean_mae_dev1_m": mean1,
        "interpolated_cells_dev0": int(interp0),
        "interpolated_cells_dev1": int(interp1),
        "output_png": str(output_png),
    }


def plot_range_mae_heatmap_dual_dev_from_csv(
    input_csv: Path,
    output_png: Path,
    *,
    method: MethodName = "music",
    dev0_xy: tuple[float, float] = (0.0, -2.0),
    dev1_xy: tuple[float, float] = (-2.0, 0.0),
    cmap: str | Colormap = "RdYlGn_r",
    dpi: int = 150,
    show: bool = False,
) -> dict[str, float | int | str]:
    """读取 RMSE CSV，绘制 dev0/dev1 距离 MAE 1×2 热力图并保存 PNG。"""
    input_csv = input_csv.resolve()
    df = pd.read_csv(input_csv)
    return plot_range_mae_heatmap_dual_dev_from_df(
        df,
        output_png,
        method=method,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        cmap=cmap,
        dpi=dpi,
        show=show,
        data_source=str(input_csv),
    )


def _range_eval_options_from_args(args: argparse.Namespace) -> RangeEvalOptions:
    return RangeEvalOptions(
        dev0_xy=(float(args.dev0_xy[0]), float(args.dev0_xy[1])),
        dev1_xy=(float(args.dev1_xy[0]), float(args.dev1_xy[1])),
        aggregate_session=bool(args.aggregate_session),
        max_samples=args.max_samples,
        session_index=args.session_index,
        show_progress=not args.no_progress,
        enable_cfar=bool(args.enable_cfar),
        cfar_type=str(args.cfar_type),
        cfar_guard=int(args.cfar_guard),
        cfar_trailing=int(args.cfar_trailing),
        cfar_pfa=float(args.cfar_pfa),
        cfar_detector=str(args.cfar_detector),
        cfar_k=args.cfar_k,
        cfar_offset=args.cfar_offset,
        esprit_num_sources=int(args.esprit_num_sources),
        esprit_subarray_size=int(args.esprit_subarray_size),
        esprit_window_size=int(args.esprit_window_size),
    )


def _add_cfar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-cfar",
        action="store_true",
        help="apply 1D CFAR before MUSIC peak selection (--method music)",
    )
    parser.add_argument(
        "--cfar-type",
        type=str,
        default=DEFAULT_CFAR_TYPE,
        choices=("ca", "os"),
        help="CFAR type (default: ca)",
    )
    parser.add_argument(
        "--cfar-guard",
        type=int,
        default=DEFAULT_CFAR_GUARD,
        help="CFAR guard cells (default: 2)",
    )
    parser.add_argument(
        "--cfar-trailing",
        type=int,
        default=DEFAULT_CFAR_TRAILING,
        help="CFAR trailing/reference cells (default: 4)",
    )
    parser.add_argument(
        "--cfar-pfa",
        type=float,
        default=DEFAULT_CFAR_PFA,
        help="CFAR false-alarm rate (default: 1e-4)",
    )
    parser.add_argument(
        "--cfar-detector",
        type=str,
        default=DEFAULT_CFAR_DETECTOR,
        choices=("linear", "squarelaw"),
        help="CFAR detector domain (default: linear)",
    )
    parser.add_argument(
        "--cfar-k",
        type=int,
        default=None,
        help="OS-CFAR rank k (required when --cfar-type os)",
    )
    parser.add_argument(
        "--cfar-offset",
        type=float,
        default=None,
        help="manual CFAR threshold scale (<1 looser, >1 stricter)",
    )


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cooperative monostatic MUSIC/ESPRIT per-device range MAE heatmap "
            "(dev0 + dev1 subplots); input CSV or HDF5"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="input CSV from run_cooperative_monostatic_*_rmse.py",
    )
    source.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="input cooperative monostatic HDF5; evaluate ranges then plot",
    )
    parser.add_argument(
        "--write-csv",
        type=Path,
        default=None,
        help="when using --h5-path, optionally write evaluation rows to CSV",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="output dual-subplot heatmap PNG path",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("music", "esprit"),
        default="music",
        help="estimator for --h5-path evaluation and plot titles (default: music)",
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
        help="limit H5 evaluation samples for debugging",
    )
    parser.add_argument(
        "--session-index",
        type=int,
        default=None,
        help="evaluate only one session index when using --h5-path",
    )
    parser.add_argument(
        "--aggregate-session",
        action="store_true",
        help="average ranges over frames per session before plotting",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm during H5 evaluation",
    )
    _add_cfar_arguments(parser)
    add_range_bias_calib_arguments(parser)
    parser.add_argument(
        "--esprit-num-sources",
        type=int,
        default=DEFAULT_ESPRIT_NUM_SOURCES,
        help=f"ESPRIT num sources (--method esprit, default: {DEFAULT_ESPRIT_NUM_SOURCES})",
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
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdYlGn_r",
        help="matplotlib colormap name (default: RdYlGn_r, green=good low MAE)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="output PNG DPI (default: 150)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show plot window after saving",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    if args.calibrate_range and args.calib_json is not None:
        raise ValueError("--calibrate-range 与 --calib-json 不能同时使用")

    method: MethodName = args.method  # type: ignore[assignment]
    dev0_xy = (float(args.dev0_xy[0]), float(args.dev0_xy[1]))
    dev1_xy = (float(args.dev1_xy[0]), float(args.dev1_xy[1]))
    plot_kwargs = {
        "method": method,
        "dev0_xy": dev0_xy,
        "dev1_xy": dev1_xy,
        "cmap": args.cmap,
        "dpi": int(args.dpi),
        "show": bool(args.show),
    }

    data_source: str
    if args.h5_path is not None:
        h5_path = args.h5_path.resolve()
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        eval_options = _range_eval_options_from_args(args)
        df = evaluate_range_estimates_to_dataframe(
            h5_path,
            method=method,
            options=eval_options,
        )
        data_source = str(h5_path)
        output_png = (
            args.output_png.resolve()
            if args.output_png is not None
            else _default_output_png(method, h5_path=h5_path)
        )
    else:
        input_csv = (
            args.input_csv.resolve()
            if args.input_csv is not None
            else _default_input_csv(method)
        )
        if not input_csv.is_file():
            raise FileNotFoundError(input_csv)
        df = pd.read_csv(input_csv)
        if "r_dev0_m" not in df.columns and "r_dev0_cal_m" in df.columns:
            df["r_dev0_m"] = df["r_dev0_cal_m"]
            df["r_dev1_m"] = df["r_dev1_cal_m"]
        data_source = str(input_csv)
        output_png = (
            args.output_png.resolve()
            if args.output_png is not None
            else _default_output_png(method)
        )

    df_cal, calib_result = resolve_range_bias_calibration(
        df,
        args,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
    )
    if args.write_csv is not None and args.h5_path is not None:
        write_csv = args.write_csv.resolve()
        write_csv.parent.mkdir(parents=True, exist_ok=True)
        df_cal.to_csv(write_csv, index=False)
        print(f"output csv: {write_csv}")

    plot_df = dataframe_for_range_mae(df_cal)
    summary = plot_range_mae_heatmap_dual_dev_from_df(
        plot_df,
        output_png,
        data_source=data_source,
        calib_result=calib_result,
        **plot_kwargs,
    )

    print(
        f"output heatmap: {summary['output_png']} "
        f"(dev0 mean MAE={summary['mean_mae_dev0_m']:.3f} m, "
        f"dev1 mean MAE={summary['mean_mae_dev1_m']:.3f} m)"
    )
    if calib_result is not None:
        print(f"calibration: {format_calib_summary(calib_result)}")


if __name__ == "__main__":
    main()
