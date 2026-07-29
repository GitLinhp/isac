#!/usr/bin/env python3
"""一键对照评测 cooperative monostatic MUSIC / ESPRIT / CNN 定位平均误差。

依次调用三个已有评测脚本，汇总 Global / Inner / Outer mean error
（逐样本欧氏距离的算术平均）。

示例::

    python script/experiment/run_cooperative_monostatic_methods_compare.py
    python script/experiment/run_cooperative_monostatic_methods_compare.py \\
        --methods music,cnn --max-samples 256 --skip-plots
    python script/experiment/run_cooperative_monostatic_methods_compare.py \\
        --shared-colorbar
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from isac import PROJECT_ROOT
from isac_imp.eval_timing import read_eval_timing_json, timing_json_path_for_csv
from isac_imp.record_target_metadata import (
    is_inner_target_xy_m,
    is_subregion_corner_xy_m,
)

PYTHON = Path(sys.executable)
EXPERIMENT_DIR = PROJECT_ROOT / "script" / "experiment"
MUSIC_SCRIPT = EXPERIMENT_DIR / "run_cooperative_monostatic_music_rmse.py"
ESPRIT_SCRIPT = EXPERIMENT_DIR / "run_cooperative_monostatic_esprit_rmse.py"
CNN_SCRIPT = EXPERIMENT_DIR / "run_cooperative_monostatic_cnn_rmse.py"

DEFAULT_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models/cnn_improve_next/aug_spec_only/best_model.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out/cooperative_monostatic/methods_compare"
DEFAULT_CNN_RANGE_ROI = (0.0, 4.0)
VALID_METHODS = ("music", "esprit", "cnn")
CFAR_COMPARE_METHODS = ("music", "esprit")
SUMMARY_FIELDS = (
    "method",
    "global_mean_err_m",
    "inner_mean_err_m",
    "outer_mean_err_m",
    "n_samples",
    "eval_s",
    "mean_ms_per_sample",
    "device",
    "csv_path",
)
DELTA_FIELDS = (
    "method",
    "delta_global_mean_err_m",
    "delta_inner_mean_err_m",
    "delta_outer_mean_err_m",
    "n_off",
    "n_cfar",
)


def _parse_methods(raw: str) -> list[str]:
    parts = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--methods 不能为空")
    unknown = [m for m in parts if m not in VALID_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--methods 仅支持 {VALID_METHODS}，收到未知项 {unknown}"
        )
    # preserve order, unique
    seen: set[str] = set()
    ordered: list[str] = []
    for m in parts:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("cnn-range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(
            f"cnn-range-roi 须满足 min < max，收到 {lo} {hi}"
        )
    return lo, hi


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot compare cooperative monostatic MUSIC / ESPRIT / CNN "
            "localization mean error on the same HDF5"
        )
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=DEFAULT_H5,
        help="input cooperative monostatic HDF5 (default: Run2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="CNN checkpoint (default: aug_spec_only best_model.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="root output directory for per-method artifacts + summary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="compute device for MUSIC / ESPRIT / CNN (default: cuda:0)",
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=list(VALID_METHODS),
        help="comma-separated subset: music,esprit,cnn (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit samples for smoke tests (passed through to all methods)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="skip heatmap/CDF PNG generation",
    )
    parser.add_argument(
        "--shared-colorbar",
        action="store_true",
        help=(
            "after eval, redraw heatmaps with shared colorbar "
            "vmin=0 and vmax=nanmax across methods (ignored with --skip-plots)"
        ),
    )
    parser.add_argument(
        "--cnn-range-roi",
        type=float,
        nargs=2,
        default=list(DEFAULT_CNN_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
        help="CNN range ROI in meters (default: 0 4)",
    )
    parser.add_argument(
        "--compare-cfar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "for music/esprit, also run CFAR-on variants and write delta summary "
            "(default: off; use --compare-cfar to enable)"
        ),
    )
    parser.add_argument(
        "--exclude-subregion-corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "exclude 4x4 corner subregions (ids 0/3/12/15) when computing "
            "summary mean error and methods CDF compare (default: off; "
            "use --exclude-subregion-corners to enable)"
        ),
    )
    return parser.parse_args()


def _load_plot_heatmap_module():
    plot_path = EXPERIMENT_DIR / "plot_cooperative_monostatic_music_rmse_heatmap.py"
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plot_range_module():
    plot_path = EXPERIMENT_DIR / "plot_cooperative_monostatic_range_rmse_heatmap.py"
    spec = importlib.util.spec_from_file_location("plot_range_mae_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load range plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shared_colorbar_vmax(csv_paths: list[Path], plot_mod) -> float | None:
    maxima: list[float] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        _xs, _ys, z, _interp = plot_mod.build_rmse_grid_10cm_interpolated(df)
        finite = z[np.isfinite(z)]
        if finite.size:
            maxima.append(float(np.nanmax(finite)))
    if not maxima:
        return None
    vmax = float(np.max(maxima))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return None
    return vmax


def _replot_shared_colorbar(
    artifacts: list[tuple[str, Path, Path]],
) -> None:
    """Re-draw heatmaps with vmin=0 and shared vmax across methods."""
    plot_mod = _load_plot_heatmap_module()
    csv_paths = [csv_path for _method, csv_path, _png in artifacts]
    vmax = _shared_colorbar_vmax(csv_paths, plot_mod)
    if vmax is None:
        print(
            "Shared colorbar skipped: no finite grid mean-error values",
            flush=True,
        )
        return
    print(
        f"\n=== Shared colorbar: vmin=0.0 vmax={vmax:.4f} ===",
        flush=True,
    )
    for method, csv_path, heatmap_png in artifacts:
        plot_mod.plot_rmse_heatmap_combined_from_csv(
            csv_path,
            heatmap_png,
            vmin=0.0,
            vmax=vmax,
        )
        print(f"rewrote heatmap ({method}): {heatmap_png}", flush=True)


_CDF_COMPARE_REGIONS = ("global", "inner", "outer", "no_corner")


def _plot_methods_cdf_compare(
    summary_rows: list[dict[str, str | float | int]],
    output_dir: Path,
) -> None:
    """Draw MUSIC / ESPRIT / CNN mean-error CDF for each region subset."""
    baseline_labels = ("music", "esprit", "cnn")
    series: list[tuple[str, Path]] = []
    for label in baseline_labels:
        for row in summary_rows:
            if str(row["method"]) != label:
                continue
            csv_path = Path(str(row["csv_path"]))
            if csv_path.is_file():
                series.append((label, csv_path))
            break
    if len(series) < 2:
        print(
            "CDF compare skipped: need >=2 of music/esprit/cnn CSVs",
            flush=True,
        )
        return

    plot_mod = _load_plot_heatmap_module()
    for region in _CDF_COMPARE_REGIONS:
        output_png = output_dir / f"methods_rmse_cdf_compare_{region}.png"
        plot_mod.plot_rmse_cdf_compare_from_csvs(
            series, output_png, region=region
        )
        print(f"output cdf compare ({region}): {output_png.resolve()}", flush=True)


def _plot_methods_range_cdf_compare(
    summary_rows: list[dict[str, str | float | int]],
    output_dir: Path,
) -> None:
    """Draw MUSIC / ESPRIT × dev0/dev1 range abs-error CDF on one figure."""
    range_labels = ("music", "esprit")
    series: list[tuple[str, Path, str]] = []
    for label in range_labels:
        csv_path: Path | None = None
        for row in summary_rows:
            if str(row["method"]) != label:
                continue
            candidate = Path(str(row["csv_path"]))
            if candidate.is_file():
                csv_path = candidate
            break
        if csv_path is None:
            continue
        series.append((label, csv_path, "dev0"))
        series.append((label, csv_path, "dev1"))
    if len(series) < 2:
        print(
            "Range CDF compare skipped: need music and/or esprit CSVs",
            flush=True,
        )
        return

    range_mod = _load_plot_range_module()
    output_png = output_dir / "methods_range_cdf_compare.png"
    range_mod.plot_range_abs_error_cdf_compare_from_csvs(series, output_png)
    print(f"output range cdf compare: {output_png.resolve()}", flush=True)


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _mean_err_metrics_from_csv(
    csv_path: Path,
    *,
    exclude_subregion_corners: bool = False,
) -> dict[str, float | int]:
    errs: list[float] = []
    inner: list[float] = []
    outer: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            err = float(row["rmse_xy_m"])
            if not np.isfinite(err):
                continue
            tx = float(row["true_x_m"])
            ty = float(row["true_y_m"])
            if exclude_subregion_corners and is_subregion_corner_xy_m(tx, ty):
                continue
            errs.append(err)
            if is_inner_target_xy_m(tx, ty):
                inner.append(err)
            else:
                outer.append(err)
    return {
        "global_mean_err_m": float(np.mean(errs)) if errs else float("nan"),
        "inner_mean_err_m": float(np.mean(inner)) if inner else float("nan"),
        "outer_mean_err_m": float(np.mean(outer)) if outer else float("nan"),
        "n_samples": len(errs),
    }


def _build_music_or_esprit_cmd(
    *,
    script: Path,
    h5_path: Path,
    out_dir: Path,
    method: str,
    max_samples: int | None,
    skip_plots: bool,
    enable_cfar: bool = False,
    device: str = "cuda:0",
) -> tuple[list[str], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{method}_rmse.csv"
    calib_json = out_dir / f"{method}_range_bias_calib.json"
    cmd = [
        str(PYTHON),
        str(script),
        "--h5-path",
        str(h5_path),
        "--output-csv",
        str(csv_path),
        "--calibrate-range",
        "--write-calib-json",
        str(calib_json),
        "--device",
        str(device),
    ]
    if enable_cfar:
        cmd.append("--enable-cfar")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if not skip_plots:
        cmd.extend(
            [
                "--plot-heatmap",
                "--output-heatmap",
                str(out_dir / f"{method}_rmse_heatmap.png"),
                "--plot-cdf",
                "--output-cdf",
                str(out_dir / f"{method}_rmse_cdf.png"),
                "--plot-range-heatmap",
                "--output-range-heatmap",
                str(out_dir / f"{method}_range_mae_heatmap_dev.png"),
                "--plot-range-cdf",
                "--output-range-cdf-dev0",
                str(out_dir / f"{method}_range_bs0_cdf.png"),
                "--output-range-cdf-dev1",
                str(out_dir / f"{method}_range_bs1_cdf.png"),
                "--plot-scatter",
                "--output-scatter",
                str(out_dir / f"{method}_xy_scatter.png"),
            ]
        )
    return cmd, csv_path


def _build_cnn_cmd(
    *,
    h5_path: Path,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    range_roi: tuple[float, float],
    max_samples: int | None,
    skip_plots: bool,
    exclude_subregion_corners: bool = False,
) -> tuple[list[str], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_rmse.csv"
    cmd = [
        str(PYTHON),
        str(CNN_SCRIPT),
        "--h5-path",
        str(h5_path),
        "--checkpoint",
        str(checkpoint),
        "--range-roi",
        str(range_roi[0]),
        str(range_roi[1]),
        "--output-csv",
        str(csv_path),
        "--output-heatmap",
        str(out_dir / "cnn_rmse_heatmap.png"),
        "--output-cdf",
        str(out_dir / "cnn_rmse_cdf.png"),
        "--output-scatter",
        str(out_dir / "cnn_xy_scatter.png"),
        "--device",
        device,
        "--no-filter-outliers",
        "--batch-size",
        "1",
    ]
    if exclude_subregion_corners:
        cmd.append("--exclude-subregion-corners")
    else:
        cmd.append("--no-exclude-subregion-corners")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if skip_plots:
        cmd.append("--no-plot")
    return cmd, csv_path


def _write_summary(
    rows: list[dict[str, str | float | int]],
    output_dir: Path,
    *,
    exclude_subregion_corners: bool = False,
) -> None:
    csv_path = output_dir / "summary.csv"
    txt_path = output_dir / "summary.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    table = tabulate(
        [
            [
                r["method"],
                float(r["global_mean_err_m"]),
                float(r["inner_mean_err_m"]),
                float(r["outer_mean_err_m"]),
                int(r["n_samples"]),
                float(r["eval_s"]),
                float(r["mean_ms_per_sample"]),
                str(r.get("device", "")),
            ]
            for r in rows
        ],
        headers=[
            "method",
            "global_mean_err",
            "inner_mean_err",
            "outer_mean_err",
            "n",
            "eval_s",
            "mean_ms",
            "device",
        ],
        tablefmt="simple_grid",
        floatfmt=".4f",
    )
    note = ""
    if exclude_subregion_corners:
        note = (
            "Note: n / mean error exclude 4x4 corner subregions "
            "(ids 0/3/12/15); heatmaps remain full-field.\n"
        )
    txt_path.write_text(note + table + "\n", encoding="utf-8")
    print("\n=== Methods compare summary ===", flush=True)
    if note:
        print(note.rstrip(), flush=True)
    print(table, flush=True)
    print(f"\nWrote {csv_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)


def _write_cfar_delta(
    rows: list[dict[str, str | float | int]],
    output_dir: Path,
) -> None:
    by_method = {str(r["method"]): r for r in rows}
    delta_rows: list[dict[str, str | float | int]] = []
    for base in CFAR_COMPARE_METHODS:
        off = by_method.get(base)
        on = by_method.get(f"{base}_cfar")
        if off is None or on is None:
            continue
        delta_rows.append(
            {
                "method": base,
                "delta_global_mean_err_m": float(on["global_mean_err_m"])
                - float(off["global_mean_err_m"]),
                "delta_inner_mean_err_m": float(on["inner_mean_err_m"])
                - float(off["inner_mean_err_m"]),
                "delta_outer_mean_err_m": float(on["outer_mean_err_m"])
                - float(off["outer_mean_err_m"]),
                "n_off": int(off["n_samples"]),
                "n_cfar": int(on["n_samples"]),
            }
        )
    if not delta_rows:
        return

    csv_path = output_dir / "summary_cfar_delta.csv"
    txt_path = output_dir / "summary_cfar_delta.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(delta_rows)

    table = tabulate(
        [
            [
                r["method"],
                float(r["delta_global_mean_err_m"]),
                float(r["delta_inner_mean_err_m"]),
                float(r["delta_outer_mean_err_m"]),
                int(r["n_off"]),
                int(r["n_cfar"]),
            ]
            for r in delta_rows
        ],
        headers=["method", "Δglobal", "Δinner", "Δouter", "n_off", "n_cfar"],
        tablefmt="simple_grid",
        floatfmt=".4f",
    )
    note = "Δ = cfar − off (negative means CFAR better)\n"
    txt_path.write_text(note + table + "\n", encoding="utf-8")
    print("\n=== CFAR delta (cfar − off) ===", flush=True)
    print(note + table, flush=True)
    print(f"\nWrote {csv_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)


def _append_summary_row(
    summary_rows: list[dict[str, str | float | int]],
    *,
    label: str,
    csv_path: Path,
    exclude_subregion_corners: bool = False,
) -> None:
    metrics = _mean_err_metrics_from_csv(
        csv_path,
        exclude_subregion_corners=exclude_subregion_corners,
    )
    timing = read_eval_timing_json(timing_json_path_for_csv(csv_path))
    if timing is None:
        eval_s = float("nan")
        mean_ms = float("nan")
        device = ""
    else:
        eval_s = float(timing["eval_s"])
        mean_ms = float(timing["mean_ms_per_sample"])
        device = str(timing.get("device", ""))
    summary_rows.append(
        {
            "method": label,
            "global_mean_err_m": metrics["global_mean_err_m"],
            "inner_mean_err_m": metrics["inner_mean_err_m"],
            "outer_mean_err_m": metrics["outer_mean_err_m"],
            "n_samples": metrics["n_samples"],
            "eval_s": eval_s,
            "mean_ms_per_sample": mean_ms,
            "device": device,
            "csv_path": str(csv_path),
        }
    )


def main() -> None:
    args = argument_parser()
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 missing: {h5_path}")

    methods: list[str] = list(args.methods)
    if "cnn" in methods:
        ckpt = args.checkpoint.resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"CNN checkpoint missing: {ckpt}")
    else:
        ckpt = args.checkpoint.resolve()

    range_roi = _parse_range_roi(list(args.cnn_range_roi))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_cfar = bool(args.compare_cfar)
    exclude_corners = bool(args.exclude_subregion_corners)

    summary_rows: list[dict[str, str | float | int]] = []
    heatmap_artifacts: list[tuple[str, Path, Path]] = []
    for method in methods:
        if method in CFAR_COMPARE_METHODS:
            script = MUSIC_SCRIPT if method == "music" else ESPRIT_SCRIPT
            variants: list[tuple[str, Path, bool]] = [
                (method, output_dir / method, False),
            ]
            if compare_cfar:
                variants.append(
                    (f"{method}_cfar", output_dir / f"{method}_cfar", True)
                )
            for label, method_dir, enable_cfar in variants:
                cmd, csv_path = _build_music_or_esprit_cmd(
                    script=script,
                    h5_path=h5_path,
                    out_dir=method_dir,
                    method=method,
                    max_samples=args.max_samples,
                    skip_plots=bool(args.skip_plots),
                    enable_cfar=enable_cfar,
                    device=str(args.device),
                )
                heatmap_png = method_dir / f"{method}_rmse_heatmap.png"
                print(f"\n=== {label} ===", flush=True)
                _run(cmd)
                if not csv_path.is_file():
                    raise FileNotFoundError(
                        f"{label} CSV missing after eval: {csv_path}"
                    )
                _append_summary_row(
                    summary_rows,
                    label=label,
                    csv_path=csv_path,
                    exclude_subregion_corners=exclude_corners,
                )
                heatmap_artifacts.append((label, csv_path, heatmap_png))
            continue

        method_dir = output_dir / method
        cmd, csv_path = _build_cnn_cmd(
            h5_path=h5_path,
            checkpoint=ckpt,
            out_dir=method_dir,
            device=str(args.device),
            range_roi=range_roi,
            max_samples=args.max_samples,
            skip_plots=bool(args.skip_plots),
            exclude_subregion_corners=exclude_corners,
        )
        heatmap_png = method_dir / "cnn_rmse_heatmap.png"
        print(f"\n=== {method} ===", flush=True)
        _run(cmd)
        if not csv_path.is_file():
            raise FileNotFoundError(f"{method} CSV missing after eval: {csv_path}")
        _append_summary_row(
            summary_rows,
            label=method,
            csv_path=csv_path,
            exclude_subregion_corners=exclude_corners,
        )
        heatmap_artifacts.append((method, csv_path, heatmap_png))

    if args.shared_colorbar and not args.skip_plots:
        _replot_shared_colorbar(heatmap_artifacts)

    _write_summary(
        summary_rows,
        output_dir,
        exclude_subregion_corners=exclude_corners,
    )
    if compare_cfar:
        _write_cfar_delta(summary_rows, output_dir)
    if not args.skip_plots:
        _plot_methods_cdf_compare(
            summary_rows,
            output_dir,
        )
        _plot_methods_range_cdf_compare(
            summary_rows,
            output_dir,
        )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
