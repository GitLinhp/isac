"""Cooperative monostatic 定位评估：共享 CSV / summary / 绘图工具。

单阶段 CNN 与两阶段评估共用主 CSV 列与 global/inner/outer 平均误差汇总格式。
逐样本列 ``rmse_xy_m`` 为平面欧氏误差（历史列名）；汇总指标为该误差的算术平均。
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from tabulate import tabulate

from isac_imp.record_target_metadata import is_inner_target_xy_m

# 与 run_cooperative_monostatic_cnn_rmse.py 主 CSV 一致
# rmse_xy_m：逐样本欧氏误差 ||est−true||（m），非集合 RMSE
LOCALIZATION_CSV_COLUMNS = (
    "sample_idx",
    "session_index",
    "frame_index",
    "true_x_m",
    "true_y_m",
    "est_x_m",
    "est_y_m",
    "rmse_xy_m",
)

# 兼容旧名
CNN_CSV_COLUMNS = LOCALIZATION_CSV_COLUMNS


def load_rmse_plot_module():
    """加载 ``plot_cooperative_monostatic_music_rmse_heatmap`` 绘图模块。"""
    plot_path = Path(__file__).resolve().with_name(
        "plot_cooperative_monostatic_music_rmse_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_rmse_heatmap", plot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heatmap plot module from {plot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_localization_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    columns: tuple[str, ...] = LOCALIZATION_CSV_COLUMNS,
) -> None:
    """写入定位评估主 CSV（默认 7 列）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})


def rmse_stats(rmses: np.ndarray) -> dict[str, float | int]:
    """计算逐样本欧氏误差数组的样本数与 mean/std/median。"""
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


def stats_table_row(
    region: str,
    stats: dict[str, float | int],
) -> list[str | int | float]:
    """将 ``rmse_stats`` 结果转为 tabulate 行。"""
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


def print_localization_rmse_summary(
    rows: list[dict[str, Any]],
    *,
    title: str = "CNN localization mean error summary",
) -> None:
    """打印 global / inner / outer 平均误差汇总表（与单 CNN 格式一致）。"""
    if not rows:
        print("无评估样本")
        return
    errs = np.asarray([float(row["rmse_xy_m"]) for row in rows], dtype=np.float64)
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
        stats_table_row("global", rmse_stats(errs)),
        stats_table_row(
            "inner (|x|,|y| <= 0.5 m)",
            rmse_stats(errs[inner_mask]),
        ),
        stats_table_row("outer", rmse_stats(errs[~inner_mask])),
    ]
    print(f"\n{title}:")
    print(
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="simple_grid",
            floatfmt=".4f",
        )
    )


def plot_localization_artifacts(
    csv_path: Path,
    *,
    heatmap: Path,
    cdf: Path,
    scatter: Path | None = None,
    cdf_title: str = "CNN localization mean error CDF",
) -> None:
    """从定位 CSV 生成 heatmap / CDF /（可选）scatter。"""
    plot_mod = load_rmse_plot_module()
    plot_mod.plot_rmse_heatmap_combined_from_csv(csv_path, heatmap)
    print(f"output heatmap: {heatmap.resolve()}")
    plot_mod.plot_rmse_cdf_from_csv(csv_path, cdf, title=cdf_title)
    print(f"output cdf: {cdf.resolve()}")
    if scatter is not None:
        plot_mod.plot_xy_estimate_scatter_from_csv(csv_path, scatter)
        print(f"output scatter: {scatter.resolve()}")


def to_localization_row(row: dict[str, Any]) -> dict[str, Any]:
    """从完整评估 dict 提取 7 列定位行。"""
    return {k: row[k] for k in LOCALIZATION_CSV_COLUMNS}


def load_two_stage_eval_metrics(
    csv_path: Path,
    *,
    diagnostics_path: Path | None = None,
) -> dict[str, float | int]:
    """从主定位 CSV（7 列）与 Region sidecar 汇总常用指标。

    Region 字段缺失时仅返回平均误差 / n。
    """
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"n": 0}
    n = len(rows)
    err = np.asarray([float(r["rmse_xy_m"]) for r in rows], dtype=np.float64)
    out: dict[str, float | int] = {
        "n": n,
        "global_mean_err_m": float(err.mean()),
        "global_mean_err_median_m": float(np.median(err)),
        "global_mean_err_p90_m": float(np.quantile(err, 0.9)),
    }
    inner: list[float] = []
    outer: list[float] = []
    for r in rows:
        val = float(r["rmse_xy_m"])
        if is_inner_target_xy_m(float(r["true_x_m"]), float(r["true_y_m"])):
            inner.append(val)
        else:
            outer.append(val)
    out["inner_mean_err_m"] = float(np.mean(inner)) if inner else float("nan")
    out["outer_mean_err_m"] = float(np.mean(outer)) if outer else float("nan")

    diag_path = diagnostics_path
    if diag_path is None:
        cand = csv_path.parent / "two_stage_region_diagnostics.csv"
        if cand.is_file():
            diag_path = cand
    if diag_path is not None and diag_path.is_file():
        drows = list(csv.DictReader(diag_path.open(newline="", encoding="utf-8")))
        if drows:
            top1 = sum(int(r["region_correct"]) for r in drows) / len(drows)
            topk_hit = sum(int(r["region_topk_hit"]) for r in drows) / len(drows)
            oracle = np.asarray(
                [float(r["rmse_xy_oracle_m"]) for r in drows], dtype=np.float64
            )
            correct = np.asarray(
                [int(r["region_correct"]) for r in drows], dtype=bool
            )
            d_err = np.asarray(
                [float(r["rmse_xy_m"]) for r in drows], dtype=np.float64
            )
            out["region_top1_acc"] = float(top1)
            out["region_topk_hit"] = float(topk_hit)
            out["oracle_region_mean_err_m"] = float(oracle.mean())
            out["mean_err_when_region_correct_m"] = (
                float(d_err[correct].mean()) if correct.any() else float("nan")
            )
            out["mean_err_when_region_wrong_m"] = (
                float(d_err[~correct].mean()) if (~correct).any() else float("nan")
            )
    return out


__all__ = [
    "CNN_CSV_COLUMNS",
    "LOCALIZATION_CSV_COLUMNS",
    "load_rmse_plot_module",
    "load_two_stage_eval_metrics",
    "plot_localization_artifacts",
    "print_localization_rmse_summary",
    "rmse_stats",
    "stats_table_row",
    "to_localization_row",
    "write_localization_csv",
]
