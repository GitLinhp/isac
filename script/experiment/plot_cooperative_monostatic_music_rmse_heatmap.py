#!/usr/bin/env python3
"""Cooperative monostatic MUSIC RMSE 热力图：按目标位置 (x, y) 聚合绘制。

示例::

    python script/experiment/plot_cooperative_monostatic_music_rmse_heatmap.py \\
        --input-csv out/cooperative_monostatic/music_rmse.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import griddata

from isac import PROJECT_ROOT
from isac_imp.record_target_metadata import INNER_RADIUS_CM, is_inner_target_xy_m

DEFAULT_GRID_MIN_M = -1.0
DEFAULT_GRID_MAX_M = 1.0
OUTER_GRID_STEP_M = 0.2
INNER_GRID_STEP_M = 0.1
UNIFIED_GRID_STEP_M = 0.1
INNER_RADIUS_M = INNER_RADIUS_CM / 100.0
GRID_COORD_DECIMALS = 1
DEFAULT_CMAP = "RdYlGn_r"


def _rounded_linspace_axis(
    min_m: float,
    max_m: float,
    step_m: float,
) -> np.ndarray:
    """生成均匀网格轴并 round，避免 ``pivot.reindex`` 浮点键不匹配。"""
    count = int(round((max_m - min_m) / step_m)) + 1
    axis = np.linspace(min_m, max_m, count, dtype=np.float64)
    return np.round(axis, GRID_COORD_DECIMALS)


def _axis_edges(axis: np.ndarray) -> np.ndarray:
    """由非均匀坐标轴构造 ``pcolormesh`` 边界。"""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size == 1:
        half = 0.1
        return np.array([axis[0] - half, axis[0] + half], dtype=np.float64)
    mids = (axis[:-1] + axis[1:]) / 2.0
    left = axis[0] - (axis[1] - axis[0]) / 2.0
    right = axis[-1] + (axis[-1] - axis[-2]) / 2.0
    return np.concatenate(([left], mids, [right]))


def _default_input_csv() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse.csv"


def _uniform_axis_edges(axis: np.ndarray, step_m: float) -> np.ndarray:
    """均匀网格坐标轴的 ``pcolormesh`` 边界。"""
    axis = np.asarray(axis, dtype=np.float64)
    half = float(step_m) / 2.0
    return np.concatenate([axis - half, [axis[-1] + half]])


def _default_output_outer_png() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse_heatmap_outer.png"


def outer_grid_axis_m(
    *,
    min_m: float = DEFAULT_GRID_MIN_M,
    max_m: float = DEFAULT_GRID_MAX_M,
    step_m: float = OUTER_GRID_STEP_M,
) -> np.ndarray:
    """外侧 20 cm 均匀网格坐标轴 (m)。"""
    return _rounded_linspace_axis(min_m, max_m, step_m)


def inner_grid_axis_m(
    *,
    radius_m: float = INNER_RADIUS_M,
    step_m: float = INNER_GRID_STEP_M,
) -> np.ndarray:
    """内侧 10 cm 均匀网格坐标轴 (m)。"""
    return _rounded_linspace_axis(-radius_m, radius_m, step_m)


def unified_grid_axis_m(
    *,
    min_m: float = DEFAULT_GRID_MIN_M,
    max_m: float = DEFAULT_GRID_MAX_M,
    step_m: float = UNIFIED_GRID_STEP_M,
) -> np.ndarray:
    """全区域 10 cm 均匀网格坐标轴 (m)。"""
    return _rounded_linspace_axis(min_m, max_m, step_m)


def build_rmse_grid(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按目标位置对 ``rmse_xy_m`` 取均值，返回 ``xs, ys, Z``。"""
    if "true_x_m" not in df.columns or "true_y_m" not in df.columns:
        raise ValueError("CSV must contain true_x_m and true_y_m columns")
    if "rmse_xy_m" not in df.columns:
        raise ValueError("CSV must contain rmse_xy_m column")

    xs = np.sort(df["true_x_m"].unique()).astype(np.float64)
    ys = np.sort(df["true_y_m"].unique()).astype(np.float64)
    grouped = (
        df.groupby(["true_y_m", "true_x_m"], as_index=False)["rmse_xy_m"]
        .mean(numeric_only=True)
    )
    pivot = grouped.pivot(index="true_y_m", columns="true_x_m", values="rmse_xy_m")
    pivot = pivot.reindex(index=ys, columns=xs)
    z = pivot.to_numpy(dtype=np.float64)
    return xs, ys, z


def _snap_to_grid_axis(value: float, axis: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(axis - float(value))))
    return float(axis[idx])


def build_rmse_grid_outer(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """外侧 11×11、20 cm 均匀网格 RMSE；内侧格置 NaN。"""
    if "true_x_m" not in df.columns or "true_y_m" not in df.columns:
        raise ValueError("CSV must contain true_x_m and true_y_m columns")
    if "rmse_xy_m" not in df.columns:
        raise ValueError("CSV must contain rmse_xy_m column")

    xs = ys = outer_grid_axis_m()
    outer_df = df[
        ~df.apply(
            lambda row: is_inner_target_xy_m(
                float(row["true_x_m"]), float(row["true_y_m"])
            ),
            axis=1,
        )
    ].copy()
    outer_df["true_x_m"] = outer_df["true_x_m"].map(lambda v: _snap_to_grid_axis(v, xs))
    outer_df["true_y_m"] = outer_df["true_y_m"].map(lambda v: _snap_to_grid_axis(v, ys))
    grouped = (
        outer_df.groupby(["true_y_m", "true_x_m"], as_index=False)["rmse_xy_m"]
        .mean(numeric_only=True)
    )
    grouped["true_x_m"] = grouped["true_x_m"].round(GRID_COORD_DECIMALS)
    grouped["true_y_m"] = grouped["true_y_m"].round(GRID_COORD_DECIMALS)
    pivot = grouped.pivot(index="true_y_m", columns="true_x_m", values="rmse_xy_m")
    pivot = pivot.reindex(index=ys, columns=xs)
    z = pivot.to_numpy(dtype=np.float64)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if is_inner_target_xy_m(float(x), float(y)):
                z[i, j] = np.nan
    return xs, ys, z


def build_rmse_grid_inner(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """内侧 11×11、10 cm 均匀网格 RMSE。"""
    if "true_x_m" not in df.columns or "true_y_m" not in df.columns:
        raise ValueError("CSV must contain true_x_m and true_y_m columns")
    if "rmse_xy_m" not in df.columns:
        raise ValueError("CSV must contain rmse_xy_m column")

    xs = ys = inner_grid_axis_m()
    inner_df = df[
        df.apply(
            lambda row: is_inner_target_xy_m(
                float(row["true_x_m"]), float(row["true_y_m"])
            ),
            axis=1,
        )
    ].copy()
    inner_df["true_x_m"] = inner_df["true_x_m"].map(lambda v: _snap_to_grid_axis(v, xs))
    inner_df["true_y_m"] = inner_df["true_y_m"].map(lambda v: _snap_to_grid_axis(v, ys))
    grouped = (
        inner_df.groupby(["true_y_m", "true_x_m"], as_index=False)["rmse_xy_m"]
        .mean(numeric_only=True)
    )
    grouped["true_x_m"] = grouped["true_x_m"].round(GRID_COORD_DECIMALS)
    grouped["true_y_m"] = grouped["true_y_m"].round(GRID_COORD_DECIMALS)
    pivot = grouped.pivot(index="true_y_m", columns="true_x_m", values="rmse_xy_m")
    pivot = pivot.reindex(index=ys, columns=xs)
    z = pivot.to_numpy(dtype=np.float64)
    return xs, ys, z


def _validate_rmse_columns(df: pd.DataFrame) -> None:
    if "true_x_m" not in df.columns or "true_y_m" not in df.columns:
        raise ValueError("CSV must contain true_x_m and true_y_m columns")
    if "rmse_xy_m" not in df.columns:
        raise ValueError("CSV must contain rmse_xy_m column")


def _outer_cell_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros_like(xx, dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            mask[i, j] = not is_inner_target_xy_m(float(x), float(y))
    return mask


def build_rmse_grid_10cm(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """全区域 21×21、10 cm 均匀网格 RMSE；返回 ``measured_invalid_mask``。"""
    _validate_rmse_columns(df)

    xs = ys = unified_grid_axis_m()
    work = df.copy()
    work["true_x_m"] = work["true_x_m"].map(lambda v: _snap_to_grid_axis(v, xs))
    work["true_y_m"] = work["true_y_m"].map(lambda v: _snap_to_grid_axis(v, ys))
    work["true_x_m"] = work["true_x_m"].round(GRID_COORD_DECIMALS)
    work["true_y_m"] = work["true_y_m"].round(GRID_COORD_DECIMALS)
    grouped = work.groupby(["true_y_m", "true_x_m"], as_index=False).agg(
        rmse_xy_m=("rmse_xy_m", "mean"),
        n=("rmse_xy_m", "size"),
    )
    grouped["true_x_m"] = grouped["true_x_m"].round(GRID_COORD_DECIMALS)
    grouped["true_y_m"] = grouped["true_y_m"].round(GRID_COORD_DECIMALS)
    pivot = grouped.pivot(index="true_y_m", columns="true_x_m", values="rmse_xy_m")
    pivot = pivot.reindex(index=ys, columns=xs)
    z = pivot.to_numpy(dtype=np.float64)
    count_pivot = grouped.pivot(index="true_y_m", columns="true_x_m", values="n")
    count_pivot = count_pivot.reindex(index=ys, columns=xs).fillna(0)
    measured_mask = count_pivot.to_numpy(dtype=np.float64) > 0
    measured_invalid_mask = measured_mask & ~np.isfinite(z)
    return xs, ys, z, measured_invalid_mask


def interpolate_outer_rmse_gaps(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    measured_invalid_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """外侧未采样格线性插值；凸包外 NaN 用 nearest 兜底。"""
    z_out = z.copy()
    xx, yy = np.meshgrid(xs, ys)
    outer_cell = _outer_cell_mask(xs, ys)
    fill_mask = outer_cell & ~np.isfinite(z_out) & ~measured_invalid_mask
    if not fill_mask.any():
        return z_out, 0

    source_mask = np.isfinite(z_out)
    points = np.column_stack([xx[source_mask], yy[source_mask]])
    values = z_out[source_mask]
    query = np.column_stack([xx[fill_mask], yy[fill_mask]])
    interp = griddata(points, values, query, method="linear")
    still_nan = ~np.isfinite(interp)
    if still_nan.any():
        interp[still_nan] = griddata(
            points,
            values,
            query[still_nan],
            method="nearest",
        )
    z_out[fill_mask] = interp
    return z_out, int(fill_mask.sum())


def build_rmse_grid_10cm_interpolated(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """全区域 10 cm 网格 RMSE，外侧缺失格插值填充。"""
    xs, ys, z, measured_invalid_mask = build_rmse_grid_10cm(df)
    z_interp, interpolated_cells = interpolate_outer_rmse_gaps(
        z,
        xs,
        ys,
        measured_invalid_mask=measured_invalid_mask,
    )
    return xs, ys, z_interp, interpolated_cells


def _cmap_with_bad(cmap: str | Colormap):
    cmap_obj = plt.get_cmap(cmap)
    if hasattr(cmap_obj, "with_extremes"):
        return cmap_obj.with_extremes(bad="#d9d9d9")
    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color="#d9d9d9")
    return cmap_obj


def _save_rmse_heatmap_figure(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    output_png: Path,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    title: str,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    cmap: str | Colormap,
    dpi: int,
    show: bool,
) -> dict[str, float | int]:
    valid = z[np.isfinite(z)]
    global_mean = float(valid.mean()) if valid.size else float("nan")
    filled_cells = int(np.isfinite(z).sum())

    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap_obj = _cmap_with_bad(cmap)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=cmap_obj,
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("RMSE (m)")

    ax.scatter(
        [dev0_xy[0]],
        [dev0_xy[1]],
        marker="^",
        s=80,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        label=f"dev0 ({dev0_xy[0]:.1f}, {dev0_xy[1]:.1f}) m",
        zorder=3,
    )
    ax.scatter(
        [dev1_xy[0]],
        [dev1_xy[1]],
        marker="s",
        s=70,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        label=f"dev1 ({dev1_xy[0]:.1f}, {dev1_xy[1]:.1f}) m",
        zorder=3,
    )

    ax.set_xlim(xs[0] - 0.1, xs[-1] + 0.1)
    ax.set_ylim(ys[0] - 0.1, ys[-1] + 0.1)
    ax.set_xlabel("Target x (m)")
    ax.set_ylabel("Target y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "filled_cells": filled_cells,
        "global_mean_rmse_m": global_mean,
    }


def plot_rmse_heatmap_from_csv(
    input_csv: Path,
    output_png: Path,
    *,
    dev0_xy: tuple[float, float] = (0.0, -2.0),
    dev1_xy: tuple[float, float] = (-2.0, 0.0),
    cmap: str | Colormap = DEFAULT_CMAP,
    dpi: int = 150,
    show: bool = False,
) -> dict[str, float | int]:
    """读取 RMSE CSV，绘制目标位置热力图并保存 PNG。"""
    df = pd.read_csv(input_csv)
    xs, ys, z = build_rmse_grid(df)

    valid = z[np.isfinite(z)]
    global_mean = float(valid.mean()) if valid.size else float("nan")
    filled_cells = int(np.isfinite(z).sum())
    title_mean = f"{global_mean:.3f}" if np.isfinite(global_mean) else "nan"
    summary = _save_rmse_heatmap_figure(
        xs,
        ys,
        z,
        output_png,
        x_edges=_axis_edges(xs),
        y_edges=_axis_edges(ys),
        title=(
            "Cooperative Monostatic MUSIC Localization RMSE\n"
            f"cells={filled_cells}, global mean RMSE={title_mean} m"
        ),
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        cmap=cmap,
        dpi=dpi,
        show=show,
    )
    return {
        **summary,
        "total_rows": int(len(df)),
    }


def plot_rmse_heatmap_outer_from_csv(
    input_csv: Path,
    output_png: Path,
    *,
    dev0_xy: tuple[float, float] = (0.0, -2.0),
    dev1_xy: tuple[float, float] = (-2.0, 0.0),
    cmap: str | Colormap = DEFAULT_CMAP,
    dpi: int = 150,
    show: bool = False,
) -> dict[str, float | int]:
    """读取 RMSE CSV，绘制外侧 20 cm 均匀网格热力图并保存 PNG。"""
    df = pd.read_csv(input_csv)
    xs, ys, z = build_rmse_grid_outer(df)

    valid = z[np.isfinite(z)]
    global_mean = float(valid.mean()) if valid.size else float("nan")
    filled_cells = int(np.isfinite(z).sum())
    title_mean = f"{global_mean:.3f}" if np.isfinite(global_mean) else "nan"
    summary = _save_rmse_heatmap_figure(
        xs,
        ys,
        z,
        output_png,
        x_edges=_uniform_axis_edges(xs, OUTER_GRID_STEP_M),
        y_edges=_uniform_axis_edges(ys, OUTER_GRID_STEP_M),
        title=(
            "Cooperative Monostatic MUSIC Localization RMSE (outer, 20 cm grid)\n"
            f"cells={filled_cells}, outer mean RMSE={title_mean} m"
        ),
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        cmap=cmap,
        dpi=dpi,
        show=show,
    )
    return {
        **summary,
        "total_rows": int(len(df)),
    }


def plot_rmse_heatmap_combined_from_csv(
    input_csv: Path,
    output_png: Path,
    *,
    dev0_xy: tuple[float, float] = (0.0, -2.0),
    dev1_xy: tuple[float, float] = (-2.0, 0.0),
    cmap: str | Colormap = DEFAULT_CMAP,
    dpi: int = 150,
    show: bool = False,
    title_prefix: str = "Cooperative Monostatic MUSIC Localization RMSE",
) -> dict[str, float | int]:
    """读取 RMSE CSV，绘制全区域 10 cm 统一网格热力图（外侧插值）。"""
    df = pd.read_csv(input_csv)
    xs, ys, z, interpolated_cells = build_rmse_grid_10cm_interpolated(df)

    valid = z[np.isfinite(z)]
    global_mean = float(valid.mean()) if valid.size else float("nan")
    filled_cells = int(np.isfinite(z).sum())
    title_mean = f"{global_mean:.3f}" if np.isfinite(global_mean) else "nan"
    summary = _save_rmse_heatmap_figure(
        xs,
        ys,
        z,
        output_png,
        x_edges=_uniform_axis_edges(xs, UNIFIED_GRID_STEP_M),
        y_edges=_uniform_axis_edges(ys, UNIFIED_GRID_STEP_M),
        title=(
            f"{title_prefix}\n"
            f"(uniform 10 cm grid, outer interpolated), cells={filled_cells}, "
            f"mean RMSE={title_mean} m"
        ),
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        cmap=cmap,
        dpi=dpi,
        show=show,
    )
    return {
        **summary,
        "interpolated_cells": interpolated_cells,
        "total_rows": int(len(df)),
    }


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对有限 RMSE 值计算经验 CDF，返回 ``(x_sorted, cdf_y)``。"""
    valid = np.sort(values[np.isfinite(values)])
    if valid.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    cdf_y = np.arange(1, valid.size + 1, dtype=np.float64) / valid.size
    return valid, cdf_y


def _split_rmse_by_region(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """按 global / inner / outer 拆分 ``rmse_xy_m`` 数组。"""
    if "true_x_m" not in df.columns or "true_y_m" not in df.columns:
        raise ValueError("CSV must contain true_x_m and true_y_m columns")
    if "rmse_xy_m" not in df.columns:
        raise ValueError("CSV must contain rmse_xy_m column")

    rmses = df["rmse_xy_m"].to_numpy(dtype=np.float64)
    inner_mask = df.apply(
        lambda row: is_inner_target_xy_m(
            float(row["true_x_m"]), float(row["true_y_m"])
        ),
        axis=1,
    ).to_numpy(dtype=bool)
    return {
        "global": rmses,
        "inner": rmses[inner_mask],
        "outer": rmses[~inner_mask],
    }


def plot_rmse_cdf_from_csv(
    input_csv: Path,
    output_png: Path,
    *,
    title: str = "Localization RMSE CDF",
    dpi: int = 150,
    show: bool = False,
) -> dict[str, int | float | str]:
    """从 RMSE CSV 绘制 global / inner / outer 经验 CDF 曲线。"""
    df = pd.read_csv(input_csv)
    by_region = _split_rmse_by_region(df)

    curve_styles = {
        "global": {"linestyle": "-", "label": "global"},
        "inner": {"linestyle": "--", "label": "inner (|x|,|y| <= 0.5 m)"},
        "outer": {"linestyle": "-.", "label": "outer"},
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    curves_plotted = 0
    summary: dict[str, int | float | str] = {"total_rows": int(len(df))}

    for region, values in by_region.items():
        valid_count = int(np.isfinite(values).sum())
        summary[f"{region}_valid"] = valid_count
        x_cdf, y_cdf = _empirical_cdf(values)
        if x_cdf.size == 0:
            continue
        style = curve_styles[region]
        ax.plot(
            x_cdf,
            y_cdf,
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=f"{style['label']} (n={valid_count})",
        )
        curves_plotted += 1

    if curves_plotted == 0:
        raise ValueError(f"no finite rmse_xy_m values in {input_csv}")

    ax.set_xlabel("RMSE (m)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    summary["curves_plotted"] = curves_plotted
    summary["output_png"] = str(output_png)
    return summary


def _default_output_png() -> Path:
    return PROJECT_ROOT / "out" / "cooperative_monostatic" / "music_rmse_heatmap.png"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cooperative monostatic MUSIC RMSE heatmap by target position"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=_default_input_csv(),
        help="input RMSE CSV from run_cooperative_monostatic_music_rmse.py",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=_default_output_png(),
        help="output combined heatmap PNG path (default)",
    )
    parser.add_argument(
        "--output-combined-png",
        type=Path,
        default=None,
        help="alias for heatmap output (overrides --output-png if set)",
    )
    parser.add_argument(
        "--legacy-nonuniform-heatmap",
        action="store_true",
        help="also write legacy non-uniform full-grid heatmap",
    )
    parser.add_argument(
        "--legacy-outer-20cm-heatmap",
        action="store_true",
        help="also write legacy outer 20 cm uniform grid heatmap",
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
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help="matplotlib colormap name (default: RdYlGn_r, green=good low RMSE)",
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
    input_csv = args.input_csv.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)

    plot_kwargs = {
        "dev0_xy": (float(args.dev0_xy[0]), float(args.dev0_xy[1])),
        "dev1_xy": (float(args.dev1_xy[0]), float(args.dev1_xy[1])),
        "cmap": args.cmap,
        "dpi": int(args.dpi),
        "show": bool(args.show),
    }
    combined_png = (
        args.output_combined_png.resolve()
        if args.output_combined_png is not None
        else args.output_png.resolve()
    )
    plot_rmse_heatmap_combined_from_csv(
        input_csv,
        combined_png,
        **plot_kwargs,
    )
    print(f"output heatmap: {combined_png}")
    if args.legacy_outer_20cm_heatmap:
        plot_rmse_heatmap_outer_from_csv(
            input_csv,
            _default_output_outer_png().resolve(),
            **plot_kwargs,
        )
    if args.legacy_nonuniform_heatmap:
        legacy_path = combined_png.with_name(
            combined_png.stem + "_legacy_nonuniform" + combined_png.suffix
        )
        plot_rmse_heatmap_from_csv(
            input_csv,
            legacy_path,
            **plot_kwargs,
        )


if __name__ == "__main__":
    main()
