#!/usr/bin/env python3
"""Cooperative monostatic 实验场景示意图：场地、双设备、三区域划分、采集样点。

示例::

    python script/experiment/plot_cooperative_monostatic_scene_schematic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from isac import PROJECT_ROOT
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_DEV0_RX_XY,
    DEFAULT_DEV0_TX_XY,
    DEFAULT_DEV0_XY,
    DEFAULT_DEV1_RX_XY,
    DEFAULT_DEV1_TX_XY,
    DEFAULT_DEV1_XY,
)
from isac_imp.record_target_metadata import (
    INNER_RADIUS_CM,
    SUBREGION_GRID_MAX_M,
    SUBREGION_GRID_MIN_M,
)

INNER_RADIUS_M = INNER_RADIUS_CM / 100.0

# 半透明分区色（中心 / 侧边 / 角）
_ZONE_COLORS = {
    "center": "#4C78A8",
    "side": "#F58518",
    "corner": "#54A24B",
}
_ZONE_ALPHA = 0.35


def _default_output_png() -> Path:
    return (
        PROJECT_ROOT
        / "out"
        / "cooperative_monostatic"
        / "methods_compare"
        / "scene_schematic.png"
    )


def _default_h5_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "experiment"
        / "cooperative_monostatic"
        / "cooperative_monostatic_dataset.h5"
    )


def load_unique_sample_xy(h5_path: Path) -> np.ndarray:
    """从数据集读取唯一目标位置 ``(N, 2)``，单位 m。"""
    h5_path = h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as handle:
        if "target_position" not in handle:
            raise KeyError(f"{h5_path} missing dataset 'target_position'")
        xy = np.asarray(handle["target_position"][:, :2], dtype=np.float64)
    rounded = np.round(xy, decimals=6)
    return np.unique(rounded, axis=0)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cooperative monostatic experiment scene schematic"
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=_default_output_png(),
        help="output PNG path",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=_default_h5_path(),
        help="dataset H5 used to load unique sample positions",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="figure DPI (default: 150)",
    )
    return parser.parse_args()


def _add_zone_rect(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    zone: str,
) -> None:
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=_ZONE_COLORS[zone],
            edgecolor="none",
            alpha=_ZONE_ALPHA,
            linewidth=0,
            zorder=1,
        )
    )


def plot_scene_schematic(
    output_png: Path,
    *,
    h5_path: Path | None = None,
    sample_xy: np.ndarray | None = None,
    dpi: int = 150,
    show: bool = False,
) -> Path:
    """绘制实验区域、三区划分、采集样点与 BS-0/BS-1 位置，保存 PNG。"""
    grid_min = float(SUBREGION_GRID_MIN_M)
    grid_max = float(SUBREGION_GRID_MAX_M)
    r = float(INNER_RADIUS_M)
    span = grid_max - grid_min  # 2.0 m
    # 角区边长：[-1,-0.5] 与 [0.5,1]
    outer_neg = -r - grid_min  # 0.5
    outer_pos = grid_max - r  # 0.5

    if sample_xy is None:
        src = h5_path if h5_path is not None else _default_h5_path()
        sample_xy = load_unique_sample_xy(Path(src))
    sample_xy = np.asarray(sample_xy, dtype=np.float64).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # --- 三区九宫格填充 ---
    # corners: SW / SE / NW / NE
    _add_zone_rect(
        ax, x0=grid_min, y0=grid_min, width=outer_neg, height=outer_neg, zone="corner"
    )
    _add_zone_rect(
        ax, x0=r, y0=grid_min, width=outer_pos, height=outer_neg, zone="corner"
    )
    _add_zone_rect(
        ax, x0=grid_min, y0=r, width=outer_neg, height=outer_pos, zone="corner"
    )
    _add_zone_rect(ax, x0=r, y0=r, width=outer_pos, height=outer_pos, zone="corner")
    # sides: S / N / W / E
    _add_zone_rect(
        ax, x0=-r, y0=grid_min, width=2 * r, height=outer_neg, zone="side"
    )
    _add_zone_rect(ax, x0=-r, y0=r, width=2 * r, height=outer_pos, zone="side")
    _add_zone_rect(
        ax, x0=grid_min, y0=-r, width=outer_neg, height=2 * r, zone="side"
    )
    _add_zone_rect(ax, x0=r, y0=-r, width=outer_pos, height=2 * r, zone="side")
    # center
    _add_zone_rect(ax, x0=-r, y0=-r, width=2 * r, height=2 * r, zone="center")

    # --- 实验区外框 ---
    ax.add_patch(
        Rectangle(
            (grid_min, grid_min),
            span,
            span,
            fill=False,
            edgecolor="#222222",
            linewidth=1.8,
            zorder=3,
        )
    )

    # --- 三区边界辅助线 ---
    for v in (-r, r):
        ax.axvline(v, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
        ax.axhline(v, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)

    # --- 数据采集样点 ---
    ax.scatter(
        sample_xy[:, 0],
        sample_xy[:, 1],
        s=22,
        c="#222222",
        marker="o",
        edgecolors="#FFFFFF",
        linewidths=0.4,
        alpha=0.95,
        zorder=4,
    )

    # --- 收发机位置（TX / RX，相对站点中点 ±5 cm）---
    tx0 = (float(DEFAULT_DEV0_TX_XY[0]), float(DEFAULT_DEV0_TX_XY[1]))
    rx0 = (float(DEFAULT_DEV0_RX_XY[0]), float(DEFAULT_DEV0_RX_XY[1]))
    tx1 = (float(DEFAULT_DEV1_TX_XY[0]), float(DEFAULT_DEV1_TX_XY[1]))
    rx1 = (float(DEFAULT_DEV1_RX_XY[0]), float(DEFAULT_DEV1_RX_XY[1]))
    dev0 = (float(DEFAULT_DEV0_XY[0]), float(DEFAULT_DEV0_XY[1]))
    dev1 = (float(DEFAULT_DEV1_XY[0]), float(DEFAULT_DEV1_XY[1]))

    ax.scatter(
        [tx0[0], tx1[0]],
        [tx0[1], tx1[1]],
        s=90,
        c="#C44E52",
        marker="s",
        zorder=5,
        edgecolors="#222222",
        linewidths=0.8,
    )
    ax.scatter(
        [rx0[0], rx1[0]],
        [rx0[1], rx1[1]],
        s=90,
        c="#4C78A8",
        marker="s",
        zorder=5,
        edgecolors="#222222",
        linewidths=0.8,
    )

    # 仅标注站点名称与载波频率；TX/RX 用颜色区分，不标具体收发名
    ax.annotate(
        "BS-0\n6.0 GHz",
        xy=dev0,
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=11,
        ha="left",
        va="bottom",
        zorder=6,
    )
    ax.annotate(
        "BS-1\n3.5 GHz",
        xy=dev1,
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=11,
        ha="left",
        va="bottom",
        zorder=6,
    )

    # 观测方向短箭头（自站点中点指向场地中心）
    for origin in (dev0, dev1):
        ax.add_patch(
            FancyArrowPatch(
                origin,
                (0.55 * origin[0], 0.55 * origin[1]),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#C44E52",
                zorder=4,
            )
        )

    # --- 坐标与图例 ---
    ax.set_xlim(-2.2, 1.2)
    ax.set_ylim(-2.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xticks([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
    ax.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5, zorder=0)
    ax.axvline(0.0, color="#888888", linewidth=0.6, alpha=0.5, zorder=0)

    legend_handles = [
        Patch(
            facecolor=_ZONE_COLORS["center"],
            edgecolor="none",
            alpha=_ZONE_ALPHA,
            label="Center Zone",
        ),
        Patch(
            facecolor=_ZONE_COLORS["side"],
            edgecolor="none",
            alpha=_ZONE_ALPHA,
            label="Side Zone",
        ),
        Patch(
            facecolor=_ZONE_COLORS["corner"],
            edgecolor="none",
            alpha=_ZONE_ALPHA,
            label="Corner Zone",
        ),
        plt.Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            color="w",
            markerfacecolor="#222222",
            markeredgecolor="#FFFFFF",
            markersize=6,
            label="Sample",
        ),
        plt.Line2D(
            [0],
            [0],
            linestyle="None",
            marker="s",
            color="#C44E52",
            markerfacecolor="#C44E52",
            markeredgecolor="#222222",
            markersize=9,
            label="TX",
        ),
        plt.Line2D(
            [0],
            [0],
            linestyle="None",
            marker="s",
            color="#4C78A8",
            markerfacecolor="#4C78A8",
            markeredgecolor="#222222",
            markersize=9,
            label="RX",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.92)

    fig.tight_layout()
    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_png


def main() -> None:
    args = argument_parser()
    sample_xy = load_unique_sample_xy(args.h5_path)
    out = plot_scene_schematic(
        args.output_png,
        sample_xy=sample_xy,
        dpi=int(args.dpi),
    )
    print(
        f"output scene schematic: {out} (n_samples={len(sample_xy)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
