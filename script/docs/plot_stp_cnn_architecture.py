#!/usr/bin/env python3
"""绘制当前最优 STP-CNN（CooperativeMonostaticCNN late+attention）架构示意图。

风格对标多视图语义通信 Encoder / Joint Decoder 论文图：左右分栏、并行支路、
模块框内标注层参数；左下角内嵌 ResBlock 双路径读图（对齐 Conv1dResidualBlock）。
默认写出 ``docs/figures/stp_cnn_architecture.png``。

示例::

    python script/docs/plot_stp_cnn_architecture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from isac import PROJECT_ROOT


def _default_output_png() -> Path:
    return PROJECT_ROOT / "docs" / "figures" / "stp_cnn_architecture.png"


def _box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "#333333",
    linewidth: float = 1.2,
    radius: float = 0.02,
    linestyle: str = "-",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_aspect=0.3,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    lw: float = 1.3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            zorder=3,
        )
    )


def _centered_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 8,
    fontweight: str = "normal",
    color: str = "#222222",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        zorder=4,
        linespacing=1.25,
    )


def _draw_resblock_inset(ax, origin: tuple[float, float]) -> None:
    """示例读图样式：主路 + 捷径 → ⊕ → ReLU（忠实于 Conv1dResidualBlock）。"""
    x0, y0 = origin
    c_block = "#f3e6c8"
    c_edge = "#c48a2a"
    c_dash = "#d4a017"
    width, height = 7.45, 1.55

    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="#fffaf0",
            edgecolor=c_dash,
            linewidth=1.5,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        x0 + width / 2,
        y0 + 1.38,
        "ResBlock (Conv1dResidualBlock)   ·   Conv(c, k, s)",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#6b4e16",
        zorder=4,
    )
    ax.text(
        x0 + width / 2,
        y0 + 1.18,
        "shortcut: Identity if same shape; else Conv(c_out, 1, s)+BN",
        ha="center",
        va="center",
        fontsize=6,
        color="#8a7040",
        style="italic",
        zorder=4,
    )

    yin, ysc, ymain = y0 + 0.55, y0 + 0.95, y0 + 0.28
    split_x = x0 + 0.55
    sum_x = x0 + 5.75

    ax.annotate(
        "",
        xy=(split_x, yin),
        xytext=(x0 + 0.15, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1),
        zorder=3,
    )
    ax.text(x0 + 0.12, yin + 0.18, "in", fontsize=6.5, ha="center", color="#444")
    ax.plot([split_x, split_x], [ymain, ysc], color="#333", lw=1.0, zorder=3)

    # Shortcut (upper)
    sc_x, sc_w = x0 + 0.75, 2.4
    ax.plot([split_x, sc_x], [ysc, ysc], color="#333", lw=1.0, zorder=3)
    _box(
        ax,
        (sc_x, ysc - 0.16),
        sc_w,
        0.32,
        facecolor=c_block,
        edgecolor=c_edge,
        radius=0.02,
    )
    _centered_text(ax, sc_x + sc_w / 2, ysc, "Conv(c_out, 1, s) / Identity", fontsize=6)
    ax.plot([sc_x + sc_w, sum_x - 0.16], [ysc, yin], color="#333", lw=1.0, zorder=3)

    # Main path (lower)
    main_specs = [
        (x0 + 0.75, 1.2, "Conv(c_out, 3, s)"),
        (x0 + 2.10, 0.85, "BN + ReLU"),
        (x0 + 3.10, 1.2, "Conv(c_out, 3, 1)"),
        (x0 + 4.45, 0.55, "BN"),
    ]
    ax.plot([split_x, main_specs[0][0]], [ymain, ymain], color="#333", lw=1.0, zorder=3)
    for i, (bx, bw, label) in enumerate(main_specs):
        _box(
            ax,
            (bx, ymain - 0.16),
            bw,
            0.32,
            facecolor=c_block,
            edgecolor=c_edge,
            radius=0.02,
        )
        _centered_text(ax, bx + bw / 2, ymain, label, fontsize=5.5)
        if i + 1 < len(main_specs):
            nx = main_specs[i + 1][0]
            ax.annotate(
                "",
                xy=(nx, ymain),
                xytext=(bx + bw, ymain),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9, mutation_scale=8),
                zorder=3,
            )
    last_x = main_specs[-1][0] + main_specs[-1][1]
    ax.plot([last_x, sum_x - 0.16], [ymain, yin], color="#333", lw=1.0, zorder=3)

    circ = Circle(
        (sum_x, yin), 0.16, facecolor="#fff8e7", edgecolor="#333", lw=1.2, zorder=3
    )
    ax.add_patch(circ)
    _centered_text(ax, sum_x, yin, r"$\oplus$", fontsize=10, fontweight="bold")

    relu_x = sum_x + 0.35
    ax.annotate(
        "",
        xy=(relu_x, yin),
        xytext=(sum_x + 0.16, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0, mutation_scale=8),
        zorder=3,
    )
    _box(
        ax,
        (relu_x, yin - 0.16),
        0.7,
        0.32,
        facecolor=c_block,
        edgecolor=c_edge,
        radius=0.02,
    )
    _centered_text(ax, relu_x + 0.35, yin, "ReLU", fontsize=6.5)
    ax.annotate(
        "",
        xy=(x0 + 7.35, yin),
        xytext=(relu_x + 0.7, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0, mutation_scale=8),
        zorder=3,
    )
    ax.text(x0 + 7.4, yin + 0.18, "out", fontsize=6.5, ha="left", color="#444")


def _draw_rap_inset(ax, origin: tuple[float, float]) -> None:
    """双分支读图：打分路上 + 特征路下 → ⊙ → Σ（忠实于 RangeAttentionPool1d）。"""
    x0, y0 = origin
    c_block = "#c5d8ef"
    c_edge = "#3d5a80"
    c_dash = "#5a7a9a"
    width, height = 6.25, 1.55

    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="#eef4fa",
            edgecolor=c_dash,
            linewidth=1.5,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        x0 + width / 2,
        y0 + 1.38,
        r"RAP (Range Attention Pool)   ·   $y_n\!\to\!s_n$",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#2c3e50",
        zorder=4,
    )
    ax.text(
        x0 + width / 2,
        y0 + 1.18,
        r"$y_n\in\mathbb{R}^{C\times L'}$  →  $s_n\in\mathbb{R}^{C}$  (C=256)",
        ha="center",
        va="center",
        fontsize=6,
        color="#4a6080",
        style="italic",
        zorder=4,
    )

    yin, ysc, yfeat = y0 + 0.55, y0 + 0.95, y0 + 0.28
    split_x = x0 + 0.45
    mul_x = x0 + 4.15

    ax.annotate(
        "",
        xy=(split_x, yin),
        xytext=(x0 + 0.12, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1),
        zorder=3,
    )
    ax.text(
        x0 + 0.12,
        yin + 0.18,
        r"$y_n$",
        fontsize=6.5,
        ha="center",
        color="#444",
    )
    ax.plot([split_x, split_x], [yfeat, ysc], color="#333", lw=1.0, zorder=3)

    # Score path (upper): Conv(1,1,1) → Softmax → α
    ax.plot([split_x, x0 + 0.65], [ysc, ysc], color="#333", lw=1.0, zorder=3)
    score_specs = [
        (x0 + 0.65, 1.35, "Conv(1, 1, 1)"),
        (x0 + 2.15, 0.95, "Softmax"),
        (x0 + 3.25, 0.45, r"$\alpha$"),
    ]
    for i, (bx, bw, label) in enumerate(score_specs):
        _box(
            ax,
            (bx, ysc - 0.16),
            bw,
            0.32,
            facecolor=c_block,
            edgecolor=c_edge,
            radius=0.02,
        )
        _centered_text(ax, bx + bw / 2, ysc, label, fontsize=6)
        if i + 1 < len(score_specs):
            nx = score_specs[i + 1][0]
            ax.annotate(
                "",
                xy=(nx, ysc),
                xytext=(bx + bw, ysc),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9, mutation_scale=8),
                zorder=3,
            )
    ax.plot(
        [score_specs[-1][0] + score_specs[-1][1], mul_x - 0.16],
        [ysc, yin],
        color="#333",
        lw=1.0,
        zorder=3,
    )

    # Feature path (lower): identity
    ax.plot([split_x, x0 + 0.65], [yfeat, yfeat], color="#333", lw=1.0, zorder=3)
    _box(
        ax,
        (x0 + 0.65, yfeat - 0.16),
        2.6,
        0.32,
        facecolor="#ffffff",
        edgecolor=c_edge,
        radius=0.02,
    )
    _centered_text(ax, x0 + 1.95, yfeat, "identity (features)", fontsize=6)
    ax.plot([x0 + 3.25, mul_x - 0.16], [yfeat, yin], color="#333", lw=1.0, zorder=3)

    # Element-wise multiply ⊙
    circ = Circle(
        (mul_x, yin), 0.16, facecolor="#e3f2fd", edgecolor="#333", lw=1.2, zorder=3
    )
    ax.add_patch(circ)
    _centered_text(ax, mul_x, yin, r"$\odot$", fontsize=10, fontweight="bold")

    # Sum over range
    sum_x = mul_x + 0.45
    ax.annotate(
        "",
        xy=(sum_x, yin),
        xytext=(mul_x + 0.16, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0, mutation_scale=8),
        zorder=3,
    )
    _box(
        ax,
        (sum_x, yin - 0.16),
        0.7,
        0.32,
        facecolor=c_block,
        edgecolor=c_edge,
        radius=0.02,
    )
    _centered_text(ax, sum_x + 0.35, yin, r"$\sum_\ell$", fontsize=7)
    ax.annotate(
        "",
        xy=(x0 + width - 0.15, yin),
        xytext=(sum_x + 0.7, yin),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0, mutation_scale=8),
        zorder=3,
    )
    ax.text(
        x0 + width - 0.1,
        yin + 0.18,
        r"$s_n$",
        fontsize=6.5,
        ha="left",
        color="#444",
    )


def plot_stp_cnn_architecture(
    output_png: Path,
    *,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    """绘制 STP-CNN 架构图并保存。"""
    fig, ax = plt.subplots(figsize=(14.5, 8.6))
    ax.set_xlim(0.0, 14.5)
    ax.set_ylim(0.0, 8.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- title ----
    ax.text(
        7.25,
        8.35,
        "STP-CNN  (CooperativeMonostaticCNN, late + attention)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax.text(
        7.25,
        8.02,
        "Dual-station ROI range spectra  →  target (x, y)   |   "
        "shared-weight encoder  ·  ~5.0×10⁵ params  ·  aug_spec_only",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    # ---- section frames ----
    ax.add_patch(
        Rectangle(
            (0.25, 2.05),
            8.35,
            5.65,
            facecolor="#f7f9fc",
            edgecolor="#7a8aa0",
            linewidth=1.4,
            linestyle="-",
            zorder=0,
        )
    )
    ax.text(
        4.4,
        7.45,
        "Dual-Station Shared Spectral Encoder (DSSE)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#2c3e50",
        zorder=1,
    )
    ax.text(
        4.4,
        7.18,
        "weights shared across Dev0 / Dev1",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#667788",
        style="italic",
        zorder=1,
    )

    ax.add_patch(
        Rectangle(
            (8.85, 2.05),
            5.4,
            5.65,
            facecolor="#f5faf5",
            edgecolor="#6a9470",
            linewidth=1.4,
            zorder=0,
        )
    )
    ax.text(
        11.55,
        7.45,
        "Late Fusion Regression Head (LFRH)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#2d5a32",
        zorder=1,
    )
    ax.text(
        11.55,
        7.18,
        "joint decision  ·  no decoder / reconstruction",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#557755",
        style="italic",
        zorder=1,
    )

    c_feb = "#e8eef7"
    c_stem = "#d6e4f5"
    c_res = "#c5d8ef"
    c_rap = "#b3cce8"
    c_fuse = "#c8e6c9"
    c_mlp = "#a5d6a7"
    c_out = "#81c784"

    # ---- Input FEB ----
    _box(ax, (0.5, 6.25), 7.85, 0.72, facecolor=c_feb, edgecolor="#5a6f8a")
    _centered_text(
        ax,
        4.425,
        6.61,
        "FEB  Feature Extraction Block  (non-learnable)\n"
        "profiles_dev0/dev1  →  ROI 0–4 m  →  real_imag  →  (B, 4, L)\n"
        "split: Dev0 (B,2,L)  |  Dev1 (B,2,L)",
        fontsize=7.5,
    )

    # ---- Two parallel station branches ----
    branch_y = {
        "dev0": 4.85,
        "dev1": 2.55,
    }
    branch_label = {
        "dev0": r"$x_0$  Dev0 (re, im)",
        "dev1": r"$x_1$  Dev1 (re, im)",
    }
    branch_out = {
        "dev0": r"$s_0\in\mathbb{R}^{256}$",
        "dev1": r"$s_1\in\mathbb{R}^{256}$",
    }

    for key, y0 in branch_y.items():
        _box(ax, (0.45, y0 + 0.55), 1.15, 0.55, facecolor="#ffffff", edgecolor="#888")
        _centered_text(ax, 1.025, y0 + 0.825, branch_label[key], fontsize=7)

        _box(ax, (1.85, y0 + 0.15), 1.55, 1.35, facecolor=c_stem)
        _centered_text(
            ax,
            2.625,
            y0 + 0.825,
            "Stem (CB₀)\n"
            "Conv1d(2→64, k=7, s=2)\n"
            "+ BN + ReLU\n"
            "MaxPool1d(k=3, s=2)\n"
            r"$L\!\downarrow\!{\approx}4$",
            fontsize=6.5,
        )

        # Compact ResBlock labels (detail in inset)
        _box(ax, (3.55, y0 + 0.35), 1.25, 0.95, facecolor=c_res)
        _centered_text(
            ax,
            4.175,
            y0 + 0.825,
            "ResBlock₁\n"
            "64→64\n"
            "s=1",
            fontsize=7,
        )

        _box(ax, (4.95, y0 + 0.35), 1.25, 0.95, facecolor=c_res)
        _centered_text(
            ax,
            5.575,
            y0 + 0.825,
            "ResBlock₂\n"
            "64→128\n"
            r"s=2, $L\!\downarrow\!2$",
            fontsize=7,
        )

        _box(ax, (6.35, y0 + 0.35), 1.25, 0.95, facecolor=c_res)
        _centered_text(
            ax,
            6.975,
            y0 + 0.825,
            "ResBlock₃\n"
            "128→256\n"
            r"$y_n\!\in\!\mathbb{R}^{256\times L'}$",
            fontsize=6.5,
        )

        _arrow(ax, (1.60, y0 + 0.825), (1.85, y0 + 0.825))
        _arrow(ax, (3.40, y0 + 0.825), (3.55, y0 + 0.825))
        _arrow(ax, (4.80, y0 + 0.825), (4.95, y0 + 0.825))
        _arrow(ax, (6.20, y0 + 0.825), (6.35, y0 + 0.825))

    ax.annotate(
        "",
        xy=(2.625, 4.75),
        xytext=(2.625, 3.95),
        arrowprops=dict(arrowstyle="<->", color="#5a6f8a", lw=1.2),
    )
    ax.text(
        2.95,
        4.35,
        "shared\nweights",
        fontsize=7,
        color="#5a6f8a",
        ha="left",
        va="center",
        style="italic",
    )

    # ---- RAP (compact on branch; detail in bottom inset) ----
    for key, y0 in branch_y.items():
        _box(ax, (7.75, y0 + 0.25), 0.7, 1.15, facecolor=c_rap, edgecolor="#3d5a80")
        _centered_text(
            ax,
            8.1,
            y0 + 0.825,
            "RAP\nAttn\nPool",
            fontsize=7,
            fontweight="bold",
        )
        _arrow(ax, (7.60, y0 + 0.825), (7.75, y0 + 0.825))

    # ---- Bottom insets: ResBlock + RAP dual-path schematics ----
    _draw_resblock_inset(ax, (0.3, 0.35))
    _draw_rap_inset(ax, (8.0, 0.35))

    # ---- Late fusion side ----
    _box(ax, (9.1, 5.4), 1.5, 0.7, facecolor="#e8f5e9", edgecolor="#558b2f")
    _centered_text(ax, 9.85, 5.75, branch_out["dev0"], fontsize=8)

    _box(ax, (9.1, 2.95), 1.5, 0.7, facecolor="#e8f5e9", edgecolor="#558b2f")
    _centered_text(ax, 9.85, 3.3, branch_out["dev1"], fontsize=8)

    _arrow(ax, (8.45, branch_y["dev0"] + 0.825), (9.1, 5.75), color="#2e7d32")
    _arrow(ax, (8.45, branch_y["dev1"] + 0.825), (9.1, 3.3), color="#2e7d32")

    _box(ax, (10.9, 3.95), 1.55, 1.2, facecolor=c_fuse, edgecolor="#2e7d32")
    _centered_text(
        ax,
        11.675,
        4.55,
        "Concat\n"
        r"$s=[s_0\,\|\,s_1]$"
        "\n"
        r"$\in\mathbb{R}^{512}$",
        fontsize=8,
        fontweight="bold",
    )
    _arrow(ax, (10.6, 5.75), (11.675, 5.15), color="#2e7d32")
    _arrow(ax, (10.6, 3.3), (11.675, 3.95), color="#2e7d32")

    _box(ax, (12.65, 3.75), 1.35, 1.6, facecolor=c_mlp, edgecolor="#1b5e20")
    _centered_text(
        ax,
        13.325,
        4.55,
        "xy_head MLP\n"
        "Linear 512→128\n"
        "ReLU\n"
        "Dropout 0.3\n"
        "Linear 128→2",
        fontsize=7.5,
    )
    _arrow(ax, (12.45, 4.55), (12.65, 4.55), color="#1b5e20")

    _box(ax, (12.55, 2.25), 1.55, 0.85, facecolor=c_out, edgecolor="#1b5e20", linewidth=1.6)
    _centered_text(
        ax,
        13.325,
        2.675,
        r"$\hat{\mathbf{p}}=(x,y)$" + "\n"
        "meters",
        fontsize=9,
        fontweight="bold",
    )
    _arrow(ax, (13.325, 3.75), (13.325, 3.1), color="#1b5e20")

    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return output_png


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot STP-CNN architecture schematic for docs"
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=_default_output_png(),
        help="output PNG path (default: docs/figures/stp_cnn_architecture.png)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="figure DPI")
    parser.add_argument("--show", action="store_true", help="display figure window")
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    out = plot_stp_cnn_architecture(args.output_png, dpi=args.dpi, show=args.show)
    print(f"wrote: {out}", flush=True)


if __name__ == "__main__":
    main()
