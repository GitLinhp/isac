#!/usr/bin/env python3
"""绘制 STP-CNN 架构示意图（对标 Fu et al. IEEE Commun. Lett. 2025 Fig. 2）。

左右分栏：Spectral Encoder / Joint Localization Decoder；并行双站支路；
虚线内嵌展开 Stem、ResBlock、RAP；记号 Conv(c,k,s)，↓ 为距离维下采样。
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

# Fig.2-like palette
_C_MOD = "#f5e6c8"  # module fill
_C_EDGE = "#c48a2a"
_C_ENC = "#faf6f0"
_C_DEC = "#f7f3ea"
_C_CALL = "#fffaf0"
_C_DASH = "#d4a017"
_C_TEXT = "#222222"


def _default_output_png() -> Path:
    return PROJECT_ROOT / "docs" / "figures" / "stp_cnn_architecture.png"


def _box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = _C_MOD,
    edgecolor: str = _C_EDGE,
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
    color: str = "#222222",
    lw: float = 1.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            zorder=3,
        )
    )


def _text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 8,
    fontweight: str = "normal",
    color: str = _C_TEXT,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        zorder=4,
        linespacing=1.2,
    )


def _draw_stem_callout(ax, origin: tuple[float, float]) -> None:
    """虚线展开 Stem/CB0（对齐 Fig.2 的 CB 内嵌竖条）。"""
    x0, y0 = origin
    w, h = 2.15, 2.35
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=_C_CALL,
            edgecolor=_C_DASH,
            linewidth=1.4,
            linestyle="--",
            zorder=1,
        )
    )
    _text(
        ax,
        x0 + w / 2,
        y0 + h - 0.18,
        r"CB$_0$ / Stem",
        fontsize=8,
        fontweight="bold",
        color="#6b4e16",
    )
    layers = [
        (y0 + 1.75, 0.42, "Conv(64, 7, 2" + r"$\downarrow$" + ")"),
        (y0 + 1.20, 0.42, "BN + ReLU"),
        (y0 + 0.65, 0.42, "MaxPool(3, 2" + r"$\downarrow$" + ")"),
        (y0 + 0.18, 0.32, r"$L\!\downarrow\!{\approx}4$"),
    ]
    for y, hh, lab in layers:
        _box(ax, (x0 + 0.2, y), w - 0.4, hh, facecolor=_C_MOD, edgecolor=_C_EDGE)
        _text(ax, x0 + w / 2, y + hh / 2, lab, fontsize=6.5)
    for i in range(len(layers) - 1):
        y_from = layers[i][0]
        y_to = layers[i + 1][0] + layers[i + 1][1]
        _arrow(ax, (x0 + w / 2, y_from), (x0 + w / 2, y_to), lw=0.9)


def _draw_resblock_callout(ax, origin: tuple[float, float]) -> None:
    """ResBlock 双路径（对齐 Fig.3 残差读图）。"""
    x0, y0 = origin
    w, h = 5.4, 1.55
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=_C_CALL,
            edgecolor=_C_DASH,
            linewidth=1.4,
            linestyle="--",
            zorder=1,
        )
    )
    _text(
        ax,
        x0 + w / 2,
        y0 + h - 0.18,
        r"ResBlock  ·  Conv($c$, $k$, $s$)",
        fontsize=8,
        fontweight="bold",
        color="#6b4e16",
    )
    _text(
        ax,
        x0 + w / 2,
        y0 + h - 0.38,
        "shortcut: Identity if same shape; else Conv($c_{out}$, 1, $s$)+BN",
        fontsize=5.5,
        color="#8a7040",
    )

    yin, ysc, ymain = y0 + 0.48, y0 + 0.88, y0 + 0.22
    split_x = x0 + 0.4
    sum_x = x0 + 4.05

    _arrow(ax, (x0 + 0.1, yin), (split_x, yin), lw=1.0)
    _text(ax, x0 + 0.12, yin + 0.16, "in", fontsize=6, ha="left")
    ax.plot([split_x, split_x], [ymain, ysc], color="#333", lw=1.0, zorder=3)

    # shortcut
    ax.plot([split_x, x0 + 0.55], [ysc, ysc], color="#333", lw=1.0, zorder=3)
    _box(ax, (x0 + 0.55, ysc - 0.14), 2.2, 0.28, radius=0.015)
    _text(ax, x0 + 1.65, ysc, r"Conv($c_{out}$, 1, $s$) / Id", fontsize=5.5)
    ax.plot([x0 + 2.75, sum_x - 0.14], [ysc, yin], color="#333", lw=1.0, zorder=3)

    # main
    specs = [
        (x0 + 0.55, 1.15, r"Conv($c_{out}$, 3, $s$)"),
        (x0 + 1.85, 0.7, "BN+ReLU"),
        (x0 + 2.70, 0.85, r"Conv($c_{out}$, 3, 1)"),
    ]
    ax.plot([split_x, specs[0][0]], [ymain, ymain], color="#333", lw=1.0, zorder=3)
    for i, (bx, bw, lab) in enumerate(specs):
        _box(ax, (bx, ymain - 0.14), bw, 0.28, radius=0.015)
        _text(ax, bx + bw / 2, ymain, lab, fontsize=5.2)
        if i + 1 < len(specs):
            nx = specs[i + 1][0]
            _arrow(ax, (bx + bw, ymain), (nx, ymain), lw=0.8)
    last = specs[-1][0] + specs[-1][1]
    # BN then to sum
    _box(ax, (last + 0.08, ymain - 0.14), 0.4, 0.28, radius=0.015)
    _text(ax, last + 0.28, ymain, "BN", fontsize=5.5)
    ax.plot([last + 0.48, sum_x - 0.14], [ymain, yin], color="#333", lw=1.0, zorder=3)

    circ = Circle((sum_x, yin), 0.13, facecolor="#fff8e7", edgecolor="#333", lw=1.1, zorder=3)
    ax.add_patch(circ)
    _text(ax, sum_x, yin, r"$\oplus$", fontsize=9, fontweight="bold")
    _arrow(ax, (sum_x + 0.13, yin), (sum_x + 0.35, yin), lw=0.9)
    _box(ax, (sum_x + 0.35, yin - 0.14), 0.55, 0.28, radius=0.015)
    _text(ax, sum_x + 0.625, yin, "ReLU", fontsize=6)
    _arrow(ax, (sum_x + 0.9, yin), (x0 + w - 0.15, yin), lw=0.9)
    _text(ax, x0 + w - 0.12, yin + 0.16, "out", fontsize=6, ha="left")


def _draw_rap_callout(ax, origin: tuple[float, float]) -> None:
    """RAP 展开（对应 Fig.2 的 SB 角色）。"""
    x0, y0 = origin
    w, h = 5.5, 1.55
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=_C_CALL,
            edgecolor="#5a7a9a",
            linewidth=1.4,
            linestyle="--",
            zorder=1,
        )
    )
    _text(
        ax,
        x0 + w / 2,
        y0 + h - 0.18,
        r"RAP (≈ SB)  ·  $y_n\!\to\!s_n$",
        fontsize=8,
        fontweight="bold",
        color="#2c3e50",
    )
    _text(
        ax,
        x0 + w / 2,
        y0 + h - 0.38,
        r"$y_n\in\mathbb{R}^{C\times L'}\!\to\!s_n\in\mathbb{R}^{C}$, $C{=}256$",
        fontsize=5.5,
        color="#4a6080",
    )

    yin, ysc, yfeat = y0 + 0.48, y0 + 0.88, y0 + 0.22
    split_x = x0 + 0.4
    mul_x = x0 + 3.55

    _arrow(ax, (x0 + 0.1, yin), (split_x, yin), lw=1.0)
    _text(ax, x0 + 0.12, yin + 0.16, r"$y_n$", fontsize=6.5, ha="left")
    ax.plot([split_x, split_x], [yfeat, ysc], color="#333", lw=1.0, zorder=3)

    ax.plot([split_x, x0 + 0.55], [ysc, ysc], color="#333", lw=1.0, zorder=3)
    score = [
        (x0 + 0.55, 1.15, "Conv(1, 1, 1)"),
        (x0 + 1.85, 0.85, "Softmax"),
        (x0 + 2.85, 0.4, r"$\alpha$"),
    ]
    for i, (bx, bw, lab) in enumerate(score):
        _box(ax, (bx, ysc - 0.14), bw, 0.28, radius=0.015)
        _text(ax, bx + bw / 2, ysc, lab, fontsize=5.5)
        if i + 1 < len(score):
            _arrow(ax, (bx + bw, ysc), (score[i + 1][0], ysc), lw=0.8)
    ax.plot(
        [score[-1][0] + score[-1][1], mul_x - 0.14],
        [ysc, yin],
        color="#333",
        lw=1.0,
        zorder=3,
    )

    ax.plot([split_x, x0 + 0.55], [yfeat, yfeat], color="#333", lw=1.0, zorder=3)
    _box(ax, (x0 + 0.55, yfeat - 0.14), 2.3, 0.28, facecolor="#ffffff", radius=0.015)
    _text(ax, x0 + 1.7, yfeat, "identity (features)", fontsize=5.5)
    ax.plot([x0 + 2.85, mul_x - 0.14], [yfeat, yin], color="#333", lw=1.0, zorder=3)

    circ = Circle((mul_x, yin), 0.13, facecolor="#eef4fa", edgecolor="#333", lw=1.1, zorder=3)
    ax.add_patch(circ)
    _text(ax, mul_x, yin, r"$\odot$", fontsize=9, fontweight="bold")
    _arrow(ax, (mul_x + 0.13, yin), (mul_x + 0.35, yin), lw=0.9)
    _box(ax, (mul_x + 0.35, yin - 0.14), 0.55, 0.28, radius=0.015)
    _text(ax, mul_x + 0.625, yin, r"$\sum_\ell$", fontsize=6.5)
    _arrow(ax, (mul_x + 0.9, yin), (x0 + w - 0.2, yin), lw=0.9)
    _text(ax, x0 + w - 0.15, yin + 0.16, r"$s_n$", fontsize=6.5, ha="left")


def plot_stp_cnn_architecture(
    output_png: Path,
    *,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    """绘制对标 DMISC Fig.2 风格的 STP-CNN 架构图。"""
    fig, ax = plt.subplots(figsize=(15.2, 9.0))
    ax.set_xlim(0.0, 15.2)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- title (paper-like) ----
    _text(
        ax,
        7.6,
        8.72,
        "STP-CNN  (CooperativeMonostaticCNN, late + attention)",
        fontsize=13,
        fontweight="bold",
    )
    _text(
        ax,
        7.6,
        8.38,
        r"Dual-station ROI range spectra $\rightarrow$ planar position estimate "
        r"$\hat{\mathbf{p}}=(x,y)$"
        "\n"
        r"Conv$(c,k,s)$: channels, kernel, stride;  $\downarrow$: down-sampling along range "
        r"(no $\uparrow$ / reconstruction)",
        fontsize=7.5,
        color="#555555",
    )

    # ---- section frames ----
    ax.add_patch(
        Rectangle(
            (0.2, 3.55),
            9.0,
            4.45,
            facecolor=_C_ENC,
            edgecolor="#b09060",
            linewidth=1.5,
            zorder=0,
        )
    )
    _text(
        ax,
        4.7,
        7.78,
        "Spectral Encoder  (DSSE, weights shared)",
        fontsize=10,
        fontweight="bold",
        color="#5a4020",
    )

    ax.add_patch(
        Rectangle(
            (9.4, 3.55),
            5.55,
            4.45,
            facecolor=_C_DEC,
            edgecolor="#b09060",
            linewidth=1.5,
            zorder=0,
        )
    )
    _text(
        ax,
        12.15,
        7.78,
        "Joint Localization Decoder  (LFRH)",
        fontsize=10,
        fontweight="bold",
        color="#5a4020",
    )
    _text(
        ax,
        12.15,
        7.52,
        "late fusion · no DeConv / up-sampling",
        fontsize=7,
        color="#7a6548",
    )

    # ---- FEB strip ----
    _box(ax, (0.4, 7.0), 8.6, 0.42, facecolor="#f0e6d4", edgecolor="#a08050")
    _text(
        ax,
        4.7,
        7.21,
        r"FEB (non-learnable): ROI crop $[0,4]$ m  $\rightarrow$  Re/Im  "
        r"$\rightarrow$ $(B,4,L)$  $\rightarrow$ split $(B,2,L)$ per BS",
        fontsize=7,
    )

    # ---- two parallel branches ----
    branch_y = {"dev0": 5.55, "dev1": 3.85}
    labels = {
        "dev0": r"$x_0$ (BS-0)",
        "dev1": r"$x_1$ (BS-1)",
    }
    outs = {
        "dev0": r"$s_0\in\mathbb{R}^{256}$",
        "dev1": r"$s_1\in\mathbb{R}^{256}$",
    }

    for key, y0 in branch_y.items():
        # input
        _box(ax, (0.4, y0 + 0.35), 1.05, 0.55, facecolor="#ffffff", edgecolor="#888")
        _text(ax, 0.925, y0 + 0.625, labels[key], fontsize=7.5)

        # CB0 / Stem (compact)
        _box(ax, (1.65, y0 + 0.15), 1.35, 0.95)
        _text(
            ax,
            2.325,
            y0 + 0.625,
            r"CB$_0$" + "\n"
            r"Stem" + "\n"
            r"$L\!\downarrow\!4$",
            fontsize=7,
        )

        # ResBlocks compact
        for i, (lab, detail) in enumerate(
            [
                (r"Res$_1$", r"64, $s{=}1$"),
                (r"Res$_2$", r"128, $s{=}2\downarrow$"),
                (r"Res$_3$", r"256, $s{=}2\downarrow$"),
            ]
        ):
            bx = 3.2 + i * 1.35
            _box(ax, (bx, y0 + 0.25), 1.2, 0.75)
            _text(ax, bx + 0.6, y0 + 0.625, f"{lab}\n{detail}", fontsize=6.5)

        # RAP
        _box(ax, (7.35, y0 + 0.2), 0.95, 0.85, facecolor="#efe0c0")
        _text(ax, 7.825, y0 + 0.625, "RAP\n(≈SB)", fontsize=7.5, fontweight="bold")

        # arrows
        cy = y0 + 0.625
        _arrow(ax, (1.45, cy), (1.65, cy))
        _arrow(ax, (3.0, cy), (3.2, cy))
        _arrow(ax, (4.4, cy), (4.55, cy))
        _arrow(ax, (5.75, cy), (5.9, cy))
        _arrow(ax, (7.1, cy), (7.35, cy))
        _arrow(ax, (8.3, cy), (8.55, cy))

        # s_n near encoder edge
        _box(ax, (8.55, y0 + 0.35), 0.5, 0.55, facecolor="#ffffff", edgecolor="#888")
        _text(ax, 8.8, y0 + 0.625, r"$s_n$" if key == "dev0" else "", fontsize=7)
        if key == "dev0":
            _text(ax, 8.8, y0 + 0.625, r"$s_0$", fontsize=7.5)
        else:
            _text(ax, 8.8, y0 + 0.625, r"$s_1$", fontsize=7.5)

    # shared weights brace
    ax.annotate(
        "",
        xy=(2.325, 5.45),
        xytext=(2.325, 4.85),
        arrowprops=dict(arrowstyle="<->", color="#8a7040", lw=1.1),
    )
    _text(ax, 2.55, 5.15, "shared\n$\\theta$", fontsize=6.5, color="#8a7040", ha="left")

    # yn annotation on Res3
    _text(
        ax,
        6.5,
        6.55,
        r"$y_n\in\mathbb{R}^{256\times L'}$, $L'\!\approx\!L/16$",
        fontsize=6.5,
        color="#666666",
    )

    # ---- decoder side ----
    _box(ax, (9.65, 6.35), 1.55, 0.55, facecolor="#ffffff", edgecolor="#888")
    _text(ax, 10.425, 6.625, outs["dev0"], fontsize=8)
    _box(ax, (9.65, 4.35), 1.55, 0.55, facecolor="#ffffff", edgecolor="#888")
    _text(ax, 10.425, 4.625, outs["dev1"], fontsize=8)

    _arrow(ax, (9.05, branch_y["dev0"] + 0.625), (9.65, 6.625))
    _arrow(ax, (9.05, branch_y["dev1"] + 0.625), (9.65, 4.625))

    _box(ax, (11.45, 5.05), 1.45, 1.1)
    _text(
        ax,
        12.175,
        5.6,
        "Concat\n"
        r"$s=[s_0\|s_1]$" + "\n"
        r"$\in\mathbb{R}^{512}$",
        fontsize=8,
        fontweight="bold",
    )
    _arrow(ax, (11.2, 6.625), (12.175, 6.15))
    _arrow(ax, (11.2, 4.625), (12.175, 5.05))

    _box(ax, (13.15, 4.85), 1.5, 1.5)
    _text(
        ax,
        13.9,
        5.6,
        "xy_head\n"
        "Linear 512→128\n"
        "ReLU\n"
        "Dropout 0.3\n"
        "Linear 128→2",
        fontsize=7,
    )
    _arrow(ax, (12.9, 5.6), (13.15, 5.6))

    _box(ax, (13.05, 3.85), 1.7, 0.7, facecolor="#e8d4a8", edgecolor="#8a6020", linewidth=1.5)
    _text(
        ax,
        13.9,
        4.2,
        r"$\hat{\mathbf{p}}=(x,y)$" + "\nmeters",
        fontsize=9,
        fontweight="bold",
    )
    _arrow(ax, (13.9, 4.85), (13.9, 4.55))

    # ---- bottom callouts ----
    _draw_stem_callout(ax, (0.25, 0.35))
    _draw_resblock_callout(ax, (2.6, 0.35))
    _draw_rap_callout(ax, (8.2, 0.35))

    # connector hints from main modules to callouts
    _text(
        ax,
        7.6,
        3.25,
        "——  dashed boxes: internal structures (cf. Fu et al., Fig. 2 / Fig. 3)  ——",
        fontsize=6.5,
        color="#888888",
    )

    output_png = output_png.resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return output_png


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot STP-CNN architecture (DMISC Fig.2 style)"
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=_default_output_png(),
        help="output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    out = plot_stp_cnn_architecture(args.output_png, dpi=args.dpi, show=args.show)
    print(f"wrote: {out}", flush=True)


if __name__ == "__main__":
    main()
