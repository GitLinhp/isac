"""时延多普勒 / 距离-速度谱图可视化（matplotlib / plotly）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch

from ..detection.metric_mode import canonical_metric_mode
from ...utils.numerical import linear_to_db
from .sensing_performance import SensingPerformance


def linear_or_db_amplitude(
    z_grid: np.ndarray,
    cfar_grid: Optional[np.ndarray],
    to_db: bool,
    eps: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], str, str]:
    """将线性幅度网格转为显示用 ``z``；返回 ``(z_disp, cfar_disp, zlabel_mpl, ztitle_plotly)``。"""
    if to_db:
        z_disp = linear_to_db(np.maximum(z_grid, eps), is_power=True)
        cfar_disp = (
            linear_to_db(np.maximum(cfar_grid, eps), is_power=True)
            if cfar_grid is not None
            else None
        )
        zlabel_mpl = "Amplitude (dB)"
        ztitle_plotly = "Amplitude (dB)"
    else:
        z_disp = z_grid
        cfar_disp = cfar_grid
        zlabel_mpl = "Magnitude"
        ztitle_plotly = "Amplitude"
    return z_disp, cfar_disp, zlabel_mpl, ztitle_plotly


def ensure_parent_and_path(file_name: Union[Path, str]) -> Path:
    """保证父目录存在并返回 ``Path``。"""
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def prepare_surface_grids(
    sensing_performance: SensingPerformance,
    roi_slices: Tuple[int, int, int, int],
    h_abs_2d: np.ndarray,
    cfar_2d: Optional[np.ndarray],
    *,
    mode: str,
    to_db: bool,
    eps: float,
) -> Dict[str, Any]:
    """为单路 2D 谱 ``(多普勒, 时延)`` 准备 matplotlib/plotly 曲面数据。"""
    dop_start, dop_end, delay_start, delay_end = roi_slices

    if mode == "delay_doppler":
        sp = sensing_performance
        delay_axis = sp.delay_bins[delay_start:delay_end]
        doppler_axis = sp.doppler_bins[dop_start:dop_end]
        x_grid, y_grid = np.meshgrid(delay_axis, doppler_axis, indexing="xy")
        z_disp, cfar_disp, z_label, z_title_plotly = linear_or_db_amplitude(
            h_abs_2d, cfar_2d, to_db, eps
        )
        return {
            "x_mpl": x_grid,
            "y_mpl": y_grid,
            "z_mpl": z_disp,
            "cfar_mpl_z": cfar_disp,
            "x_plot": x_grid,
            "y_plot": y_grid,
            "z_plot": z_disp,
            "cfar_plot_z": cfar_disp,
            "x_label": "Delay (ns)",
            "y_label": "Doppler (Hz)",
            "z_label": z_label,
            "title_mpl": (
                "Delay-Doppler Spectrum (dB)" if to_db else "Delay-Doppler Spectrum"
            ),
            "title_plotly": "Delay-Doppler Map with CFAR Threshold",
            "z_title_plotly": z_title_plotly,
            "plotly_cfar_opacity": 0.35,
        }

    sp = sensing_performance
    range_axis = sp.range_bins[delay_start:delay_end]
    velocity_axis = sp.velocity_bins[dop_start:dop_end]
    z_disp, cfar_disp, z_label, z_title_plotly = linear_or_db_amplitude(
        h_abs_2d, cfar_2d, to_db, eps
    )
    R, V = np.meshgrid(range_axis, velocity_axis, indexing="xy")
    return {
        "x_mpl": R,
        "y_mpl": V,
        "z_mpl": z_disp,
        "cfar_mpl_z": cfar_disp,
        "x_plot": range_axis,
        "y_plot": velocity_axis,
        "z_plot": np.transpose(z_disp),
        "cfar_plot_z": np.transpose(cfar_disp) if cfar_disp is not None else None,
        "x_label": "Range (m)",
        "y_label": "Velocity (m/s)",
        "z_label": z_label,
        "title_mpl": (
            "Range-Velocity Map with CFAR Threshold"
            if cfar_disp is not None
            else "Range-Velocity Map"
        ),
        "title_plotly": "Range-Velocity Map with 2D CFAR Threshold",
        "z_title_plotly": z_title_plotly,
        "plotly_cfar_opacity": 0.7,
    }


def plot_matplotlib_3d_surface(
    ax: Any,
    grids: Dict[str, Any],
    *,
    title_suffix: str = "",
) -> None:
    """在已有 3D ``Axes`` 上绘制谱曲面。"""
    ax.plot_surface(
        grids["x_mpl"],
        grids["y_mpl"],
        grids["z_mpl"],
        cmap="viridis",
        edgecolor="none",
    )
    cfar_mpl_z = grids.get("cfar_mpl_z")
    if cfar_mpl_z is not None:
        ax.plot_surface(
            grids["x_mpl"],
            grids["y_mpl"],
            cfar_mpl_z,
            color="red",
            alpha=0.35,
            edgecolor="none",
        )
    ax.set_xlabel(grids["x_label"])
    ax.set_ylabel(grids["y_label"])
    ax.set_zlabel(grids["z_label"])
    ax.zaxis.labelpad = 2
    ax.view_init(elev=53, azim=-32)
    title = grids["title_mpl"]
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title)


def visualize_delay_doppler_spectrum(
    *,
    sensing_performance: SensingPerformance,
    h_delay_doppler: torch.Tensor,
    roi_slices: Tuple[int, int, int, int],
    file_name: Union[Path, str, None] = None,
    cfar: Optional[Union[np.ndarray, torch.Tensor]] = None,
    to_db: bool = True,
    eps: float = 1e-12,
    mode: str = "delay_doppler",
    metric_mode: Optional[str] = None,
    backend: str = "matplotlib",
    panel_labels: Optional[list[str]] = None,
    announce_save: bool = True,
) -> None:
    """渲染 DD / RV 谱图，可选叠加 CFAR 阈值面。"""
    effective_mode = metric_mode if metric_mode is not None else mode
    mode = canonical_metric_mode(effective_mode)

    backend_to_use = backend.lower()
    if backend_to_use not in {"matplotlib", "plotly"}:
        raise ValueError(
            f"Unknown backend: {backend!r}. Expected 'matplotlib' or 'plotly'."
        )
    if mode not in {"delay_doppler", "range_velocity"}:
        raise ValueError(
            f"Unknown visualize mode: {mode}. Expected 'delay_doppler' or 'range_velocity'."
        )

    h_abs_np = torch.abs(h_delay_doppler).detach().cpu().numpy()
    cfar_np = None
    if cfar is not None:
        cfar_np = (
            cfar.detach().cpu().numpy() if isinstance(cfar, torch.Tensor) else cfar
        )
        if cfar_np.ndim != h_abs_np.ndim:
            raise ValueError(
                f"cfar 与 h_delay_doppler 秩须一致，"
                f"cfar {cfar_np.shape}，谱 {h_abs_np.shape}"
            )

    if h_abs_np.ndim not in (2, 3):
        raise ValueError(
            f"h_delay_doppler 须为 2D (S,F) 或 3D (rx_num,S,F)，当前 {h_abs_np.shape}"
        )

    if h_abs_np.ndim == 3 and backend_to_use == "plotly":
        print("3D 谱 (rx_num, S, F) 暂不支持 plotly 多 RX 子图，已回退为 matplotlib")
        backend_to_use = "matplotlib"

    if backend_to_use == "matplotlib":
        if h_abs_np.ndim == 3:
            rx_num = h_abs_np.shape[0]
            fig = plt.figure(figsize=(8 * max(rx_num, 1), 10))
            if rx_num == 1:
                axes = [fig.add_subplot(111, projection="3d")]
            else:
                axes = list(
                    fig.subplots(1, rx_num, subplot_kw={"projection": "3d"}).flat
                )
            for r, ax in enumerate(axes):
                cfar_r = cfar_np[r] if cfar_np is not None else None
                grids = prepare_surface_grids(
                    sensing_performance,
                    roi_slices,
                    h_abs_np[r],
                    cfar_r,
                    mode=mode,
                    to_db=to_db,
                    eps=eps,
                )
                suffix = (
                    panel_labels[r]
                    if panel_labels is not None and r < len(panel_labels)
                    else f"RX {r}"
                )
                plot_matplotlib_3d_surface(ax, grids, title_suffix=suffix)
            fig.tight_layout()
        else:
            grids = prepare_surface_grids(
                sensing_performance,
                roi_slices,
                h_abs_np,
                cfar_np,
                mode=mode,
                to_db=to_db,
                eps=eps,
            )
            fig = plt.figure(figsize=(8, 10))
            ax = fig.add_subplot(111, projection="3d")
            plot_matplotlib_3d_surface(ax, grids)

        if file_name is not None:
            out_path = ensure_parent_and_path(file_name)
            plt.savefig(out_path)
            plt.close()
            if announce_save:
                print(f"谱图已保存: {out_path.resolve()}")
        else:
            plt.show()
        return

    if h_abs_np.ndim == 3:
        raise ValueError("plotly 仅支持 2D 谱；多 RX 请使用 backend='matplotlib'")

    grids = prepare_surface_grids(
        sensing_performance,
        roi_slices,
        h_abs_np,
        cfar_np,
        mode=mode,
        to_db=to_db,
        eps=eps,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grids["x_plot"],
            y=grids["y_plot"],
            z=grids["z_plot"],
            colorscale="Rainbow",
            name="Radar Returns",
            showscale=True,
        )
    )
    if grids["cfar_plot_z"] is not None:
        fig.add_trace(
            go.Surface(
                x=grids["x_plot"],
                y=grids["y_plot"],
                z=grids["cfar_plot_z"],
                colorscale="Viridis",
                name="CFAR Threshold",
                showscale=False,
                opacity=grids["plotly_cfar_opacity"],
            )
        )
    fig.update_layout(
        title=grids["title_plotly"],
        height=600,
        scene=dict(
            xaxis=dict(title=grids["x_label"]),
            yaxis=dict(title=grids["y_label"]),
            zaxis=dict(title=grids["z_title_plotly"]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=60, t=100),
    )
    if file_name is not None:
        out_path = ensure_parent_and_path(file_name)
        fig.write_image(str(out_path))
        if announce_save:
            print(f"谱图已保存: {out_path.resolve()}")
    else:
        fig.show()
