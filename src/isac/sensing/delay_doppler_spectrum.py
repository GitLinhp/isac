"""时延多普勒谱处理模块"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple, Union

import plotly.graph_objs as go

from .sensing_performance import SensingPerformance
from ..utils import convert
from ..utils.numerical import linear_to_db
from ..utils.windows import apply_window


class DelayDopplerSpectrum:
    """时延多普勒谱处理类

    提供时延多普勒谱的计算、处理和可视化功能。
    """

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        delay_window: Optional[Union[str, tuple[Any, ...], Dict[str, Any]]] = None,
        doppler_window: Optional[Union[str, tuple[Any, ...], Dict[str, Any]]] = None,
        max_range_m: Optional[float] = None,
        max_velocity_mps: Optional[float] = None,
    ):
        self.sensing_performance = sensing_performance  # 感知性能
        self.device = device  # 设备
        self.h_delay_doppler: Optional[torch.Tensor] = None  # 时延多普勒谱
        self.delay_window = delay_window  # 时延窗函数
        self.doppler_window = doppler_window  # 多普勒窗函数
        self.max_range_m = max_range_m  # 最大探测距离
        self.max_velocity_mps = max_velocity_mps  # 最大探测速度
        self._roi_slices: Optional[Tuple[int, int, int, int]] = (
            None  # 时延多普勒谱 ROI 切片索引
        )

    @property
    def has_roi(self) -> bool:
        return self.max_range_m is not None and self.max_velocity_mps is not None

    def _validate_roi(self) -> None:
        if not self.has_roi:
            raise ValueError(
                "__call__ 要求配置 max_range_m / max_velocity_mps（[dd_spectrum_roi]）"
            )
        assert self.max_range_m is not None and self.max_velocity_mps is not None
        if self.max_range_m <= 0:
            raise ValueError(f"max_range_m 须为正，收到 {self.max_range_m}")
        if self.max_velocity_mps <= 0:
            raise ValueError(f"max_velocity_mps 须为正，收到 {self.max_velocity_mps}")

    def roi_delay_bins(self) -> int:
        """返回时延多普勒谱 ROI 的时延 bin 数。"""
        dr = self.sensing_performance.range_resolution  # 距离分辨率
        assert self.max_range_m is not None
        return max(1, int(self.max_range_m / dr) + 1)

    def roi_doppler_half_bins(self) -> int:
        """返回时延多普勒谱 ROI 的多普勒 bin 数。"""
        dv = self.sensing_performance.velocity_resolution  # 速度分辨率
        assert self.max_velocity_mps is not None
        return max(1, int(round(self.max_velocity_mps / dv)))

    def bin_slices(self, h_dd: torch.Tensor) -> tuple[int, int, int, int]:
        """返回 ``(dop_start, dop_end, delay_start, delay_end)`` 切片索引。"""
        self._validate_roi()
        n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
        delay_bins = min(n_delay, self.roi_delay_bins())
        dop_half = min(n_doppler // 2, self.roi_doppler_half_bins())
        dop_center = n_doppler // 2
        dop_start = max(0, dop_center - dop_half)
        dop_end = min(n_doppler, dop_center + dop_half)
        return dop_start, dop_end, 0, delay_bins

    def __call__(self, h_freq: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """计算时延多普勒谱（使用 Torch 实现）

        参数:
        ----------
        - h_freq : np.ndarray | torch.Tensor
            频域信道响应，形状为 ``(num_ofdm_symbols, fft_size)`` 或
            ``(rx_num, num_ofdm_symbols, fft_size)``。

        返回:
        ----------
        - torch.Tensor
            时延多普勒谱，末两维为 (多普勒, 时延)，与输入同秩。
        """
        h = convert(h_freq, "torch", dtype=torch.complex64, device=self.device)
        rg = self.sensing_performance.rg
        s, f = rg.num_ofdm_symbols, rg.fft_size
        if h.ndim not in (2, 3):
            raise ValueError(
                f"h_freq 须为 2D (S,F) 或 3D (rx_num,S,F)，收到 ndim={h.ndim}"
            )
        if h.shape[-2:] != (s, f):
            raise ValueError(f"h_freq 末两维须为 ({s}, {f})，收到 {tuple(h.shape)}")

        h = torch.fft.fftshift(h, dim=-1)
        h = apply_window(h, dim=-1, window=self.delay_window)
        h_delay = torch.fft.ifft(h, dim=-1, norm="ortho")
        h_delay = apply_window(h_delay, dim=-2, window=self.doppler_window)
        h_delay_doppler = torch.fft.fft(h_delay, dim=-2, norm="ortho")
        h_delay_doppler = torch.fft.fftshift(h_delay_doppler, dim=-2)

        self._validate_roi()
        dop_start, dop_end, delay_start, delay_end = self.bin_slices(h_delay_doppler)
        self._roi_slices = (dop_start, dop_end, delay_start, delay_end)
        h_delay_doppler = h_delay_doppler[..., dop_start:dop_end, delay_start:delay_end]

        self.h_delay_doppler = h_delay_doppler.to(
            device=self.device, dtype=torch.complex64
        )

        return self.h_delay_doppler

    @staticmethod
    def _linear_or_db_amplitude(
        z_grid: np.ndarray,
        cfar_grid: Optional[np.ndarray],
        to_db: bool,
        eps: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str, str]:
        """将线性幅度网格转为显示用 ``z``；可选同步转换 CFAR。

        dB 时委托 ``linear_to_db``：``is_power=True`` 对应源码中 ``factor==20``（与原先 ``20*log10`` 一致）。
        调用前先 ``np.maximum(..., eps)``，以保留 ``visualize(eps=...)`` 语义（不限于 ``linear_to_db`` 内部 clip）。

        返回 ``(z_disp, cfar_disp, zlabel_mpl, ztitle_plotly)``。
        """
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

    @staticmethod
    def _ensure_parent_and_path(file_name: Union[Path, str]) -> Path:
        """保证父目录存在并返回 ``Path``（用于保存图像）。"""
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def _prepare_surface_grids(
        self,
        h_abs_2d: np.ndarray,
        cfar_2d: Optional[np.ndarray],
        *,
        mode: str,
        to_db: bool,
        eps: float,
    ) -> Dict[str, Any]:
        """为单路 2D 谱 ``(多普勒, 时延)`` 准备 matplotlib/plotly 曲面数据。"""
        if self._roi_slices is None:
            raise ValueError(
                "_prepare_surface_grids 要求已通过 __call__ 设置 _roi_slices"
            )
        dop_start, dop_end, delay_start, delay_end = self._roi_slices

        x_label = y_label = z_label = ""
        title_mpl = ""
        title_plotly = ""
        z_title_plotly = ""
        plotly_cfar_opacity = 0.7

        if mode == "delay_doppler":
            sp = self.sensing_performance
            delay_axis = sp.delay_bins[delay_start:delay_end]
            doppler_axis = sp.doppler_bins[dop_start:dop_end]
            x_grid, y_grid = np.meshgrid(delay_axis, doppler_axis, indexing="xy")
            h_sl = h_abs_2d
            cf_sl = cfar_2d

            z_disp, cfar_disp, z_label, z_title_plotly = self._linear_or_db_amplitude(
                h_sl, cf_sl, to_db, eps
            )

            x_mpl, y_mpl, z_mpl = x_grid, y_grid, z_disp
            x_plot, y_plot, z_plot = x_grid, y_grid, z_disp
            cfar_plot_z = cfar_disp

            x_label = "Delay (ns)"
            y_label = "Doppler (Hz)"
            title_mpl = (
                "Delay-Doppler Spectrum (dB)" if to_db else "Delay-Doppler Spectrum"
            )
            title_plotly = "Delay-Doppler Map with CFAR Threshold"
            plotly_cfar_opacity = 0.35

        else:
            sp = self.sensing_performance
            range_bins = sp.range_bins
            velocity_bins = sp.velocity_bins

            range_axis = range_bins[delay_start:delay_end]
            velocity_axis = velocity_bins[dop_start:dop_end]
            h_sl = h_abs_2d
            cf_sl = cfar_2d

            z_disp, cfar_disp, z_label, z_title_plotly = self._linear_or_db_amplitude(
                h_sl, cf_sl, to_db, eps
            )

            R, V = np.meshgrid(range_axis, velocity_axis, indexing="xy")
            x_mpl, y_mpl, z_mpl = R, V, z_disp
            x_plot, y_plot = range_axis, velocity_axis
            z_plot = np.transpose(z_disp)
            cfar_plot_z = np.transpose(cfar_disp) if cfar_disp is not None else None

            x_label = "Range (m)"
            y_label = "Velocity (m/s)"
            title_mpl = (
                "Range-Velocity Map with CFAR Threshold"
                if cfar_disp is not None
                else "Range-Velocity Map"
            )
            title_plotly = "Range-Velocity Map with 2D CFAR Threshold"
            plotly_cfar_opacity = 0.7

        return {
            "x_mpl": x_mpl,
            "y_mpl": y_mpl,
            "z_mpl": z_mpl,
            "cfar_mpl_z": cfar_disp,
            "x_plot": x_plot,
            "y_plot": y_plot,
            "z_plot": z_plot,
            "cfar_plot_z": cfar_plot_z,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "title_mpl": title_mpl,
            "title_plotly": title_plotly,
            "z_title_plotly": z_title_plotly,
            "plotly_cfar_opacity": plotly_cfar_opacity,
        }

    @staticmethod
    def _plot_matplotlib_3d_surface(
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

    def visualize(
        self,
        file_name: Union[Path, str] = None,
        cfar: Optional[Union[np.ndarray, torch.Tensor]] = None,
        to_db: bool = True,
        eps: float = 1e-12,
        mode: str = "delay_doppler",
        metric_mode: Optional[str] = None,
        backend: str = "matplotlib",
        panel_labels: Optional[list[str]] = None,
        announce_save: bool = True,
    ) -> None:
        """可视化谱图（时延-多普勒 / 距离-速度），可选叠加 CFAR 阈值。

        参数
        ----------
        - cfar : np.ndarray | torch.Tensor | None
            当 `cfar` 不为 `None` 时，会在两种模式下叠加 CFAR（以 `abs(h)` 的同一单位体系）。
            两种 mode 在两种 backend 下都支持透明 CFAR 曲面叠加。
        - file_name : Path | str | None
            保存图像路径；为 None 时直接显示。
        - to_db : bool
            若为 True，将幅度转换为 dB（`20*log10(|.|)`）显示；否则显示线性幅度。
        - eps : float
            用于保护 `log10`，避免出现 `-inf`。
        - mode : str
            - `"delay_doppler"`：显示时延-多普勒谱（matplotlib 3D）
            - `"range_velocity"`：距离轴用 ``range_bins`` (m)、速度轴用 ``velocity_bins`` (m/s)，
              与 ``delay_doppler`` 的时延/多普勒 (ns/Hz) 坐标分开；谱网格与 ``h`` 的 bin 对齐不变。
        - backend : str
            渲染后端：``"matplotlib"``（3D 曲面）或 ``"plotly"``（交互 3D）。
        - announce_save : bool
            保存到 ``file_name`` 时是否打印保存路径，默认 ``True``。
        """

        if not hasattr(self, "h_delay_doppler"):
            raise ValueError("时延多普勒谱数据未计算，请先调用 __call__ 方法")
        if metric_mode is not None:
            mode = metric_mode
        mode_aliases = {"dd": "delay_doppler", "rv": "range_velocity"}
        mode = mode_aliases.get(mode.strip().lower(), mode)
        # ----------------------
        # 1) 准备区：先根据 mode 准备好 x/y/z/cfar（避免 mode×backend 四象限）
        # ----------------------
        backend_to_use = backend.lower()
        if backend_to_use not in {"matplotlib", "plotly"}:
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'matplotlib' or 'plotly'."
            )

        if mode not in {"delay_doppler", "range_velocity"}:
            raise ValueError(
                f"Unknown visualize mode: {mode}. Expected 'delay_doppler' or 'range_velocity'."
            )

        h_abs_np = torch.abs(self.h_delay_doppler).detach().cpu().numpy()
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
            print(
                "3D 谱 (rx_num, S, F) 暂不支持 plotly 多 RX 子图，已回退为 matplotlib"
            )
            backend_to_use = "matplotlib"

        if self._roi_slices is None:
            raise ValueError("visualize 要求先通过 __call__ 计算并裁剪 DD 谱")

        # ----------------------
        # 2) 渲染区
        # ----------------------
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
                    grids = self._prepare_surface_grids(
                        h_abs_np[r],
                        cfar_r,
                        mode=mode,
                        to_db=to_db,
                        eps=eps,
                    )
                    if panel_labels is not None and r < len(panel_labels):
                        suffix = panel_labels[r]
                    else:
                        suffix = f"RX {r}"
                    self._plot_matplotlib_3d_surface(ax, grids, title_suffix=suffix)
                fig.tight_layout()
            else:
                grids = self._prepare_surface_grids(
                    h_abs_np,
                    cfar_np,
                    mode=mode,
                    to_db=to_db,
                    eps=eps,
                )
                fig = plt.figure(figsize=(8, 10))
                ax = fig.add_subplot(111, projection="3d")
                self._plot_matplotlib_3d_surface(ax, grids)

            if file_name is not None:
                out_path = self._ensure_parent_and_path(file_name)
                plt.savefig(out_path)
                plt.close()
                if announce_save:
                    print(f"谱图已保存: {out_path.resolve()}")
            else:
                plt.show()
            return

        if backend_to_use == "plotly":
            if h_abs_np.ndim == 3:
                raise ValueError(
                    "plotly 仅支持 2D 谱；多 RX 请使用 backend='matplotlib'"
                )
            grids = self._prepare_surface_grids(
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
                out_path = self._ensure_parent_and_path(file_name)
                fig.write_image(str(out_path))
                if announce_save:
                    print(f"谱图已保存: {out_path.resolve()}")
            else:
                fig.show()
            return

        raise ValueError(
            f"Unknown backend: {backend_to_use}. Expected 'matplotlib' or 'plotly'."
        )
