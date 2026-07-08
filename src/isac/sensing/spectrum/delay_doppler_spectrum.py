"""时延多普勒谱计算与可视化：频域信道 → 2D DD 谱 + ROI 裁切 + 3D 出图。

典型流水线：``h_freq`` → :meth:`DelayDopplerSpectrum.__call__` → 裁切 ``h_delay_doppler``
→ 可选 :meth:`DelayDopplerSpectrum.visualize` → 下游 MUSIC 检峰。

**张量约定**

- 输入 ``h_freq`` 末两维 ``(S, F) = (num_ofdm_symbols, fft_size)``；可为
  ``(S, F)`` 或 ``(rx_num, S, F)``。
- 输出 / 缓存 ``h_delay_doppler`` 末两维 ``(多普勒, 时延)``，ROI 裁切后尺寸缩小。
- ``h_abs_2d`` 与 ``h_delay_doppler`` 对齐，形状 ``(n_doppler_roi, n_delay_roi)``。

**FFT 链**（末两维）：子载波 ``fftshift`` → delay 窗 → IFFT（时延维）；
符号 doppler 窗 → FFT → ``fftshift``（多普勒维）。

**ROI**：由 TOML ``[dd_spectrum_roi]`` 的 ``max_range_m`` / ``max_velocity_mps`` 配置，
经 :class:`~isac.sensing.metric.SpectrumMetric` 按**单基地径向**尺度换算 bin 并裁切。

**展示域**

- ``metric_mode='dd'``：时延 (ns) × 多普勒 (Hz) 轴。
- ``metric_mode='rv'``：距离/速度轴，物理尺度由 ``sens_mode`` 选单/双基地
  （``monostatic`` / ``bistatic``）；不改变 FFT，仅影响轴标定与出图标题。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch

from ..metric import SpectrumMetric
from ...data_structures.types import MetricMode, RoiSlices, SensMode
from .sensing_performance import SensingPerformance
from ...utils import convert
from ...utils.numerical import linear_to_db
from ...utils.windows import apply_window


@dataclass
class SurfaceGrids:
    """单路 2D 谱的 matplotlib / plotly 3D 曲面数据。

    由 :meth:`DelayDopplerSpectrum._prepare_surface_grids` 构造，供两种后端共用。

    Attributes
    ----------
    x_mpl, y_mpl, z_mpl :
        matplotlib ``plot_surface`` 用的 meshgrid 坐标与幅度网格（dd/rv 均为 2D）。
    cfar_mpl_z :
        CFAR 阈值面的 z 网格；``None`` 表示不叠加阈值面。
    x_plot, y_plot, z_plot :
        plotly ``go.Surface`` 坐标。dd 模式与 mpl 相同（meshgrid）；rv 模式下
        ``x_plot``/``y_plot`` 为 1D 轴向量，``z_plot`` 转置为 ``(len(y), len(x))``。
    x_label, y_label, z_label :
        坐标轴标签（物理单位随 ``metric_mode`` 变化）。
    title_mpl, title_plotly :
        各后端图标题。
    z_title_plotly :
        plotly 场景 z 轴标题。
    plotly_cfar_opacity :
        plotly CFAR 阈值面透明度。
    """

    x_mpl: np.ndarray
    y_mpl: np.ndarray
    z_mpl: np.ndarray
    cfar_mpl_z: Optional[np.ndarray]
    x_plot: Any
    y_plot: Any
    z_plot: np.ndarray
    cfar_plot_z: Optional[np.ndarray]
    x_label: str
    y_label: str
    z_label: str
    title_mpl: str
    title_plotly: str
    z_title_plotly: str
    plotly_cfar_opacity: float


class DelayDopplerSpectrum:
    """时延多普勒谱计算与 3D 可视化。

    对实例调用 ``dd(h_freq)`` 完成 FFT 变换与 ROI 裁切，结果缓存在 ``h_delay_doppler``；
    ``_roi_slices`` 记录裁切索引 ``(dop_start, dop_end, delay_start, delay_end)``，
    供 :meth:`visualize` 与 :class:`~isac.sensing.metric.SpectrumMetric` 生成物理坐标轴。

    **须配置 ROI**：``max_range_m`` 与 ``max_velocity_mps`` 均非 ``None`` 时
    :meth:`__call__` 才会执行裁切（通常来自 TOML ``[dd_spectrum_roi]``）。
    ROI 物理量按单基地径向距离 / 速度解释，与 ``SpectrumMetric.roi_*`` 一致。

    **典型调用**：``h_dd = dd(h_freq)`` → ``dd.visualize(file_name=..., metric_mode=...)``

    **可视化后端**：``matplotlib`` 支持单谱或 ``(rx_num, S, F)`` 多 RX 1×N 子图；
    ``plotly`` 仅 2D 单谱（多 RX 时 :meth:`_resolve_backend` 自动回退 matplotlib）。
    """

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        delay_window: Optional[Union[str, tuple, dict]] = None,
        doppler_window: Optional[Union[str, tuple, dict]] = None,
        max_range_m: Optional[float] = None,
        max_velocity_mps: Optional[float] = None,
    ):
        """初始化时延多普勒谱计算器。

        参数
        ----
        sensing_performance :
            提供 ``ResourceGrid``、分辨率与物理轴标定（``SensingPerformance``）。
        device :
            FFT 与缓存张量的目标设备。
        delay_window :
            时延维（子载波 / dim=-1）窗函数配置，传给 :func:`~isac.utils.windows.apply_window`。
        doppler_window :
            多普勒维（符号 / dim=-2）窗函数配置。
        max_range_m :
            ROI 时延轴上界 (m)，从 0 起；``None`` 表示未配置 ROI。
        max_velocity_mps :
            ROI 多普勒半幅 (m/s)，零多普勒中心 ±；``None`` 表示未配置 ROI。
        """
        self.sensing_performance = sensing_performance
        self._metric = SpectrumMetric(sensing_performance)
        self.device = device
        self.delay_window = delay_window
        self.doppler_window = doppler_window
        self.max_range_m = max_range_m
        self.max_velocity_mps = max_velocity_mps
        self._roi_slices: Optional[RoiSlices] = None

    # ==================== ROI 配置与切片 ====================
    @property
    def has_roi(self) -> bool:
        """是否已同时配置 ``max_range_m`` 与 ``max_velocity_mps``。"""
        return self.max_range_m is not None and self.max_velocity_mps is not None

    def _validate_roi(self) -> None:
        """校验 ROI 已配置且物理量为正。"""
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
        """``max_range_m`` 对应的时延维 bin 数（含零时延 bin）。

        计数规则：``max(1, int(max_range_m / Δr_mono) + 1)``，
        ``Δr_mono = range_resolution_monostatic``。
        """
        assert self.max_range_m is not None
        return self._metric.roi_delay_bin_count(self.max_range_m)

    def roi_doppler_half_bins(self) -> int:
        """``max_velocity_mps`` 对应的多普勒半宽 bin 数（零多普勒两侧各 ``dop_half``）。

        计数规则：``max(1, round(max_velocity_mps / Δv_mono))``。
        """
        assert self.max_velocity_mps is not None
        return self._metric.roi_doppler_half_bins(self.max_velocity_mps)

    def bin_slices(self, h_dd: torch.Tensor) -> RoiSlices:
        """由 ROI 物理量与全尺寸谱形状计算裁切切片。

        参数
        ----
        h_dd :
            全 FFT 尺寸 DD 谱，末两维 ``(n_doppler_full, n_delay_full)``。

        返回
        ----
        RoiSlices
            ``(dop_start, dop_end, delay_start, delay_end)``，Python 切片语义
            ``h_dd[..., dop_start:dop_end, delay_start:delay_end]``。
        """
        self._validate_roi()
        assert self.max_range_m is not None and self.max_velocity_mps is not None
        n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
        return self._metric.bin_slices(
            n_doppler,
            n_delay,
            self.max_range_m,
            self.max_velocity_mps,
        )

    # ==================== FFT 变换 ====================
    def _transform_freq_to_dd(self, h: torch.Tensor) -> torch.Tensor:
        """频域资源网格 → 全尺寸时延-多普勒谱。

        变换链（末两维 ``(..., S, F)`` → ``(..., S, F)``，语义变为多普勒×时延）：

        1. ``fftshift(dim=-1)``：子载波频轴居中。
        2. delay 窗 + ``ifft(dim=-1)``：子载波 IFFT → **时延维**（列 / 末维）。
        3. doppler 窗 + ``fft(dim=-2)``：符号 FFT → **多普勒维**（行 / 倒数第二维）。
        4. ``fftshift(dim=-2)``：多普勒 bin 以零多普勒为中心。
        """
        h = torch.fft.fftshift(h, dim=-1)
        h = apply_window(h, dim=-1, window=self.delay_window)
        h_delay = torch.fft.ifft(h, dim=-1, norm="ortho")
        h_delay = apply_window(h_delay, dim=-2, window=self.doppler_window)
        h_dd = torch.fft.fft(h_delay, dim=-2, norm="ortho")
        return torch.fft.fftshift(h_dd, dim=-2)

    def __call__(self, h_freq: torch.Tensor) -> torch.Tensor:
        """频域信道 → ROI 裁切后的时延多普勒谱。

        参数
        ----
        h_freq :
            LS 估计等得到的频域信道，``complex``，形状 ``(S, F)`` 或 ``(rx_num, S, F)``，
            其中 ``S = rg.num_ofdm_symbols``，``F = rg.fft_size``。

        返回
        ----
        torch.Tensor
            裁切后的复数 DD 谱，末两维 ``(n_doppler_roi, n_delay_roi)``；
            同时写入实例属性 ``h_delay_doppler`` 与 ``_roi_slices``。
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

        h_delay_doppler = self._transform_freq_to_dd(h)

        self._validate_roi()
        dop_start, dop_end, delay_start, delay_end = self.bin_slices(h_delay_doppler)
        self._roi_slices = (dop_start, dop_end, delay_start, delay_end)
        h_delay_doppler = h_delay_doppler[..., dop_start:dop_end, delay_start:delay_end]

        self.h_delay_doppler = h_delay_doppler.to(
            device=self.device, dtype=torch.complex64
        )
        return self.h_delay_doppler

    # ==================== 可视化入口 ====================
    def visualize(
        self,
        file_name: Union[Path, str, None] = None,
        cfar: Optional[Union[np.ndarray, torch.Tensor]] = None,
        to_db: bool = True,
        eps: float = 1e-12,
        metric_mode: MetricMode = "dd",
        backend: str = "matplotlib",
        panel_labels: Optional[list[str]] = None,
        sens_mode: SensMode = "monostatic",
    ) -> None:
        """渲染最近一次 :meth:`__call__` 得到的 DD/RV 谱 3D 曲面图。

        须先调用 :meth:`__call__` 以填充 ``h_delay_doppler`` 与 ``_roi_slices``。

        参数
        ----
        file_name :
            输出 PNG 路径；``None`` 时调用 ``show()`` 交互显示。
        cfar :
            与 ``h_delay_doppler`` 同形状的 CFAR 阈值幅度；可选，叠加红色阈值面。
        to_db :
            ``True`` 时 z 轴用功率 dB；``False`` 用线性幅度。
        eps :
            转 dB 前的幅度下限，避免 ``log(0)``。
        metric_mode :
            ``"dd"`` 时延-多普勒轴；``"rv"`` 距离-速度轴。
        backend :
            ``"matplotlib"`` 或 ``"plotly"``；多 RX 时 plotly 会回退 matplotlib。
        panel_labels :
            多 RX 子图标题后缀；缺省为 ``"RX {r}"``。
        sens_mode :
            仅 ``metric_mode='rv'`` 时生效，选择单/双基地物理轴
            （``range_bins_{sens_mode}`` / ``velocity_bins_{sens_mode}``）。
        """
        self._require_cached_spectrum()
        h_abs_np, cfar_np = self._abs_numpy_for_viz(cfar)
        backend_to_use = self._resolve_backend(backend, h_abs_np.ndim)
        viz_kw = dict(
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            to_db=to_db,
            eps=eps,
        )
        if backend_to_use == "matplotlib":
            self._visualize_matplotlib(
                h_abs_np,
                cfar_np,
                file_name=file_name,
                panel_labels=panel_labels,
                **viz_kw,
            )
        else:
            self._visualize_plotly(
                h_abs_np,
                cfar_np,
                file_name=file_name,
                **viz_kw,
            )

    # ==================== 可视化辅助 ====================
    def _require_cached_spectrum(self) -> None:
        """校验已调用 :meth:`__call__` 并缓存谱与 ROI 切片。"""
        if not hasattr(self, "h_delay_doppler"):
            raise ValueError("时延多普勒谱数据未计算，请先调用 __call__ 方法")
        if self._roi_slices is None:
            raise ValueError("visualize 要求先通过 __call__ 计算并裁剪 DD 谱")

    def _abs_numpy_for_viz(
        self,
        cfar: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """取谱幅度并转 CPU numpy，供可视化后端使用。

        返回
        ----
        h_abs_np :
            ``|h_delay_doppler|``，形状 ``(n_dop, n_delay)`` 或 ``(rx_num, n_dop, n_delay)``。
        cfar_np :
            与 ``h_abs_np`` 同秩的 CFAR 阈值；``cfar is None`` 时为 ``None``。
        """
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
        return h_abs_np, cfar_np

    @staticmethod
    def _resolve_backend(backend: str, ndim: int) -> str:
        """规范化 backend 字符串；``ndim==3`` 且 plotly 时回退 matplotlib。

        参数
        ----
        ndim :
            ``h_abs_np.ndim``，2 为单 RX，3 为多 RX。
        """
        backend_to_use = backend.lower()
        if backend_to_use not in {"matplotlib", "plotly"}:
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'matplotlib' or 'plotly'."
            )
        if ndim == 3 and backend_to_use == "plotly":
            print("3D 谱 (rx_num, S, F) 暂不支持 plotly 多 RX 子图，已回退为 matplotlib")
            return "matplotlib"
        return backend_to_use

    def _grids(
        self,
        h_abs_2d: np.ndarray,
        cfar_2d: Optional[np.ndarray],
        *,
        metric_mode: MetricMode,
        sens_mode: SensMode,
        to_db: bool,
        eps: float,
    ) -> SurfaceGrids:
        """为单路 2D 谱准备曲面数据（绑定实例 ``_metric`` 与 ``_roi_slices``）。

        参数
        ----
        h_abs_2d, cfar_2d :
            形状 ``(n_doppler_roi, n_delay_roi)`` 的幅度与可选 CFAR 阈值。
        """
        assert self._roi_slices is not None
        return self._prepare_surface_grids(
            self._metric,
            self._roi_slices,
            h_abs_2d,
            cfar_2d,
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            to_db=to_db,
            eps=eps,
        )

    def _visualize_matplotlib(
        self,
        h_abs_np: np.ndarray,
        cfar_np: Optional[np.ndarray],
        *,
        file_name: Union[Path, str, None],
        panel_labels: Optional[list[str]],
        metric_mode: MetricMode,
        sens_mode: SensMode,
        to_db: bool,
        eps: float,
    ) -> None:
        """matplotlib 3D 曲面：单谱 ``(n_dop, n_delay)`` 或 1×N 多 RX 子图。"""
        grid_kw = dict(
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            to_db=to_db,
            eps=eps,
        )
        if h_abs_np.ndim == 3:
            rx_num = h_abs_np.shape[0]
            fig = plt.figure(figsize=(8 * max(rx_num, 1), 10))
            if rx_num == 1:
                axes = [fig.add_subplot(111, projection="3d")]
            else:
                subfig = fig.subplots(1, rx_num, subplot_kw={"projection": "3d"})
                axes: list[Any] = (
                    subfig.ravel().tolist()
                    if isinstance(subfig, np.ndarray)
                    else [subfig]
                )
            for r, ax in enumerate(axes):
                cfar_r = cfar_np[r] if cfar_np is not None else None
                grids = self._grids(h_abs_np[r], cfar_r, **grid_kw)
                suffix = (
                    panel_labels[r]
                    if panel_labels is not None and r < len(panel_labels)
                    else f"RX {r}"
                )
                self._plot_matplotlib_3d_surface(ax, grids, title_suffix=suffix)
            fig.tight_layout()
        else:
            grids = self._grids(h_abs_np, cfar_np, **grid_kw)
            fig = plt.figure(figsize=(8, 10))
            ax = fig.add_subplot(111, projection="3d")
            self._plot_matplotlib_3d_surface(ax, grids)

        if file_name is not None:
            out_path = self._ensure_parent_and_path(file_name)
            plt.savefig(out_path)
            plt.close()
            print(f"谱图已保存: {out_path.resolve()}\n")
        else:
            plt.show()

    def _visualize_plotly(
        self,
        h_abs_np: np.ndarray,
        cfar_np: Optional[np.ndarray],
        *,
        file_name: Union[Path, str, None],
        metric_mode: MetricMode,
        sens_mode: SensMode,
        to_db: bool,
        eps: float,
    ) -> None:
        """plotly 3D 曲面（仅 2D 单谱 ``h_abs_np.ndim == 2``）。"""
        if h_abs_np.ndim == 3:
            raise ValueError("plotly 仅支持 2D 谱；多 RX 请使用 backend='matplotlib'")

        grids = self._grids(
            h_abs_np,
            cfar_np,
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            to_db=to_db,
            eps=eps,
        )
        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=grids.x_plot,
                y=grids.y_plot,
                z=grids.z_plot,
                colorscale="Rainbow",
                name="Radar Returns",
                showscale=True,
            )
        )
        if grids.cfar_plot_z is not None:
            fig.add_trace(
                go.Surface(
                    x=grids.x_plot,
                    y=grids.y_plot,
                    z=grids.cfar_plot_z,
                    colorscale="Viridis",
                    name="CFAR Threshold",
                    showscale=False,
                    opacity=grids.plotly_cfar_opacity,
                )
            )
        fig.update_layout(
            title=grids.title_plotly,
            height=600,
            scene=dict(
                xaxis=dict(title=grids.x_label),
                yaxis=dict(title=grids.y_label),
                zaxis=dict(title=grids.z_title_plotly),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            margin=dict(l=0, r=0, b=60, t=100),
        )
        if file_name is not None:
            out_path = self._ensure_parent_and_path(file_name)
            fig.write_image(str(out_path))
            print(f"谱图已保存: {out_path.resolve()}\n")
        else:
            fig.show()

    @staticmethod
    def _linear_or_db_amplitude(
        z_grid: np.ndarray,
        cfar_grid: Optional[np.ndarray],
        to_db: bool,
        eps: float,
    ) -> tuple[np.ndarray, Optional[np.ndarray], str, str]:
        """线性幅度网格 → 显示用 z 及轴标签。

        ``z_grid`` 形状 ``(n_doppler, n_delay)``，与 ``h_abs_2d`` 一致。
        """
        if to_db:
            z_disp = np.asarray(linear_to_db(np.maximum(z_grid, eps), is_power=True))
            cfar_disp = (
                np.asarray(linear_to_db(np.maximum(cfar_grid, eps), is_power=True))
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
        """创建输出文件的父目录并返回 ``Path``。"""
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    @staticmethod
    def _prepare_surface_grids(
        metric: SpectrumMetric,
        roi_slices: RoiSlices,
        h_abs_2d: np.ndarray,
        cfar_2d: Optional[np.ndarray],
        *,
        metric_mode: MetricMode,
        sens_mode: SensMode,
        to_db: bool,
        eps: float,
    ) -> SurfaceGrids:
        """为单路 2D 谱 ``(多普勒, 时延)`` 准备 matplotlib/plotly 曲面数据。

        dd 模式：``meshgrid(x_axis, y_axis)`` 供 mpl/plotly 共用。
        rv 模式：mpl 仍用 meshgrid；plotly ``Surface`` 要求 ``z.shape == (len(y), len(x))``，
        故 ``z_plot``/``cfar_plot_z`` 相对 ``h_abs_2d`` 转置，``x_plot``/``y_plot`` 改为 1D 轴向量。
        """
        x_axis, y_axis, x_label, y_label = metric.axes_for_roi(
            roi_slices, metric_mode, sens_mode
        )
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
        z_disp, cfar_disp, z_label, z_title_plotly = (
            DelayDopplerSpectrum._linear_or_db_amplitude(
                h_abs_2d, cfar_2d, to_db, eps
            )
        )

        base = SurfaceGrids(
            x_mpl=x_grid,
            y_mpl=y_grid,
            z_mpl=z_disp,
            cfar_mpl_z=cfar_disp,
            x_plot=x_grid,
            y_plot=y_grid,
            z_plot=z_disp,
            cfar_plot_z=cfar_disp,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            title_mpl="",
            title_plotly="",
            z_title_plotly=z_title_plotly,
            plotly_cfar_opacity=0.35,
        )

        if metric_mode == "dd":
            return replace(
                base,
                title_mpl=(
                    "Delay-Doppler Spectrum (dB)" if to_db else "Delay-Doppler Spectrum"
                ),
                title_plotly="Delay-Doppler Map with CFAR Threshold",
            )

        return replace(
            base,
            x_plot=x_axis,
            y_plot=y_axis,
            z_plot=np.transpose(z_disp),
            cfar_plot_z=np.transpose(cfar_disp) if cfar_disp is not None else None,
            title_mpl=(
                "Range-Velocity Map with CFAR Threshold"
                if cfar_disp is not None
                else "Range-Velocity Map"
            ),
            title_plotly="Range-Velocity Map with 2D CFAR Threshold",
            plotly_cfar_opacity=0.7,
        )

    @staticmethod
    def _plot_matplotlib_3d_surface(
        ax: Any,
        grids: SurfaceGrids,
        *,
        title_suffix: str = "",
    ) -> None:
        """在已有 3D Axes 上绘制谱曲面与可选 CFAR 阈值面。

        主曲面用 viridis  colormap；CFAR 阈值为半透明红色叠加面（``alpha=0.35``），
        便于对比检峰门限与回波幅度。
        """
        ax.plot_surface(
            grids.x_mpl,
            grids.y_mpl,
            grids.z_mpl,
            cmap="viridis",
            edgecolor="none",
        )
        if grids.cfar_mpl_z is not None:
            ax.plot_surface(
                grids.x_mpl,
                grids.y_mpl,
                grids.cfar_mpl_z,
                color="red",
                alpha=0.35,
                edgecolor="none",
            )
        ax.set_xlabel(grids.x_label)
        ax.set_ylabel(grids.y_label)
        ax.set_zlabel(grids.z_label)
        ax.zaxis.labelpad = 2
        ax.view_init(elev=53, azim=-32)
        title = grids.title_mpl
        if title_suffix:
            title = f"{title} — {title_suffix}"
        ax.set_title(title)
