"""时延多普勒谱计算模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .delay_doppler_visualize import visualize_delay_doppler_spectrum
from .sensing_performance import SensingPerformance
from ...utils import convert
from ...utils.windows import apply_window


def compute_dd_roi_slices(
    sensing_performance: SensingPerformance,
    max_range_m: float,
    max_velocity_mps: float,
    n_doppler: int,
    n_delay: int,
) -> tuple[int, int, int, int]:
    """由 ROI 物理量与全谱尺寸计算 ``(dop_start, dop_end, delay_start, delay_end)``。"""
    if max_range_m <= 0:
        raise ValueError(f"max_range_m 须为正，收到 {max_range_m}")
    if max_velocity_mps <= 0:
        raise ValueError(f"max_velocity_mps 须为正，收到 {max_velocity_mps}")
    dr = sensing_performance.range_resolution
    dv = sensing_performance.velocity_resolution
    delay_bins = min(n_delay, max(1, int(max_range_m / dr) + 1))
    dop_half = min(n_doppler // 2, max(1, int(round(max_velocity_mps / dv))))
    dop_center = n_doppler // 2
    dop_start = max(0, dop_center - dop_half)
    dop_end = min(n_doppler, dop_center + dop_half)
    return dop_start, dop_end, 0, delay_bins


class DelayDopplerSpectrum:
    """时延多普勒谱计算；可视化委托 :mod:`delay_doppler_visualize`。"""

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
        self.sensing_performance = sensing_performance
        self.device = device
        self.delay_window = delay_window
        self.doppler_window = doppler_window
        self.max_range_m = max_range_m
        self.max_velocity_mps = max_velocity_mps
        self._roi_slices: Optional[Tuple[int, int, int, int]] = None

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
        dr = self.sensing_performance.range_resolution
        assert self.max_range_m is not None
        return max(1, int(self.max_range_m / dr) + 1)

    def roi_doppler_half_bins(self) -> int:
        dv = self.sensing_performance.velocity_resolution
        assert self.max_velocity_mps is not None
        return max(1, int(round(self.max_velocity_mps / dv)))

    def bin_slices(self, h_dd: torch.Tensor) -> tuple[int, int, int, int]:
        self._validate_roi()
        assert self.max_range_m is not None and self.max_velocity_mps is not None
        n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
        return compute_dd_roi_slices(
            self.sensing_performance,
            self.max_range_m,
            self.max_velocity_mps,
            n_doppler,
            n_delay,
        )

    def __call__(self, h_freq: torch.Tensor) -> torch.Tensor:
        """频域信道 → 时延多普勒谱（末两维：多普勒 × 时延）。"""
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

    def visualize(
        self,
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
        """可视化谱图；实现见 :func:`~.delay_doppler_visualize.visualize_delay_doppler_spectrum`。"""
        if not hasattr(self, "h_delay_doppler"):
            raise ValueError("时延多普勒谱数据未计算，请先调用 __call__ 方法")
        if self._roi_slices is None:
            raise ValueError("visualize 要求先通过 __call__ 计算并裁剪 DD 谱")
        visualize_delay_doppler_spectrum(
            sensing_performance=self.sensing_performance,
            h_delay_doppler=self.h_delay_doppler,
            roi_slices=self._roi_slices,
            file_name=file_name,
            cfar=cfar,
            to_db=to_db,
            eps=eps,
            mode=mode,
            metric_mode=metric_mode,
            backend=backend,
            panel_labels=panel_labels,
            announce_save=announce_save,
        )
