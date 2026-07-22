"""距离谱 ROI 向量切片 GR 块与 ``compute_range_roi`` 工具函数。

``compute_range_roi`` 供 :mod:`isac_imp.range_profile_plot` 在显示块内裁剪 ROI。
``RangeProfileRoiSliceBlock`` 仍可用于需要在 GR 链中显式输出 ROI 向量的流图；
data_collection 显示链已改用 ``RangeProfilePlotBlock``（内置 ROI + PyQtGraph）。
"""

from __future__ import annotations

import warnings

import numpy as np
from gnuradio import gr

_DEFAULT_ROI_MAX_M = 30.0


def compute_range_roi(
    *,
    range_roi: tuple[float, float],
    range_bin_step: float,
    vlen_in: int,
) -> tuple[int, int, float]:
    """返回 ``(start_bin, num_bins, x_start_m)``，保证 ROI 不越界。

    ``range_roi`` 为 ``(min_m, max_m)``。若 ``max_m <= min_m``，
    回退为 ``[0, min(30, 全谱距离上界)]``。
    """
    vlen_in = int(vlen_in)
    step = float(range_bin_step)
    if vlen_in < 1 or step <= 0:
        raise ValueError(f"vlen_in must be >= 1 and range_bin_step > 0, got {vlen_in}, {step}")

    min_m = float(range_roi[0])
    max_m = float(range_roi[1])
    if max_m <= min_m:
        warnings.warn(
            f"range_roi {range_roi!r} invalid (max <= min); "
            f"falling back to [0, {_DEFAULT_ROI_MAX_M}] m",
            stacklevel=2,
        )
        min_m = 0.0
        max_span_m = min(_DEFAULT_ROI_MAX_M, (vlen_in - 1) * step)
        max_m = max(min_m + step, max_span_m)

    start_bin = int(round(min_m / step))
    start_bin = max(0, min(start_bin, vlen_in - 1))
    num_bins = int(round((max_m - min_m) / step)) + 1
    num_bins = max(1, min(num_bins, vlen_in - start_bin))
    x_start_m = start_bin * step
    return start_bin, num_bins, x_start_m


class RangeProfileRoiSliceBlock(gr.sync_block):
    """float32 距离谱向量 → 按 bin 索引切片后的 ROI 向量。"""

    def __init__(
        self,
        vlen_in: int = 4096,
        start_bin: int = 0,
        num_bins: int = 99,
    ) -> None:
        self._vlen_in = int(vlen_in)
        start_bin, num_bins, _ = self._clamp_bins(int(start_bin), int(num_bins))

        self._start_bin = start_bin
        self._num_bins = num_bins

        gr.sync_block.__init__(
            self,
            name="Range Profile ROI Slice",
            in_sig=[(np.float32, self._vlen_in)],
            out_sig=[(np.float32, self._num_bins)],
        )

    def _clamp_bins(self, start_bin: int, num_bins: int) -> tuple[int, int, float]:
        step_hint = 1.0
        x_start = start_bin * step_hint
        if not 0 <= start_bin < self._vlen_in:
            warnings.warn(
                f"start_bin {start_bin} out of [0, {self._vlen_in}); clamping",
                stacklevel=2,
            )
            start_bin = max(0, min(start_bin, self._vlen_in - 1))
        if num_bins < 1:
            warnings.warn(f"num_bins {num_bins} < 1; using 1", stacklevel=2)
            num_bins = 1
        if start_bin + num_bins > self._vlen_in:
            warnings.warn(
                f"start_bin + num_bins ({start_bin + num_bins}) exceeds vlen_in "
                f"({self._vlen_in}); clamping num_bins",
                stacklevel=2,
            )
            num_bins = self._vlen_in - start_bin
        return start_bin, num_bins, x_start

    def set_start_bin(self, start_bin: int) -> None:
        """更新起始 bin（输出 vlen 不变，仅平移 ROI 窗口）。"""
        start_bin, num_bins, _ = self._clamp_bins(int(start_bin), self._num_bins)
        self._start_bin = start_bin
        self._num_bins = num_bins

    def set_num_bins(self, num_bins: int) -> None:
        """更新 bin 数（须与下游 vector_sink vlen 一致；改变 vlen 需重启流图）。"""
        start_bin, num_bins, _ = self._clamp_bins(self._start_bin, int(num_bins))
        self._start_bin = start_bin
        self._num_bins = num_bins

    def work(self, input_items, output_items) -> int:
        s, n = self._start_bin, self._num_bins
        output_items[0][0][:] = input_items[0][0][s : s + n]
        return 1
