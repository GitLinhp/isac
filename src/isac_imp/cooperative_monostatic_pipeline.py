"""Cooperative monostatic 离线 DSP：divide CPI → 距离谱 → MUSIC / ESPRIT。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from gnuradio.fft import window

from isac.sensing.detection.cfar import CFARDetector
from isac.sensing.detection.range_esprit_estimator import RangeEspritEstimator
from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac.sensing.localization import localize_xy_two_monostatic_ranges

DEFAULT_FFT_LEN = 2048
DEFAULT_ZEROPADDING_FAC = 4
DEFAULT_TRANSPOSE_LEN = 4
DEFAULT_SUBCARRIER_SPACING_HZ = 120e3
DEFAULT_RANGE_ROI = (0.0, 4.0)
DEFAULT_MUSIC_NUM_SOURCES = 1
DEFAULT_MUSIC_SUBARRAY_SIZE = 16
DEFAULT_MUSIC_THRESHOLD = 0.1
DEFAULT_ESPRIT_NUM_SOURCES = 1
DEFAULT_ESPRIT_SUBARRAY_SIZE = 16
DEFAULT_ESPRIT_WINDOW_SIZE = 32
DEFAULT_CFAR_TYPE = "ca"
DEFAULT_CFAR_GUARD = 2
DEFAULT_CFAR_TRAILING = 4
DEFAULT_CFAR_PFA = 1e-4
DEFAULT_CFAR_DETECTOR = "linear"


def cooperative_range_bin_step_m(
    *,
    fft_len: int = DEFAULT_FFT_LEN,
    subcarrier_spacing_hz: float = DEFAULT_SUBCARRIER_SPACING_HZ,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
) -> float:
    """与 GRC ``range_bin_step`` 一致：``c / (2 * fft_len * scs * zeropadding_fac)``。"""
    return 3e8 / (2 * int(fft_len) * float(subcarrier_spacing_hz) * int(zeropadding_fac))


def divide_cpi_to_complex_range_profile(
    cpi_flat: Sequence[complex] | np.ndarray,
    *,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    transpose_len: int = DEFAULT_TRANSPOSE_LEN,
) -> np.ndarray:
    """将 flatten divide CPI 转为与 ``OfdmRangeProfileBlock`` out1 等价的 CPI 复数距离谱。"""
    vlen_range = int(fft_len) * int(zeropadding_fac)
    expected = vlen_range * int(transpose_len)
    flat = np.asarray(cpi_flat, dtype=np.complex64).reshape(-1)
    if flat.size != expected:
        raise ValueError(
            f"expected divide CPI length {expected}, got {flat.size}"
        )

    divide_buf = flat.reshape(int(transpose_len), vlen_range)
    bh_window = np.asarray(window.blackmanharris(vlen_range), dtype=np.float32)
    complex_acc = np.zeros(vlen_range, dtype=np.complex128)
    for symbol in divide_buf:
        h_win = symbol * bh_window
        complex_acc += np.fft.fft(h_win).astype(np.complex128, copy=False)
    return complex_acc.astype(np.complex64, copy=False)


def _cfar_input_from_magnitude(
    magnitude: np.ndarray | torch.Tensor,
    *,
    detector: str,
) -> torch.Tensor:
    """按 CFAR detector 类型从幅度谱构造实数检测输入。"""
    mag = torch.as_tensor(magnitude, dtype=torch.float32)
    if detector == "squarelaw":
        return mag.square()
    if detector == "linear":
        return mag
    raise ValueError(f"unknown CFAR detector: {detector!r}")


def default_range_cfar_detector(
    *,
    cfar_type: str = DEFAULT_CFAR_TYPE,
    guard: int = DEFAULT_CFAR_GUARD,
    trailing: int = DEFAULT_CFAR_TRAILING,
    pfa: float = DEFAULT_CFAR_PFA,
    detector: str = DEFAULT_CFAR_DETECTOR,
    k: int | None = None,
    offset: float | None = None,
) -> CFARDetector:
    """1D 距离谱 CFAR 检测器默认参数。

    短 ROI（约 34 bin @ 0–5 m）建议 ``2*(guard+trailing)+1 <= num_roi_bins/2``。
    """
    return CFARDetector(
        cfar_type=cfar_type,
        guard=guard,
        trailing=trailing,
        pfa=pfa,
        detector=detector,
        k=k,
        offset=offset,
    )


def compute_1d_cfar_threshold(
    magnitude: np.ndarray,
    cfar_detector: CFARDetector,
) -> np.ndarray:
    """对 1D 幅度谱计算 CFAR 阈值面。"""
    cfar_input = _cfar_input_from_magnitude(magnitude, detector=cfar_detector.detector)
    threshold = cfar_detector(cfar_input, mode="1d")
    if isinstance(threshold, torch.Tensor):
        return threshold.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(threshold, dtype=np.float64)


def _roi_magnitude_from_profile(
    profile_complex: Sequence[complex] | np.ndarray,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float],
) -> np.ndarray:
    from isac_imp.range_profile_roi_slice import compute_range_roi

    profile = np.asarray(profile_complex, dtype=np.complex64).reshape(-1)
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=float(range_bin_step),
        vlen_in=profile.size,
    )
    spectrum = profile[start_bin : start_bin + num_bins]
    return np.abs(spectrum).astype(np.float64, copy=False)


def estimate_monostatic_range_m(
    profile_complex: Sequence[complex] | np.ndarray,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    num_sources: int = DEFAULT_MUSIC_NUM_SOURCES,
    subarray_size: int = DEFAULT_MUSIC_SUBARRAY_SIZE,
    threshold: float = DEFAULT_MUSIC_THRESHOLD,
    estimator: RangeMusicEstimator | None = None,
    cfar_detector: CFARDetector | None = None,
) -> float:
    """对 CPI 复数距离谱做 1D MUSIC，返回最强峰距离 (m)；无峰时 ``nan``。"""
    est = estimator or RangeMusicEstimator()
    cfar_th: np.ndarray | None = None
    if cfar_detector is not None:
        magnitude = _roi_magnitude_from_profile(
            profile_complex,
            range_bin_step=range_bin_step,
            range_roi=range_roi,
        )
        cfar_th = compute_1d_cfar_threshold(magnitude, cfar_detector)
    peaks = est(
        profile_complex,
        range_bin_step=float(range_bin_step),
        range_roi=range_roi,
        num_sources=int(num_sources),
        subarray_size=int(subarray_size),
        threshold=float(threshold),
        cfar=cfar_th,
    )
    if peaks.peak_ranges_m.size == 0:
        return float("nan")
    return float(peaks.peak_ranges_m[0])


def estimate_monostatic_range_esprit_m(
    profile_complex: Sequence[complex] | np.ndarray,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    num_sources: int = DEFAULT_ESPRIT_NUM_SOURCES,
    subarray_size: int = DEFAULT_ESPRIT_SUBARRAY_SIZE,
    window_size: int = DEFAULT_ESPRIT_WINDOW_SIZE,
    estimator: RangeEspritEstimator | None = None,
) -> float:
    """对 CPI 复数距离谱做 1D ESPRIT，返回最强峰距离 (m)；无峰时 ``nan``。"""
    est = estimator or RangeEspritEstimator()
    peaks = est(
        profile_complex,
        range_bin_step=float(range_bin_step),
        range_roi=range_roi,
        num_sources=int(num_sources),
        subarray_size=int(subarray_size),
        window_size=int(window_size),
    )
    if peaks.peak_ranges_m.size == 0:
        return float("nan")
    return float(peaks.peak_ranges_m[0])


def localize_xy_from_two_ranges(
    pos0_xy: Sequence[float],
    r0_m: float,
    pos1_xy: Sequence[float],
    r1_m: float,
    *,
    y_hint: float | None = None,
) -> tuple[float, float]:
    """由两单基地斜距在 z=0 平面交会目标 (x, y)。"""
    return localize_xy_two_monostatic_ranges(
        pos0_xy,
        r0_m,
        pos1_xy,
        r1_m,
        y_hint=y_hint,
    )


def grc_cooperative_processing_params() -> dict[str, Any]:
    """返回与 cooperative monostatic GRC 对齐的默认处理参数字典。"""
    fft_len = DEFAULT_FFT_LEN
    zeropadding_fac = DEFAULT_ZEROPADDING_FAC
    transpose_len = DEFAULT_TRANSPOSE_LEN
    return {
        "fft_len": fft_len,
        "zeropadding_fac": zeropadding_fac,
        "transpose_len": transpose_len,
        "vlen_divide_cpi": fft_len * zeropadding_fac * transpose_len,
        "vlen_range": fft_len * zeropadding_fac,
        "subcarrier_spacing_hz": DEFAULT_SUBCARRIER_SPACING_HZ,
        "range_bin_step": cooperative_range_bin_step_m(
            fft_len=fft_len,
            subcarrier_spacing_hz=DEFAULT_SUBCARRIER_SPACING_HZ,
            zeropadding_fac=zeropadding_fac,
        ),
        "range_roi": DEFAULT_RANGE_ROI,
        "music_num_sources": DEFAULT_MUSIC_NUM_SOURCES,
        "music_subarray_size": DEFAULT_MUSIC_SUBARRAY_SIZE,
        "music_threshold": DEFAULT_MUSIC_THRESHOLD,
        "esprit_num_sources": DEFAULT_ESPRIT_NUM_SOURCES,
        "esprit_subarray_size": DEFAULT_ESPRIT_SUBARRAY_SIZE,
        "esprit_window_size": DEFAULT_ESPRIT_WINDOW_SIZE,
        "cfar_enabled": False,
    }
