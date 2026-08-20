"""Cooperative monostatic 离线 DSP：divide CPI → 距离谱 → MUSIC / ESPRIT。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from isac.sensing.detection.cfar import CFARDetector
from isac.utils.windows import blackmanharris
from isac.sensing.detection.range_esprit_estimator import RangeEspritEstimator
from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac.sensing.localization import (
    localize_xy_two_monostatic_ranges,
    localize_xy_two_quasi_monostatic_path_sums,
)

DEFAULT_FFT_LEN = 2048
DEFAULT_ZEROPADDING_FAC = 4
DEFAULT_TRANSPOSE_LEN = 4
DEFAULT_SUBCARRIER_SPACING_HZ = 120e3
DEFAULT_RANGE_ROI = (0.0, 3.5)
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

# 站点中点（热力图 / Horn）；TX/RX 对称半轴默认 5 cm（物理收发分离）。
DEFAULT_DEV0_XY = (0.0, -2.0)
DEFAULT_DEV1_XY = (-2.0, 0.0)
DEFAULT_ANTENNA_OFFSET_DEV0_M = 0.05
DEFAULT_ANTENNA_OFFSET_DEV1_M = 0.05
DEFAULT_DEV0_TX_XY = (DEFAULT_ANTENNA_OFFSET_DEV0_M, -2.0)
DEFAULT_DEV0_RX_XY = (-DEFAULT_ANTENNA_OFFSET_DEV0_M, -2.0)
DEFAULT_DEV1_TX_XY = (-2.0, DEFAULT_ANTENNA_OFFSET_DEV1_M)
DEFAULT_DEV1_RX_XY = (-2.0, -DEFAULT_ANTENNA_OFFSET_DEV1_M)

# 论文 / 图表对外表述（内部仍用 dev0/dev1 索引与 HDF5 键名）
BS_DISPLAY_NAMES: tuple[str, str] = ("BS-0", "BS-1")


def bs_display_name(device: int | str) -> str:
    """将 dev0/dev1 或 0/1 映射为对外 BS 标签（BS-0 / BS-1）。"""
    if isinstance(device, int):
        if device in (0, 1):
            return BS_DISPLAY_NAMES[device]
        raise ValueError(f"BS index must be 0 or 1, got {device}")
    key = str(device).strip().lower()
    if key in ("dev0", "0", "bs0", "bs-0"):
        return BS_DISPLAY_NAMES[0]
    if key in ("dev1", "1", "bs1", "bs-1"):
        return BS_DISPLAY_NAMES[1]
    raise ValueError(f"unknown device key {device!r}")


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
    bh_window = np.asarray(blackmanharris(vlen_range), dtype=np.float32)
    complex_acc = np.zeros(vlen_range, dtype=np.complex128)
    for symbol in divide_buf:
        h_win = symbol * bh_window
        complex_acc += np.fft.fft(h_win).astype(np.complex128, copy=False)
    return complex_acc.astype(np.complex64, copy=False)


def divide_cpi_to_complex_range_profile_torch(
    cpi_flat: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    transpose_len: int = DEFAULT_TRANSPOSE_LEN,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Torch 版 divide CPI → 全谱复数距离向量（可在 CUDA 上）。"""
    device_t = torch.device(device)
    vlen_range = int(fft_len) * int(zeropadding_fac)
    expected = vlen_range * int(transpose_len)
    if isinstance(cpi_flat, torch.Tensor):
        flat = cpi_flat.to(device=device_t, dtype=torch.complex64).reshape(-1)
    else:
        flat = torch.as_tensor(
            np.asarray(cpi_flat, dtype=np.complex64).reshape(-1),
            dtype=torch.complex64,
            device=device_t,
        )
    if int(flat.numel()) != expected:
        raise ValueError(
            f"expected divide CPI length {expected}, got {int(flat.numel())}"
        )

    divide_buf = flat.reshape(int(transpose_len), vlen_range)
    bh_window = torch.as_tensor(
        np.asarray(blackmanharris(vlen_range), dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    )
    complex_acc = torch.zeros(vlen_range, dtype=torch.complex64, device=device_t)
    for symbol in divide_buf:
        complex_acc = complex_acc + torch.fft.fft(symbol * bh_window)
    return complex_acc


def divide_cpi_to_roi_range_profile(
    cpi_flat: Sequence[complex] | np.ndarray,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    transpose_len: int = DEFAULT_TRANSPOSE_LEN,
) -> np.ndarray:
    """divide CPI → 固定 ROI 复数距离谱 ``(L,)`` complex64。

    当 ``range_roi[0] == 0`` 时，输出可直接作为 MUSIC/ESPRIT 输入
    （估计器内 ``start_bin=0``，不会二次裁切）。
    """
    from isac_imp.range_profile_roi_slice import compute_range_roi

    profile = divide_cpi_to_complex_range_profile(
        cpi_flat,
        fft_len=fft_len,
        zeropadding_fac=zeropadding_fac,
        transpose_len=transpose_len,
    )
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=float(range_bin_step),
        vlen_in=int(profile.size),
    )
    return profile[start_bin : start_bin + num_bins].astype(np.complex64, copy=False)


def divide_cpi_to_roi_range_profile_torch(
    cpi_flat: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    transpose_len: int = DEFAULT_TRANSPOSE_LEN,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Torch 版 divide CPI → ROI 复数距离谱。"""
    from isac_imp.range_profile_roi_slice import compute_range_roi

    profile = divide_cpi_to_complex_range_profile_torch(
        cpi_flat,
        fft_len=fft_len,
        zeropadding_fac=zeropadding_fac,
        transpose_len=transpose_len,
        device=device,
    )
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=float(range_bin_step),
        vlen_in=int(profile.numel()),
    )
    return profile[start_bin : start_bin + num_bins]


def _use_torch_device(device: torch.device | str | None) -> torch.device | None:
    """``None`` / ``cpu`` → NumPy 路径；否则返回 torch device。"""
    if device is None:
        return None
    device_t = torch.device(device)
    if device_t.type == "cpu":
        return None
    return device_t


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
    profile_complex: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    num_sources: int = DEFAULT_MUSIC_NUM_SOURCES,
    subarray_size: int = DEFAULT_MUSIC_SUBARRAY_SIZE,
    threshold: float = DEFAULT_MUSIC_THRESHOLD,
    estimator: RangeMusicEstimator | None = None,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """对 CPI 复数距离谱做 1D MUSIC，返回最强峰距离 (m)；无峰时 ``nan``。

    ``profile_complex`` 可为全谱，或在 ``range_roi[0] == 0`` 时为 ROI 预裁切向量。
    ``device`` 为 CUDA 时走 Torch 估计器。
    """
    torch_dev = _use_torch_device(device)
    cfar_th: np.ndarray | None = None
    if cfar_detector is not None:
        if isinstance(profile_complex, torch.Tensor):
            magnitude = _roi_magnitude_from_profile(
                profile_complex.detach().cpu().numpy(),
                range_bin_step=range_bin_step,
                range_roi=range_roi,
            )
        else:
            magnitude = _roi_magnitude_from_profile(
                profile_complex,
                range_bin_step=range_bin_step,
                range_roi=range_roi,
            )
        cfar_th = compute_1d_cfar_threshold(magnitude, cfar_detector)

    if torch_dev is not None:
        from isac.sensing.detection.range_music_estimator_torch import (
            RangeMusicEstimatorTorch,
        )

        est_t = RangeMusicEstimatorTorch(device=torch_dev)
        peaks = est_t(
            profile_complex,
            range_bin_step=float(range_bin_step),
            range_roi=range_roi,
            num_sources=int(num_sources),
            subarray_size=int(subarray_size),
            threshold=float(threshold),
            cfar=cfar_th,
        )
    else:
        est = estimator or RangeMusicEstimator()
        peaks = est(
            profile_complex
            if not isinstance(profile_complex, torch.Tensor)
            else profile_complex.detach().cpu().numpy(),
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
    profile_complex: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    range_bin_step: float,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    num_sources: int = DEFAULT_ESPRIT_NUM_SOURCES,
    subarray_size: int = DEFAULT_ESPRIT_SUBARRAY_SIZE,
    window_size: int = DEFAULT_ESPRIT_WINDOW_SIZE,
    estimator: RangeEspritEstimator | None = None,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """对 CPI 复数距离谱做 1D ESPRIT，返回最强峰距离 (m)；无峰时 ``nan``。

    ``device`` 为 CUDA 时走 Torch 估计器。
    """
    torch_dev = _use_torch_device(device)
    cfar_th: np.ndarray | None = None
    if cfar_detector is not None:
        if isinstance(profile_complex, torch.Tensor):
            magnitude = _roi_magnitude_from_profile(
                profile_complex.detach().cpu().numpy(),
                range_bin_step=range_bin_step,
                range_roi=range_roi,
            )
        else:
            magnitude = _roi_magnitude_from_profile(
                profile_complex,
                range_bin_step=range_bin_step,
                range_roi=range_roi,
            )
        cfar_th = compute_1d_cfar_threshold(magnitude, cfar_detector)

    if torch_dev is not None:
        from isac.sensing.detection.range_esprit_estimator_torch import (
            RangeEspritEstimatorTorch,
        )

        est_t = RangeEspritEstimatorTorch(device=torch_dev)
        peaks = est_t(
            profile_complex,
            range_bin_step=float(range_bin_step),
            range_roi=range_roi,
            num_sources=int(num_sources),
            subarray_size=int(subarray_size),
            window_size=int(window_size),
            cfar=cfar_th,
        )
    else:
        est = estimator or RangeEspritEstimator()
        peaks = est(
            profile_complex
            if not isinstance(profile_complex, torch.Tensor)
            else profile_complex.detach().cpu().numpy(),
            range_bin_step=float(range_bin_step),
            range_roi=range_roi,
            num_sources=int(num_sources),
            subarray_size=int(subarray_size),
            window_size=int(window_size),
            cfar=cfar_th,
        )
    if peaks.peak_ranges_m.size == 0:
        return float("nan")
    return float(peaks.peak_ranges_m[0])


def music_range_from_roi_profile(
    profile_roi: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """预裁切 ROI 复数距离谱 → 1D MUSIC 最强峰距离 (m)。"""
    return estimate_monostatic_range_m(
        profile_roi,
        range_bin_step=float(proc_params["range_bin_step"]),
        range_roi=range_roi,
        num_sources=int(proc_params["music_num_sources"]),
        subarray_size=int(proc_params["music_subarray_size"]),
        threshold=float(proc_params["music_threshold"]),
        cfar_detector=cfar_detector,
        device=device,
    )


def esprit_range_from_roi_profile(
    profile_roi: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """预裁切 ROI 复数距离谱 → 1D ESPRIT 最强峰距离 (m)。"""
    return estimate_monostatic_range_esprit_m(
        profile_roi,
        range_bin_step=float(proc_params["range_bin_step"]),
        range_roi=range_roi,
        num_sources=int(proc_params["esprit_num_sources"]),
        subarray_size=int(proc_params["esprit_subarray_size"]),
        window_size=int(proc_params["esprit_window_size"]),
        cfar_detector=cfar_detector,
        device=device,
    )


def music_range_from_divide_cpi(
    divide_cpi: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """divide CPI → 固定 ROI 复数谱 → 1D MUSIC 最强峰距离 (m)。"""
    torch_dev = _use_torch_device(device)
    if torch_dev is not None:
        profile_roi = divide_cpi_to_roi_range_profile_torch(
            divide_cpi,
            range_bin_step=float(proc_params["range_bin_step"]),
            range_roi=range_roi,
            fft_len=int(proc_params["fft_len"]),
            zeropadding_fac=int(proc_params["zeropadding_fac"]),
            transpose_len=int(proc_params["transpose_len"]),
            device=torch_dev,
        )
    else:
        profile_roi = divide_cpi_to_roi_range_profile(
            divide_cpi
            if not isinstance(divide_cpi, torch.Tensor)
            else divide_cpi.detach().cpu().numpy(),
            range_bin_step=float(proc_params["range_bin_step"]),
            range_roi=range_roi,
            fft_len=int(proc_params["fft_len"]),
            zeropadding_fac=int(proc_params["zeropadding_fac"]),
            transpose_len=int(proc_params["transpose_len"]),
        )
    return music_range_from_roi_profile(
        profile_roi,
        proc_params=proc_params,
        range_roi=range_roi,
        cfar_detector=cfar_detector,
        device=device,
    )


def esprit_range_from_divide_cpi(
    divide_cpi: Sequence[complex] | np.ndarray | torch.Tensor,
    *,
    proc_params: dict[str, Any],
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    cfar_detector: CFARDetector | None = None,
    device: torch.device | str | None = None,
) -> float:
    """divide CPI → 固定 ROI 复数谱 → 1D ESPRIT 最强峰距离 (m)。"""
    torch_dev = _use_torch_device(device)
    if torch_dev is not None:
        profile_roi = divide_cpi_to_roi_range_profile_torch(
            divide_cpi,
            range_bin_step=float(proc_params["range_bin_step"]),
            range_roi=range_roi,
            fft_len=int(proc_params["fft_len"]),
            zeropadding_fac=int(proc_params["zeropadding_fac"]),
            transpose_len=int(proc_params["transpose_len"]),
            device=torch_dev,
        )
    else:
        profile_roi = divide_cpi_to_roi_range_profile(
            divide_cpi
            if not isinstance(divide_cpi, torch.Tensor)
            else divide_cpi.detach().cpu().numpy(),
            range_bin_step=float(proc_params["range_bin_step"]),
            range_roi=range_roi,
            fft_len=int(proc_params["fft_len"]),
            zeropadding_fac=int(proc_params["zeropadding_fac"]),
            transpose_len=int(proc_params["transpose_len"]),
        )
    return esprit_range_from_roi_profile(
        profile_roi,
        proc_params=proc_params,
        range_roi=range_roi,
        cfar_detector=cfar_detector,
        device=device,
    )


def localize_xy_from_two_ranges(
    pos0_xy: Sequence[float],
    r0_m: float,
    pos1_xy: Sequence[float],
    r1_m: float,
    *,
    y_hint: float | None = None,
    use_horn_disambiguation: bool = True,
    horn_aim_xy: Sequence[float] = (0.0, 0.0),
    horn_cos_power: float = 2.0,
    tx0_xy: Sequence[float] | None = None,
    rx0_xy: Sequence[float] | None = None,
    tx1_xy: Sequence[float] | None = None,
    rx1_xy: Sequence[float] | None = None,
    use_tx_rx_ellipses: bool = True,
) -> tuple[float, float]:
    """由两单基地斜距在 z=0 平面交会目标 (x, y)。

    默认按 TX/RX ±5 cm 椭圆交会（路径和 ``2r``）；``use_tx_rx_ellipses=False``
    时用 ``pos0/pos1`` 共址圆交会。Horn 消歧指向 ``horn_aim_xy``。
    """
    if use_tx_rx_ellipses:
        return localize_xy_two_quasi_monostatic_path_sums(
            tx0_xy if tx0_xy is not None else DEFAULT_DEV0_TX_XY,
            rx0_xy if rx0_xy is not None else DEFAULT_DEV0_RX_XY,
            r0_m,
            tx1_xy if tx1_xy is not None else DEFAULT_DEV1_TX_XY,
            rx1_xy if rx1_xy is not None else DEFAULT_DEV1_RX_XY,
            r1_m,
            y_hint=y_hint,
            use_horn_disambiguation=use_horn_disambiguation,
            horn_aim_xy=horn_aim_xy,
            horn_cos_power=horn_cos_power,
        )
    return localize_xy_two_monostatic_ranges(
        pos0_xy,
        r0_m,
        pos1_xy,
        r1_m,
        y_hint=y_hint,
        use_horn_disambiguation=use_horn_disambiguation,
        horn_aim_xy=horn_aim_xy,
        horn_cos_power=horn_cos_power,
    )


def localize_xy_from_two_ranges_with_bias(
    pos0_xy: Sequence[float],
    r0_m: float,
    pos1_xy: Sequence[float],
    r1_m: float,
    *,
    bias_dev0_m: float = 0.0,
    bias_dev1_m: float = 0.0,
    y_hint: float | None = None,
    use_horn_disambiguation: bool = True,
    horn_aim_xy: Sequence[float] = (0.0, 0.0),
    horn_cos_power: float = 2.0,
    tx0_xy: Sequence[float] | None = None,
    rx0_xy: Sequence[float] | None = None,
    tx1_xy: Sequence[float] | None = None,
    rx1_xy: Sequence[float] | None = None,
    use_tx_rx_ellipses: bool = True,
) -> tuple[float, float]:
    """应用 per-dev 距离偏置后再做交会定位（默认 TX/RX 椭圆）。"""
    from isac_imp.cooperative_monostatic_range_calibration import (
        correct_monostatic_range_pair,
    )

    r0_cal, r1_cal = correct_monostatic_range_pair(
        r0_m,
        r1_m,
        bias_dev0_m=bias_dev0_m,
        bias_dev1_m=bias_dev1_m,
    )
    return localize_xy_from_two_ranges(
        pos0_xy,
        r0_cal,
        pos1_xy,
        r1_cal,
        y_hint=y_hint,
        use_horn_disambiguation=use_horn_disambiguation,
        horn_aim_xy=horn_aim_xy,
        horn_cos_power=horn_cos_power,
        tx0_xy=tx0_xy,
        rx0_xy=rx0_xy,
        tx1_xy=tx1_xy,
        rx1_xy=rx1_xy,
        use_tx_rx_ellipses=use_tx_rx_ellipses,
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
