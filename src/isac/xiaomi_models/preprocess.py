"""小米单站测距模型：预处理。"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from isac_imp.range_profile_roi_slice import compute_range_roi

FeatureMode = Literal["real_imag", "mag_phase"]
FEATURE_MODES: tuple[FeatureMode, ...] = ("real_imag", "mag_phase")

DEFAULT_FFT_LEN = 4096
DEFAULT_ZEROPADDING_FAC = 4
DEFAULT_SUBCARRIER_SPACING_HZ = 60e3
_C_LIGHT = 3e8
DEFAULT_RANGE_ROI: tuple[float, float] = (0.0, 8.0)


def default_range_bin_step(
    *,
    fft_len: int = DEFAULT_FFT_LEN,
    zeropadding_fac: int = DEFAULT_ZEROPADDING_FAC,
    subcarrier_spacing_hz: float = DEFAULT_SUBCARRIER_SPACING_HZ,
) -> float:
    """``c / (2 · fft_len · scs · zp)``，与 USRP 单站流图一致。"""
    return float(_C_LIGHT / (2.0 * int(fft_len) * float(subcarrier_spacing_hz) * int(zeropadding_fac)))


def profile_to_roi(
    profile: np.ndarray | torch.Tensor,
    *,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    range_bin_step: float | None = None,
) -> np.ndarray:
    """复数距离谱 → ROI 切片 ``(L,)`` complex64。"""
    arr = np.asarray(profile)
    if arr.ndim != 1:
        raise ValueError(f"profile 须为一维，收到 shape {arr.shape}")
    step = float(range_bin_step) if range_bin_step is not None else default_range_bin_step()
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=step,
        vlen_in=int(arr.shape[0]),
    )
    return np.asarray(arr[start_bin : start_bin + num_bins], dtype=np.complex64)


def _mag_phase_features(profile_complex: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    mag = torch.sqrt(
        profile_complex.real.to(dtype=torch.float32) ** 2
        + profile_complex.imag.to(dtype=torch.float32) ** 2
    ).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)
    phase = torch.atan2(profile_complex.imag, profile_complex.real) / np.pi
    return torch.stack([mag_db, phase], dim=0).to(dtype=torch.float32)


def _real_imag_features(profile_complex: torch.Tensor) -> torch.Tensor:
    real = profile_complex.real.to(dtype=torch.float32)
    imag = profile_complex.imag.to(dtype=torch.float32)
    return torch.stack([real, imag], dim=0)


def profile_to_features(
    profile_roi: np.ndarray | torch.Tensor,
    *,
    mode: FeatureMode = "real_imag",
    eps: float = 1e-12,
) -> torch.Tensor:
    """ROI 复数距离谱 → ``(C, L)`` float32 特征。"""
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode 须为 {FEATURE_MODES}，收到 {mode!r}")
    if isinstance(profile_roi, np.ndarray):
        tensor = torch.from_numpy(np.asarray(profile_roi, dtype=np.complex64))
    else:
        tensor = profile_roi
    if tensor.ndim != 1:
        raise ValueError(f"profile_roi 须为一维，收到 shape {tuple(tensor.shape)}")
    if mode == "real_imag":
        return _real_imag_features(tensor)
    return _mag_phase_features(tensor, eps=eps)


def feature_in_channels(mode: FeatureMode = "real_imag") -> int:
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode 须为 {FEATURE_MODES}，收到 {mode!r}")
    return 2


def profiles_batch_to_features(
    profiles: np.ndarray | torch.Tensor,
    *,
    range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
    range_bin_step: float | None = None,
    mode: FeatureMode = "real_imag",
) -> torch.Tensor:
    """``(B, vlen)`` 复数谱 → ``(B, C, L)`` 特征。"""
    if isinstance(profiles, torch.Tensor):
        arr = profiles.detach().cpu().numpy()
    else:
        arr = np.asarray(profiles)
    if arr.ndim != 2:
        raise ValueError(f"profiles 须为 (B, vlen)，收到 {arr.shape}")
    feats = [
        profile_to_features(
            profile_to_roi(row, range_roi=range_roi, range_bin_step=range_bin_step),
            mode=mode,
        )
        for row in arr
    ]
    return torch.stack(feats, dim=0)
