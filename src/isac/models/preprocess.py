"""时延–多普勒谱预处理：特征提取与运动学标签生成。

CNN / MUSIC 共用 ``spectrum_tensor``（复数裁切谱）作为估计器输入；
训练标签由 ``kinematics_to_target_bins`` 从运动学统一生成。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..data_structures.types import SensMode
from ..sensing.geometry import compute_range, compute_vel, monostatic_range_velocity
from ..sensing.metric import SpectrumMetric
from ..sensing.spectrum.sensing_performance import SensingPerformance


def dd_spectrum_to_features(
    h_dd: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """将单条 ROI 裁切复数谱转为 CNN 特征 ``(C, H, W)``。

    - 通道 0：幅度 dB（逐样本零均值、单位方差）
    - 通道 1（可选）：相位，映射到 ``[-1, 1]``
    """
    mag = torch.abs(h_dd).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)

    channels = [mag_db]
    if use_phase:
        phase = torch.angle(h_dd) / np.pi
        channels.append(phase)

    return torch.stack(channels, dim=0)


def normalize_spectrum_batch(spectrum_tensor: torch.Tensor) -> torch.Tensor:
    """将复数谱规范为 ``(B, H, W)``。"""
    if spectrum_tensor.ndim == 2:
        return spectrum_tensor.unsqueeze(0)
    if spectrum_tensor.ndim == 3:
        return spectrum_tensor
    raise ValueError(
        "spectrum_tensor 须为 (H, W) 或 (B, H, W)，"
        f"收到 {tuple(spectrum_tensor.shape)}"
    )


def spectrum_tensor_to_features(
    spectrum_tensor: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """复数裁切谱 → CNN 特征 ``(B, C, H, W)`` float32。"""
    batch = normalize_spectrum_batch(spectrum_tensor)
    return torch.stack(
        [
            dd_spectrum_to_features(batch[i], eps=eps, use_phase=use_phase)
            for i in range(batch.shape[0])
        ],
        dim=0,
    ).to(dtype=torch.float32)


def _bistatic_range_velocity_batch(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运动学 → 双基地折叠路径长与路径变化率 ``(B,)`` float32。"""
    pos = target_position.reshape(-1, 3)
    vel = target_velocity.reshape(-1, 3)
    tx = tx_pos.reshape(-1).detach().cpu().numpy()
    rx = rx_pos.reshape(-1).detach().cpu().numpy()

    device = target_position.device
    t_pos = pos.to(dtype=torch.float64)
    t_vel = vel.to(dtype=torch.float64)
    r_pos = torch.as_tensor(rx, dtype=torch.float64, device=device).reshape(1, 3)
    x_stack = torch.as_tensor(tx, dtype=torch.float64, device=device).reshape(1, 3)
    r_vel = torch.zeros(1, 3, dtype=torch.float64, device=device)
    x_vel = torch.zeros(1, 3, dtype=torch.float64, device=device)

    is_bistatic = torch.ones(1, pos.shape[0], 1, dtype=torch.bool, device=device)
    range_m = compute_range(is_bistatic, t_pos, r_pos, x_stack)[0, :, 0]
    vel_mps = compute_vel(
        is_bistatic, t_pos, t_vel, r_pos, r_vel, x_stack, x_vel
    )[0, :, 0]

    device = target_position.device
    dtype = torch.float32
    return (
        range_m.to(dtype=dtype, device=device),
        vel_mps.to(dtype=dtype, device=device),
    )


def kinematics_to_range_velocity(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    bs_pos: torch.Tensor,
    *,
    sens_mode: SensMode = "monostatic",
    tx_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运动学 → 距离与速度真值 ``(B,)`` float32。

    - ``monostatic``：斜距与 RX 视线径向速度（``tx_pos`` 缺省为 ``bs_pos`` 共址）
    - ``bistatic``：折叠路径长与路径变化率（须提供 ``tx_pos``，``bs_pos`` 为 RX）
    """
    if sens_mode == "bistatic":
        if tx_pos is None:
            raise ValueError("sens_mode='bistatic' 时须提供 tx_pos")
        return _bistatic_range_velocity_batch(
            target_position, target_velocity, tx_pos, bs_pos
        )

    pos = target_position.reshape(-1, 3)
    vel = target_velocity.reshape(-1, 3)
    bs = bs_pos.reshape(-1).detach().cpu().numpy()

    ranges: list[float] = []
    velocities: list[float] = []
    for i in range(pos.shape[0]):
        r, v = monostatic_range_velocity(
            pos[i].detach().cpu().numpy(),
            vel[i].detach().cpu().numpy(),
            bs,
        )
        ranges.append(r)
        velocities.append(v)

    device = target_position.device
    dtype = torch.float32
    return (
        torch.tensor(ranges, dtype=dtype, device=device),
        torch.tensor(velocities, dtype=dtype, device=device),
    )


def kinematics_to_target_bins(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    bs_pos: torch.Tensor,
    *,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
    sens_mode: SensMode = "monostatic",
    tx_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """运动学 → ROI 局部 bin 监督 ``(B, 2)`` = ``[peaks_delay, peaks_doppler]``。"""
    range_m, vel_mps = kinematics_to_range_velocity(
        target_position,
        target_velocity,
        bs_pos,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    metric = SpectrumMetric(sensing_performance)
    delay_bin, doppler_bin = metric.physical_to_local_bins(
        range_m,
        vel_mps,
        num_doppler_bins=num_doppler_bins,
        sens_mode=sens_mode,
    )
    return torch.stack([delay_bin, doppler_bin], dim=-1).to(
        dtype=range_m.dtype,
        device=range_m.device,
    )


def range_profile_to_features(
    profile_complex: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """单设备 ROI 复数距离谱 → ``(C, L)`` float 特征。"""
    mag = torch.abs(profile_complex).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)
    channels = [mag_db]
    if use_phase:
        channels.append(torch.angle(profile_complex) / np.pi)
    return torch.stack(channels, dim=0).to(dtype=torch.float32)


def dual_range_profile_to_features(
    profile_dev0: torch.Tensor,
    profile_dev1: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """双设备 ROI 复数距离谱 → ``(4, L)`` float 特征。"""
    feat0 = range_profile_to_features(profile_dev0, eps=eps, use_phase=use_phase)
    feat1 = range_profile_to_features(profile_dev1, eps=eps, use_phase=use_phase)
    return torch.cat([feat0, feat1], dim=0)


def _normalize_dual_range_batch(dual_profiles: torch.Tensor) -> torch.Tensor:
    """规范为 ``(B, 2, L)`` complex。"""
    if dual_profiles.ndim == 2:
        if dual_profiles.shape[0] != 2:
            raise ValueError(
                "dual_profiles 单样本须为 (2, L) complex，"
                f"收到 {tuple(dual_profiles.shape)}"
            )
        return dual_profiles.unsqueeze(0)
    if dual_profiles.ndim == 3:
        if dual_profiles.shape[1] != 2:
            raise ValueError(
                "dual_profiles batch 须为 (B, 2, L) complex，"
                f"收到 {tuple(dual_profiles.shape)}"
            )
        return dual_profiles
    raise ValueError(
        "dual_profiles 须为 (2, L) 或 (B, 2, L) complex，"
        f"收到 {tuple(dual_profiles.shape)}"
    )


def dual_range_profiles_to_features(
    dual_profiles: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """双设备 ROI 复数距离谱 batch → ``(B, 4, L)`` float 特征。"""
    batch = _normalize_dual_range_batch(dual_profiles)
    return torch.stack(
        [
            dual_range_profile_to_features(
                batch[i, 0],
                batch[i, 1],
                eps=eps,
                use_phase=use_phase,
            )
            for i in range(batch.shape[0])
        ],
        dim=0,
    )


def divide_cpi_to_roi_range_profile_np(
    cpi_flat,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
):
    """divide CPI → ROI 复数距离谱 ``(L,)`` complex64 numpy。"""
    from isac_imp.cooperative_monostatic_pipeline import (
        divide_cpi_to_complex_range_profile,
    )
    from isac_imp.range_profile_roi_slice import compute_range_roi

    profile = divide_cpi_to_complex_range_profile(
        cpi_flat,
        fft_len=proc_params["fft_len"],
        zeropadding_fac=proc_params["zeropadding_fac"],
        transpose_len=proc_params["transpose_len"],
    )
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=float(proc_params["range_bin_step"]),
        vlen_in=int(profile.size),
    )
    return profile[start_bin : start_bin + num_bins].astype(np.complex64, copy=False)


def divide_cpi_dual_to_roi_range_profiles_np(
    cpi_dev0,
    cpi_dev1,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """双设备 divide CPI → ROI 复数距离谱。"""
    roi0 = divide_cpi_to_roi_range_profile_np(
        cpi_dev0, proc_params=proc_params, range_roi=range_roi
    )
    roi1 = divide_cpi_to_roi_range_profile_np(
        cpi_dev1, proc_params=proc_params, range_roi=range_roi
    )
    return roi0, roi1


def divide_cpi_to_roi_range_slowtime_np(
    cpi_flat,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
) -> np.ndarray:
    """divide CPI → per-symbol ROI 复数距离谱 ``(T, L)``，T=transpose_len。"""
    from isac.utils.windows import blackmanharris
    from isac_imp.range_profile_roi_slice import compute_range_roi

    fft_len = int(proc_params["fft_len"])
    zeropadding_fac = int(proc_params["zeropadding_fac"])
    transpose_len = int(proc_params["transpose_len"])
    vlen_range = int(proc_params.get("vlen_range", fft_len * zeropadding_fac))
    expected = vlen_range * transpose_len
    flat = np.asarray(cpi_flat, dtype=np.complex64).reshape(-1)
    if flat.size != expected:
        raise ValueError(f"expected divide CPI length {expected}, got {flat.size}")

    divide_buf = flat.reshape(transpose_len, vlen_range)
    bh_window = np.asarray(blackmanharris(vlen_range), dtype=np.float32)
    spectra = []
    for symbol in divide_buf:
        h_win = symbol * bh_window
        spectra.append(np.fft.fft(h_win).astype(np.complex64, copy=False))
    profile = np.stack(spectra, axis=0)
    start_bin, num_bins, _ = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=float(proc_params["range_bin_step"]),
        vlen_in=int(profile.shape[1]),
    )
    return profile[:, start_bin : start_bin + num_bins].astype(np.complex64, copy=False)


def divide_cpi_dual_to_roi_range_slowtime_np(
    cpi_dev0,
    cpi_dev1,
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
) -> np.ndarray:
    """双设备 divide CPI → ``(2, T, L)`` 复数 range×slow-time ROI。"""
    roi0 = divide_cpi_to_roi_range_slowtime_np(
        cpi_dev0, proc_params=proc_params, range_roi=range_roi
    )
    roi1 = divide_cpi_to_roi_range_slowtime_np(
        cpi_dev1, proc_params=proc_params, range_roi=range_roi
    )
    return np.stack([roi0, roi1], axis=0).astype(np.complex64, copy=False)


CooperativeFeatureMode = Literal[
    "complex_roi",
    "real_imag",
    "logmag_fixed_norm",
    "legacy_4ch",
    "range_slowtime_2d",
]
COOPERATIVE_FEATURE_MODES: tuple[CooperativeFeatureMode, ...] = (
    "complex_roi",
    "real_imag",
    "logmag_fixed_norm",
    "legacy_4ch",
    "range_slowtime_2d",
)


def cooperative_feature_in_channels(mode: CooperativeFeatureMode) -> int:
    """CNN ``in_channels`` for a cooperative monostatic feature mode."""
    if mode == "logmag_fixed_norm":
        return 2
    return 4


def cooperative_model_type(mode: CooperativeFeatureMode) -> str:
    """Return ``'2d'`` for range×slow-time CNN, else ``'1d'``."""
    if mode == "range_slowtime_2d":
        return "2d"
    return "1d"


def cooperative_uses_slowtime_input(mode: CooperativeFeatureMode) -> bool:
    return mode == "range_slowtime_2d"


def cooperative_input_is_complex(mode: CooperativeFeatureMode) -> bool:
    return mode == "complex_roi"


def _dual_roi_to_batch(dual_roi: torch.Tensor) -> torch.Tensor:
    """规范为 ``(B, 2, L)`` complex。"""
    if dual_roi.ndim == 2:
        if dual_roi.shape[0] != 2:
            raise ValueError(
                "dual_roi 单样本须为 (2, L) complex，"
                f"收到 {tuple(dual_roi.shape)}"
            )
        return dual_roi.unsqueeze(0)
    if dual_roi.ndim == 3 and dual_roi.shape[1] == 2:
        return dual_roi
    raise ValueError(
        "dual_roi complex 须为 (2, L) 或 (B, 2, L)，"
        f"收到 {tuple(dual_roi.shape)}"
    )


def range_profile_to_real_imag_features(profile_complex: torch.Tensor) -> torch.Tensor:
    """单设备 ROI 复数距离谱 → ``(2, L)`` real+imag float。"""
    real = profile_complex.real.to(dtype=torch.float32)
    imag = profile_complex.imag.to(dtype=torch.float32)
    return torch.stack([real, imag], dim=0)


def dual_roi_to_real_imag_features(dual_roi: torch.Tensor) -> torch.Tensor:
    """双设备 ROI 复数距离谱 → ``(4, L)`` 或 ``(B, 4, L)`` real+imag float。"""
    batch = _dual_roi_to_batch(dual_roi)
    feats = []
    for i in range(batch.shape[0]):
        feat0 = range_profile_to_real_imag_features(batch[i, 0])
        feat1 = range_profile_to_real_imag_features(batch[i, 1])
        feats.append(torch.cat([feat0, feat1], dim=0))
    if dual_roi.ndim == 2:
        return feats[0]
    return torch.stack(feats, dim=0)


def _dual_slowtime_to_batch(dual_slowtime: torch.Tensor) -> torch.Tensor:
    """规范为 ``(B, 2, T, L)`` complex。"""
    if dual_slowtime.ndim == 3:
        if dual_slowtime.shape[0] != 2:
            raise ValueError(
                "dual_slowtime 单样本须为 (2, T, L) complex，"
                f"收到 {tuple(dual_slowtime.shape)}"
            )
        return dual_slowtime.unsqueeze(0)
    if dual_slowtime.ndim == 4 and dual_slowtime.shape[1] == 2:
        return dual_slowtime
    raise ValueError(
        "dual_slowtime complex 须为 (2, T, L) 或 (B, 2, T, L)，"
        f"收到 {tuple(dual_slowtime.shape)}"
    )


def dual_slowtime_to_real_imag_features(dual_slowtime: torch.Tensor) -> torch.Tensor:
    """双设备 range×slow-time → ``(4, T, L)`` 或 ``(B, 4, T, L)`` real+imag float。"""
    batch = _dual_slowtime_to_batch(dual_slowtime)
    feats = []
    for i in range(batch.shape[0]):
        dev_feats = []
        for dev in range(2):
            real = batch[i, dev].real.to(dtype=torch.float32)
            imag = batch[i, dev].imag.to(dtype=torch.float32)
            dev_feats.extend([real, imag])
        feats.append(torch.stack(dev_feats, dim=0))
    if dual_slowtime.ndim == 3:
        return feats[0]
    return torch.stack(feats, dim=0)


def dual_slowtime_to_model_input(
    dual_slowtime: torch.Tensor,
    *,
    mode: CooperativeFeatureMode = "range_slowtime_2d",
) -> torch.Tensor:
    """双设备 range×slow-time 复数谱 → 2D CNN 输入 tensor。"""
    if mode != "range_slowtime_2d":
        raise ValueError(f"dual_slowtime_to_model_input 仅支持 range_slowtime_2d，收到 {mode!r}")
    return dual_slowtime_to_real_imag_features(dual_slowtime)


def _normalize_logmag_with_stats(
    mag_db: torch.Tensor,
    *,
    mean: float,
    std: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    return (mag_db - mean) / (std + eps)


def range_profile_to_logmag_fixed_norm(
    profile_complex: torch.Tensor,
    *,
    mean: float,
    std: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """单设备 ROI → ``(1, L)`` 固定统计量归一化 log-mag。"""
    mag = torch.abs(profile_complex).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    return _normalize_logmag_with_stats(mag_db, mean=mean, std=std, eps=eps).unsqueeze(0)


def dual_roi_to_logmag_fixed_norm(
    dual_roi: torch.Tensor,
    *,
    norm_means: np.ndarray | torch.Tensor,
    norm_stds: np.ndarray | torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """双设备 ROI → ``(2, L)`` 或 ``(B, 2, L)`` 固定统计量 log-mag float。"""
    batch = _dual_roi_to_batch(dual_roi)
    means = torch.as_tensor(norm_means, dtype=torch.float32).reshape(-1)
    stds = torch.as_tensor(norm_stds, dtype=torch.float32).reshape(-1)
    if means.numel() != 2 or stds.numel() != 2:
        raise ValueError("logmag_fixed_norm 须提供 2 个 mean/std（dev0, dev1）")
    feats = []
    for i in range(batch.shape[0]):
        ch0 = range_profile_to_logmag_fixed_norm(
            batch[i, 0], mean=float(means[0]), std=float(stds[0]), eps=eps
        )
        ch1 = range_profile_to_logmag_fixed_norm(
            batch[i, 1], mean=float(means[1]), std=float(stds[1]), eps=eps
        )
        feats.append(torch.cat([ch0, ch1], dim=0))
    if dual_roi.ndim == 2:
        return feats[0]
    return torch.stack(feats, dim=0)


def dual_roi_to_model_input(
    dual_roi: torch.Tensor,
    *,
    mode: CooperativeFeatureMode = "real_imag",
    norm_means: np.ndarray | torch.Tensor | None = None,
    norm_stds: np.ndarray | torch.Tensor | None = None,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """双设备 ROI 复数距离谱 → 模型输入 tensor。"""
    if mode == "complex_roi":
        if dual_roi.ndim == 2:
            return dual_roi
        if dual_roi.ndim == 3 and dual_roi.shape[1] == 2:
            return dual_roi
        raise ValueError(
            "complex_roi 须为 (2, L) 或 (B, 2, L) complex，"
            f"收到 {tuple(dual_roi.shape)}"
        )
    if mode == "real_imag":
        return dual_roi_to_real_imag_features(dual_roi)
    if mode == "logmag_fixed_norm":
        if norm_means is None or norm_stds is None:
            raise ValueError("logmag_fixed_norm 须提供 norm_means / norm_stds")
        return dual_roi_to_logmag_fixed_norm(
            dual_roi,
            norm_means=norm_means,
            norm_stds=norm_stds,
            eps=eps,
        )
    if mode == "legacy_4ch":
        out = dual_range_profiles_to_features(dual_roi, eps=eps, use_phase=use_phase)
        if dual_roi.ndim == 2:
            return out[0]
        return out
    raise ValueError(f"未知 feature mode: {mode!r}")


def apply_cooperative_feature_augmentation(
    features: torch.Tensor,
    *,
    noise_std: float = 0.0,
    spec_augment_prob: float = 0.0,
    spec_augment_max_bins: int = 3,
    spec_augment_max_slowtime_rows: int = 1,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """训练时对 float 特征做噪声 / SpecAugment（支持 1D ``(C,L)`` 与 2D ``(C,T,L)``）。

    噪声与 SpecAugment 均由 ``rng``（缺省新建 Generator）驱动，便于按 seed 严格复现。
    """
    out = features.clone()
    gen = rng or np.random.default_rng()
    if noise_std > 0.0:
        noise = gen.normal(0.0, float(noise_std), size=tuple(out.shape)).astype(
            np.float32, copy=False
        )
        out = out + torch.from_numpy(noise).to(dtype=out.dtype, device=out.device)
    if spec_augment_prob > 0.0:
        if gen.random() < spec_augment_prob:
            if out.ndim == 2:
                length = int(out.shape[-1])
                width = int(gen.integers(1, max(2, spec_augment_max_bins + 1)))
                start = int(gen.integers(0, max(1, length - width + 1)))
                out[..., start : start + width] = 0.0
            elif out.ndim == 3:
                length = int(out.shape[-1])
                width = int(gen.integers(1, max(2, spec_augment_max_bins + 1)))
                start = int(gen.integers(0, max(1, length - width + 1)))
                out[..., start : start + width] = 0.0
                if spec_augment_max_slowtime_rows > 0 and out.shape[-2] > 1:
                    n_rows = int(
                        gen.integers(1, min(out.shape[-2], spec_augment_max_slowtime_rows) + 1)
                    )
                    row_idx = gen.choice(out.shape[-2], size=n_rows, replace=False)
                    out[..., row_idx, :] = 0.0
    return out


def save_cooperative_norm_stats(
    path: str | Path,
    *,
    means: np.ndarray,
    stds: np.ndarray,
    feature_mode: CooperativeFeatureMode = "logmag_fixed_norm",
) -> Path:
    """保存 Run1-only 固定归一化统计量。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        means=np.asarray(means, dtype=np.float64),
        stds=np.asarray(stds, dtype=np.float64),
        feature_mode=np.array(feature_mode),
    )
    return out


def load_cooperative_norm_stats(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, CooperativeFeatureMode]:
    """加载固定归一化统计量 npz。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"norm stats 不存在: {p}")
    data = np.load(p, allow_pickle=False)
    means = np.asarray(data["means"], dtype=np.float64)
    stds = np.asarray(data["stds"], dtype=np.float64)
    mode_raw = str(np.asarray(data.get("feature_mode", "logmag_fixed_norm")).item())
    if mode_raw not in COOPERATIVE_FEATURE_MODES:
        raise ValueError(f"norm stats feature_mode 无效: {mode_raw!r}")
    return means, stds, mode_raw  # type: ignore[return-value]


def compute_logmag_norm_stats_from_dual_rois(
    dual_rois: list[np.ndarray],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """从 ROI 复数谱列表估计 dev0/dev1 log-mag 全局 mean/std。"""
    if not dual_rois:
        raise ValueError("dual_rois 不能为空")
    dev_mags: list[list[np.ndarray]] = [[], []]
    for dual in dual_rois:
        arr = np.asarray(dual, dtype=np.complex64)
        if arr.shape[0] != 2:
            raise ValueError(f"dual ROI 须为 (2, L)，收到 {arr.shape}")
        for dev in range(2):
            mag = np.abs(arr[dev]).clip(min=eps)
            dev_mags[dev].append(20.0 * np.log10(mag))
    means = np.array(
        [float(np.mean(np.concatenate(dev_mags[dev]))) for dev in range(2)],
        dtype=np.float64,
    )
    stds = np.array(
        [float(np.std(np.concatenate(dev_mags[dev]))) for dev in range(2)],
        dtype=np.float64,
    )
    stds = np.maximum(stds, eps)
    return means, stds


def compute_logmag_norm_stats_from_h5(
    h5_path: str | Path,
    frame_indices: np.ndarray | Sequence[int],
    *,
    proc_params: dict,
    range_roi: tuple[float, float],
    max_samples: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """从 raw cooperative H5 帧索引估计 log-mag 固定归一化统计量。"""
    import h5py

    indices = np.asarray(frame_indices, dtype=np.int64)
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    dual_rois: list[np.ndarray] = []
    iterator: Iterable[int] = indices.tolist()
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="norm stats", unit="frame")

    with h5py.File(h5_path, "r") as f:
        dev0_ds = f["profiles_dev0"]
        dev1_ds = f["profiles_dev1"]
        for idx in iterator:
            roi0, roi1 = divide_cpi_dual_to_roi_range_profiles_np(
                dev0_ds[int(idx)],
                dev1_ds[int(idx)],
                proc_params=proc_params,
                range_roi=range_roi,
            )
            dual_rois.append(np.stack([roi0, roi1], axis=0))
    return compute_logmag_norm_stats_from_dual_rois(dual_rois)
