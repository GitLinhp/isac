"""采集样本可检测性质量门控：LoS 路径强度 + 时延–多普勒谱峰突出度。"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from scipy.constants import c

from .delay_doppler_spectrum import DelayDopplerSpectrum
from .sensing_performance import SensingPerformance

RejectReason = Literal[
    "no_valid_paths",
    "weak_los",
    "low_peak_prominence",
    "peak_misaligned",
]


@dataclass(frozen=True)
class SampleQualityConfig:
    """质量门控阈值。"""

    require_los: bool = True
    min_los_ratio: float = 0.3
    min_peak_prominence_db: float = 6.0
    max_bin_offset: int = 3
    rx_idx: int = 0
    tx_idx: int = 0
    eps: float = 1e-12


@dataclass
class SampleQualityResult:
    """单次质量评估结果。"""

    passed: bool
    reason: RejectReason | None = None
    los_ratio: float | None = None
    peak_prominence_db: float | None = None
    peak_delay_offset: int | None = None
    peak_doppler_offset: int | None = None


@dataclass
class QualityFilterStats:
    """采集期拒绝采样统计。"""

    accepted: int = 0
    rejected: int = 0
    reject_counts: dict[str, int] = field(default_factory=dict)

    def record_reject(self, reason: RejectReason) -> None:
        self.rejected += 1
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def record_accept(self) -> None:
        self.accepted += 1

    def summary_line(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.reject_counts.items()))
        return (
            f"质量过滤: accepted={self.accepted}, rejected={self.rejected}"
            + (f" ({parts})" if parts else "")
        )


def _extract_path_slices(
    rt_scene: object,
    *,
    rx_idx: int,
    tx_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取 ``(tau, valid, |a|)`` 一维路径切片，与 ``paths.tau[rx,tx,:]`` 对齐。

    Sionna ``paths.a`` 为 ``(Re, Im)`` 元组且含天线维；改用 ``paths.cir`` 复增益。
    """
    paths = rt_scene.paths
    tau_np = np.asarray(paths.tau, dtype=np.float64)
    valid_np = np.asarray(paths.valid, dtype=bool)

    if tau_np.ndim != 3:
        raise ValueError(
            f"paths.tau 须为 [rx, tx, max_paths]，收到 shape={tau_np.shape}"
        )

    n_paths = int(tau_np.shape[-1])
    tau_slice = tau_np[rx_idx, tx_idx, :]
    valid_slice = valid_np[rx_idx, tx_idx, :]

    a_cpx, _ = paths.cir(out_type="numpy")
    a_cpx = np.asarray(a_cpx)
    if a_cpx.shape[-1] == 1 and a_cpx.shape[-2] == n_paths:
        a_cpx = a_cpx[..., 0]

    a_work = np.squeeze(a_cpx)
    if a_work.ndim == 1:
        if a_work.shape[0] != n_paths:
            raise ValueError(
                f"CIR 路径数 {a_work.shape[0]} 与 tau 路径数 {n_paths} 不一致"
            )
        amp_slice = np.abs(a_work.astype(np.complex128))
    elif a_work.ndim == 3:
        amp_slice = np.abs(a_work[rx_idx, tx_idx, :].astype(np.complex128))
    elif a_work.ndim == 5:
        amp_slice = np.abs(a_work[rx_idx, 0, tx_idx, 0, :].astype(np.complex128))
    else:
        flat = a_work.reshape(-1)
        if flat.size < n_paths:
            raise ValueError(
                f"无法从 CIR shape={a_cpx.shape} 对齐 tau 路径维 n_paths={n_paths}"
            )
        amp_slice = np.abs(flat[:n_paths].astype(np.complex128))

    if amp_slice.shape[0] != n_paths:
        raise ValueError(
            f"路径增益长度 {amp_slice.shape[0]} 与 tau 路径数 {n_paths} 不一致"
        )
    return tau_slice, valid_slice, amp_slice


def _squeeze_cfr_to_sf(cfr: np.ndarray | torch.Tensor) -> torch.Tensor:
    h = torch.as_tensor(cfr)
    if h.ndim == 2:
        return h.to(dtype=torch.complex64)
    while h.ndim > 2 and h.shape[0] == 1:
        h = h.squeeze(0)
    if h.ndim != 2:
        h = h.reshape(-1, h.shape[-2], h.shape[-1])[0]
    return h.to(dtype=torch.complex64)


def check_los_path(
    rt_scene: object,
    true_range_m: float,
    *,
    cfg: SampleQualityConfig,
) -> SampleQualityResult:
    """检查与几何斜距一致的 RT 路径是否存在且足够强。"""
    if not cfg.require_los:
        return SampleQualityResult(passed=True)

    rx_i, tx_i = cfg.rx_idx, cfg.tx_idx
    tau_slice, valid_slice, amp_slice = _extract_path_slices(
        rt_scene, rx_idx=rx_i, tx_idx=tx_i
    )

    candidates = np.flatnonzero(valid_slice & (tau_slice >= 0.0))
    if candidates.size == 0:
        candidates = np.flatnonzero(valid_slice)
    if candidates.size == 0:
        return SampleQualityResult(passed=False, reason="no_valid_paths")

    tau_geo = 2.0 * float(true_range_m) / c
    err = np.abs(tau_slice[candidates] - tau_geo)
    best = int(candidates[int(np.argmin(err))])

    amps = amp_slice[candidates]
    max_amp = float(np.max(amps))
    nearest_amp = float(amp_slice[best])
    if max_amp <= cfg.eps:
        return SampleQualityResult(passed=False, reason="no_valid_paths")

    los_ratio = nearest_amp / max_amp
    if los_ratio < cfg.min_los_ratio:
        return SampleQualityResult(
            passed=False,
            reason="weak_los",
            los_ratio=los_ratio,
        )
    return SampleQualityResult(passed=True, los_ratio=los_ratio)


def _geometry_bins(
    true_range_m: float,
    true_velocity_mps: float,
    sensing_performance: SensingPerformance,
) -> tuple[int, int]:
    """由几何真值估计时延–多普勒 bin（单基地）。"""
    sp = sensing_performance
    delay_bin = int(round(float(true_range_m) / sp.range_resolution))
    delay_bin = int(np.clip(delay_bin, 0, sp.rg.fft_size - 1))

    fd_hz = float(true_velocity_mps) * 2.0 * sp.carrier_frequency / c
    doppler_bins = sp.doppler_bins
    doppler_bin = int(np.argmin(np.abs(doppler_bins - fd_hz)))
    doppler_bin = int(np.clip(doppler_bin, 0, sp.rg.num_ofdm_symbols - 1))
    return delay_bin, doppler_bin


def _dd_peak_result_from_magnitude(
    mag: np.ndarray,
    true_range_m: float,
    true_velocity_mps: float,
    sensing_performance: SensingPerformance,
    *,
    cfg: SampleQualityConfig,
) -> SampleQualityResult:
    """由幅度谱 ``(多普勒, 时延)`` 执行峰突出度与对齐检查。"""

    exp_delay, exp_doppler = _geometry_bins(
        true_range_m, true_velocity_mps, sensing_performance
    )
    n_dop, n_delay = mag.shape
    half = cfg.max_bin_offset

    dop_lo = max(0, exp_doppler - half)
    dop_hi = min(n_dop, exp_doppler + half + 1)
    del_lo = max(0, exp_delay - half)
    del_hi = min(n_delay, exp_delay + half + 1)

    region = mag[dop_lo:dop_hi, del_lo:del_hi]
    if region.size == 0:
        return SampleQualityResult(passed=False, reason="low_peak_prominence")

    peak_val = float(region.max())
    mean_val = float(mag.mean())
    prominence_db = 20.0 * np.log10(max(peak_val, cfg.eps) / max(mean_val, cfg.eps))

    if prominence_db < cfg.min_peak_prominence_db:
        return SampleQualityResult(
            passed=False,
            reason="low_peak_prominence",
            peak_prominence_db=prominence_db,
        )

    flat_idx = int(np.argmax(region))
    rel_dop, rel_del = np.unravel_index(flat_idx, region.shape)
    peak_dop = dop_lo + int(rel_dop)
    peak_del = del_lo + int(rel_del)
    dop_off = abs(peak_dop - exp_doppler)
    del_off = abs(peak_del - exp_delay)

    if dop_off > cfg.max_bin_offset or del_off > cfg.max_bin_offset:
        return SampleQualityResult(
            passed=False,
            reason="peak_misaligned",
            peak_prominence_db=prominence_db,
            peak_doppler_offset=dop_off,
            peak_delay_offset=del_off,
        )

    return SampleQualityResult(
        passed=True,
        peak_prominence_db=prominence_db,
        peak_doppler_offset=dop_off,
        peak_delay_offset=del_off,
    )


def check_dd_peak(
    cfr: np.ndarray | torch.Tensor | None,
    true_range_m: float,
    true_velocity_mps: float,
    sensing_performance: SensingPerformance,
    *,
    cfg: SampleQualityConfig,
    device: torch.device | None = None,
    h_dd: torch.Tensor | None = None,
) -> SampleQualityResult:
    """检查时延–多普勒谱在几何 bin 附近是否有足够突出的峰。"""
    if h_dd is None:
        if cfr is None:
            raise ValueError("check_dd_peak 需要 cfr 或 h_dd 之一")
        dev = device or torch.device("cpu")
        dd = DelayDopplerSpectrum(sensing_performance, device=dev)
        h_sf = _squeeze_cfr_to_sf(cfr)
        h_dd = dd(h_sf)
    mag = torch.abs(h_dd).detach().cpu().numpy()
    return _dd_peak_result_from_magnitude(
        mag,
        true_range_m,
        true_velocity_mps,
        sensing_performance,
        cfg=cfg,
    )


def evaluate_sample_quality(
    rt_scene: object,
    cfr: np.ndarray | torch.Tensor | None,
    true_range_m: float,
    true_velocity_mps: float,
    sensing_performance: SensingPerformance,
    *,
    cfg: SampleQualityConfig | None = None,
    device: torch.device | None = None,
) -> SampleQualityResult:
    """组合 LoS 与 DD 谱峰检查；任一失败即拒绝。"""
    qcfg = cfg or SampleQualityConfig()
    los = check_los_path(rt_scene, true_range_m, cfg=qcfg)
    if not los.passed:
        return los
    dd = check_dd_peak(
        cfr,
        true_range_m,
        true_velocity_mps,
        sensing_performance,
        cfg=qcfg,
        device=device,
    )
    if not dd.passed:
        return dd
    return SampleQualityResult(
        passed=True,
        los_ratio=los.los_ratio,
        peak_prominence_db=dd.peak_prominence_db,
        peak_doppler_offset=dd.peak_doppler_offset,
        peak_delay_offset=dd.peak_delay_offset,
    )
