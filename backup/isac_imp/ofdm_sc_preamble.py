"""Schmidl-Cox 风格 OFDM 同步前导：与 GNU Radio ``ofdm_txrx`` 同构。

用于 Style-1 突发外层：TX 在载荷 ``x_time`` 前拼接时域前导；RX 用已知模板
相关检测 timing，并用 sync1 两半相位估粗 CFO。前导不进入 LS 参考网格。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# 与 gnuradio.digital.ofdm_txrx._seq_seed 一致
_SEQ_SEED = 42

# 默认搜索跨度相对前导长度的倍数（约 32k @ fft=2048/cp=0）
SEARCH_SPAN_MULT = 8


def default_search_span(preamble_length: int) -> int:
    """短窗搜峰长度：``SEARCH_SPAN_MULT * preamble_len``。"""
    return max(int(preamble_length), int(SEARCH_SPAN_MULT) * int(preamble_length))


class PreambleDetect(NamedTuple):
    """前导检测结果：缓冲内起点下标与粗 CFO (Hz)。"""

    start_idx: int
    cfo_hz: float
    peak_metric: float


def preamble_tpl_rev_conj(template: np.ndarray) -> np.ndarray:
    """模板共轭翻转，供 FFT 线性相关复用（避免每拍重算）。"""
    tpl = np.asarray(template, dtype=np.complex64).ravel()
    return np.conj(tpl[::-1]).astype(np.complex64, copy=False)


def _active_carriers(fft_size: int) -> list[int]:
    """全带占用（除 DC），与 style1 ``n_carriers = fft_len - 2`` 一致。"""
    n = int(fft_size) - 2
    if n < 2 or n % 2:
        raise ValueError(f"fft_size={fft_size} 无法构成偶长度 occupied 集合")
    # 自然下标 0..fft-1：负载波映射到 fft+k
    carriers: list[int] = []
    for k in list(range(-n // 2, 0)) + list(range(1, n // 2 + 1)):
        carriers.append(k + fft_size if k < 0 else k)
    return carriers


def make_sync_word1(fft_size: int) -> np.ndarray:
    """频域 sync word 1（fftshift 后）：偶载波为 0，奇载波 BPSK·√2。"""
    fft_size = int(fft_size)
    active = set(_active_carriers(fft_size))
    rng = np.random.RandomState(_SEQ_SEED)
    bpsk = {0: np.sqrt(2.0), 1: -np.sqrt(2.0)}
    sw = np.zeros(fft_size, dtype=np.complex128)
    for x in range(fft_size):
        if x in active and (x % 2):
            sw[x] = bpsk[int(rng.randint(2))]
    return np.fft.fftshift(sw)


def make_sync_word2(fft_size: int) -> np.ndarray:
    """频域 sync word 2（fftshift 后）：占用载波 BPSK，DC 强制 0。"""
    fft_size = int(fft_size)
    active = set(_active_carriers(fft_size))
    rng = np.random.RandomState(_SEQ_SEED)
    bpsk = {0: 1.0, 1: -1.0}
    sw = np.zeros(fft_size, dtype=np.complex128)
    for x in range(fft_size):
        if x in active:
            sw[x] = bpsk[int(rng.randint(2))]
    sw[0] = 0.0
    return np.fft.fftshift(sw)


def _symbol_time(freq_fftshift: np.ndarray, cp_len: int) -> np.ndarray:
    """fftshift 频域符号 → IFFT → 加 CP → complex64。"""
    nat = np.fft.ifftshift(np.asarray(freq_fftshift, dtype=np.complex128))
    t = np.fft.ifft(nat)
    cp = int(cp_len)
    if cp > 0:
        t = np.concatenate([t[-cp:], t])
    return t.astype(np.complex64, copy=False)


def preamble_len(fft_size: int, cp_len: int) -> int:
    """时域前导样点数：2 * (fft_size + cp)。"""
    return 2 * (int(fft_size) + int(cp_len))


def preamble_time(fft_size: int, cp_len: int = 0) -> np.ndarray:
    """拼接 sync1|sync2 时域前导（含 CP）。"""
    s1 = _symbol_time(make_sync_word1(fft_size), cp_len)
    s2 = _symbol_time(make_sync_word2(fft_size), cp_len)
    return np.concatenate([s1, s2]).astype(np.complex64, copy=False)


def apply_cfo(x: np.ndarray, cfo_hz: float, samp_rate: float) -> np.ndarray:
    """对样点序列施加 ``exp(-j 2π f t)`` 补偿（就地安全的新数组）。"""
    if abs(float(cfo_hz)) < 1e-6 or samp_rate <= 0:
        return np.asarray(x, dtype=np.complex64)
    n = np.arange(len(x), dtype=np.float64)
    ph = np.exp(-1j * 2.0 * np.pi * float(cfo_hz) * n / float(samp_rate))
    return (np.asarray(x, dtype=np.complex64) * ph.astype(np.complex64)).astype(
        np.complex64, copy=False
    )


def estimate_cfo_hz(
    sync1_body: np.ndarray,
    fft_size: int,
    samp_rate: float,
) -> float:
    """由 sync1 有用符号两半相位估粗 CFO (Hz)。

    GR 奇载波 sync1 在零频偏时两半近似反相，故先乘 -1 再取角：
    ``Δf = angle(-P) / π * (samp_rate / fft_size)``。
    """
    fft_size = int(fft_size)
    half = fft_size // 2
    body = np.asarray(sync1_body, dtype=np.complex128).ravel()
    if body.size < fft_size:
        return 0.0
    a = body[:half]
    b = body[half:fft_size]
    p = np.vdot(a, b)  # sum conj(a)*b
    # 零 CFO 期望 P≈负实数 → 去掉 π
    phi = float(np.angle(-p))
    scs = float(samp_rate) / float(fft_size)
    return phi / np.pi * scs


def detect_preamble(
    buf: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.6,
    fft_size: int | None = None,
    cp_len: int = 0,
    samp_rate: float = 1.0,
    max_search: int | None = None,
    tpl_rev_conj: np.ndarray | None = None,
) -> PreambleDetect | None:
    """在 ``buf`` 上对已知前导做归一化相关，返回起点与粗 CFO。

    使用 ``complex64`` FFT 线性相关；峰值为
    ``|corr| / (||tpl|| * ||buf_win||)``，超过 ``threshold`` 才接受。

    若 ``max_search`` 给定且 ``buf`` 更长，只对**尾部** ``buf[-max_search:]``
    做相关，返回的 ``start_idx`` 已换算为相对完整 ``buf`` 的下标。
    ``tpl_rev_conj`` 可传入预计算的 ``conj(template[::-1])`` 以避免每拍重算。
    """
    buf_full = np.asarray(buf, dtype=np.complex64).ravel()
    tpl = np.asarray(template, dtype=np.complex64).ravel()
    m = int(tpl.size)
    if m <= 0 or buf_full.size < m:
        return None

    offset = 0
    work = buf_full
    if max_search is not None:
        ms = int(max_search)
        if ms < m:
            ms = m
        if buf_full.size > ms:
            offset = buf_full.size - ms
            work = buf_full[offset:]

    # FFT 相关：corr[k] = sum_i work[k+i] * conj(tpl[i])
    n_fft = 1 << int(np.ceil(np.log2(work.size + m - 1)))
    B = np.fft.fft(work, n_fft)
    if tpl_rev_conj is None:
        rev = np.conj(tpl[::-1]).astype(np.complex64, copy=False)
    else:
        rev = np.asarray(tpl_rev_conj, dtype=np.complex64).ravel()
        if rev.size != m:
            rev = np.conj(tpl[::-1]).astype(np.complex64, copy=False)
    T = np.fft.fft(rev, n_fft)
    corr_full = np.fft.ifft(B * T)
    # 线性相关 r[k] 对应 ifft 下标 k+m-1
    corr = corr_full[m - 1 : m - 1 + (work.size - m + 1)]

    energy_tpl = float(np.vdot(tpl, tpl).real)
    if energy_tpl <= 0:
        return None
    mag2 = (work.real.astype(np.float32) ** 2) + (work.imag.astype(np.float32) ** 2)
    csum = np.concatenate([[0.0], np.cumsum(mag2, dtype=np.float64)])
    win_e = csum[m:] - csum[:-m]
    denom = np.sqrt(np.maximum(win_e * energy_tpl, 1e-20))
    metric = np.abs(corr) / denom

    peak_local = int(np.argmax(metric))
    peak_v = float(metric[peak_local])
    if peak_v < float(threshold):
        return None

    peak_i = offset + peak_local
    fft_size = int(fft_size) if fft_size is not None else (m // 2 - int(cp_len))
    cp = int(cp_len)
    body0 = peak_i + cp
    body1 = body0 + fft_size
    if body1 > buf_full.size:
        cfo = 0.0
    else:
        cfo = estimate_cfo_hz(buf_full[body0:body1], fft_size, samp_rate)

    return PreambleDetect(start_idx=peak_i, cfo_hz=float(cfo), peak_metric=peak_v)
