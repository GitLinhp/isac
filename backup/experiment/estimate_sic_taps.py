#!/usr/bin/env python3
"""Estimate fixed FIR taps for time-domain self-interference cancellation (SIC).

Reads raw complex64 TX reference and RX IQ captured by the calibration GRC
(``sic_tap_calibration`` → ``sic_cal_tx.dat`` / ``sic_cal_rx.dat``), aligns TX/RX
per CPI burst (default ``burst_len=5120``), then solves least-squares for ``h`` in
``rx[n] ≈ sum_k h[k] * tx[n-k]`` (same as GNU Radio ``fir_filter_ccc``).

By default only the first ``--max-samples`` IQ samples are loaded and used, so long
calibration recordings at 245.76 MS/s do not require manual truncation.

Output ``sic_taps.npy`` is loaded by collection flowgraphs at runtime.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COMPLEX_DTYPE = np.complex64
_DEFAULT_MAX_SAMPLES = 10_000_000
_DEFAULT_MAX_LS_SAMPLES = 200_000
_DEFAULT_MAX_LAG = 5120
_DEFAULT_BURST_LEN = 5120
_DEFAULT_LAG_HINT = 278
_DEFAULT_MIN_CANCEL_DB = 3.0
_DEFAULT_SAMP_RATE = 245_760_000.0
_MIN_NORM_CORR = 0.05
_MIN_BURSTS_USED = 10
_MIN_DURATION_MS = 100.0


@dataclass
class EstimationStats:
    h: np.ndarray
    lag: int
    ls_pre: float
    ls_post: float
    rt_pre: float
    rt_post: float
    lag_peak: float
    samples_used: int
    norm_corr: float
    burst_count: int = 0
    bursts_used: int = 0
    burst_len: int = 0
    lag_median: float = 0.0
    lag_std: float = 0.0
    burst_align: bool = False


def read_complex64_iq(
    path: str | Path,
    *,
    max_samples: int | None,
) -> tuple[np.ndarray, int]:
    """Load up to ``max_samples`` complex64 IQ samples; return ``(segment, total_in_file)``."""
    path = Path(path)
    itemsize = np.dtype(_COMPLEX_DTYPE).itemsize
    total = path.stat().st_size // itemsize
    if total == 0:
        raise ValueError(f"empty IQ file: {path}")

    count = total if max_samples is None else min(total, max_samples)
    data = np.fromfile(path, dtype=_COMPLEX_DTYPE, count=count)
    if data.size == 0:
        raise ValueError(f"empty IQ file: {path}")
    return data, total


def _lag_search_range(
    n: int,
    *,
    max_lag: int,
    lag_hint: int | None,
) -> range:
    if lag_hint is None:
        lo, hi = -max_lag, max_lag
    else:
        lo, hi = lag_hint - max_lag, lag_hint + max_lag
    lo = max(lo, -(n - 1))
    hi = min(hi, n - 1)
    if lo > hi:
        raise ValueError(f"no valid lag range for segment length {n}, max_lag={max_lag}")
    return range(lo, hi + 1)


def _segments_at_lag(rx: np.ndarray, tx: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(rx.size, tx.size)
    if lag >= 0:
        return rx[: n - lag], tx[lag:n]
    shift = -lag
    return rx[shift:n], tx[: n - shift]


def normalized_correlation(rx: np.ndarray, tx: np.ndarray, lag: int) -> float:
    seg_rx, seg_tx = _segments_at_lag(rx, tx, lag)
    if seg_rx.size == 0:
        return 0.0
    denom = float(np.linalg.norm(seg_rx) * np.linalg.norm(seg_tx))
    if denom <= 0.0:
        return 0.0
    return float(np.abs(np.vdot(seg_rx, seg_tx)) / denom)


def find_coarse_lag(
    rx: np.ndarray,
    tx: np.ndarray,
    *,
    max_lag: int,
    lag_hint: int | None = None,
) -> tuple[int, float]:
    """Return ``(lag, peak|corr|)``; positive lag means rx leads tx."""
    n = min(rx.size, tx.size)
    if n <= 1:
        raise ValueError(f"segment too short for lag search: {n}")

    effective_max = min(max_lag, n - 1)
    best_lag = 0
    best_val = -1.0
    for lag in _lag_search_range(n, max_lag=effective_max, lag_hint=lag_hint):
        seg_rx, seg_tx = _segments_at_lag(rx, tx, lag)
        if seg_rx.size == 0:
            continue
        val = float(np.abs(np.vdot(seg_rx, seg_tx)))
        if val > best_val:
            best_val = val
            best_lag = lag
    return best_lag, best_val


def align_by_lag(rx: np.ndarray, tx: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        tx_aligned = tx[lag:]
        rx_aligned = rx[: tx_aligned.size]
    else:
        shift = -lag
        rx_aligned = rx[shift:]
        tx_aligned = tx[: rx_aligned.size]
    m = min(tx_aligned.size, rx_aligned.size)
    return tx_aligned[:m], rx_aligned[:m]


def build_toeplitz(tx: np.ndarray, num_taps: int, start: int) -> np.ndarray:
    """Build ``X`` for causal FIR: ``(X @ h)[i] ≈ sum_k h[k] * tx[start + num_taps - 1 + i - k]``."""
    n_rows = tx.size - start - num_taps + 1
    if n_rows <= num_taps:
        raise ValueError(
            f"not enough samples for LS (need > 2*num_taps); "
            f"tx_len={tx.size}, start={start}, num_taps={num_taps}"
        )
    cols = [
        tx[start + (num_taps - 1 - k) : start + (num_taps - 1 - k) + n_rows]
        for k in range(num_taps)
    ]
    return np.column_stack(cols)


def apply_causal_fir(tx: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal FIR matching GNU Radio ``fir_filter_ccc``: ``y[n] = sum_k h[k]*tx[n-k]``."""
    h = np.asarray(h, dtype=np.complex128)
    tx = np.asarray(tx, dtype=np.complex128)
    num_taps = h.size
    out = np.zeros(tx.size, dtype=np.complex128)
    for n in range(num_taps - 1, tx.size):
        out[n] = np.dot(h, tx[n - num_taps + 1 : n + 1][::-1])
    return out


def cancellation_db(pre_power: float, post_power: float) -> float:
    if post_power <= 0.0 or pre_power <= 0.0:
        return float("inf")
    return 10.0 * np.log10(pre_power / post_power)


def evaluate_cancellation_db(rx: np.ndarray, tx: np.ndarray, h: np.ndarray) -> tuple[float, float, float]:
    """Return ``(pre_power, post_power, cancel_db)`` on valid FIR output indices."""
    y_hat = apply_causal_fir(tx, h)
    num_taps = h.size
    rx_seg = rx[num_taps - 1 :]
    y_hat = y_hat[num_taps - 1 :]
    n_eval = min(50_000, rx_seg.size)
    rx_seg = rx_seg[:n_eval]
    y_hat = y_hat[:n_eval]
    residual = rx_seg - y_hat
    pre_power = float(np.mean(np.abs(rx_seg) ** 2))
    post_power = float(np.mean(np.abs(residual) ** 2))
    return pre_power, post_power, cancellation_db(pre_power, post_power)


def fit_ls_on_aligned(
    tx_aligned: np.ndarray,
    rx_aligned: np.ndarray,
    *,
    num_taps: int,
) -> tuple[np.ndarray, float, float]:
    """Return ``(h, ls_pre, ls_post)`` from aligned segments."""
    start = 0
    x_mat = build_toeplitz(tx_aligned, num_taps, start)
    y_vec = rx_aligned[start + num_taps - 1 : start + num_taps - 1 + x_mat.shape[0]]
    h, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    h = h.astype(_COMPLEX_DTYPE)

    n_eval = min(50_000, y_vec.size)
    y_hat = x_mat[:n_eval] @ h
    y_act = y_vec[:n_eval]
    y_res = y_act - y_hat
    ls_pre = float(np.mean(np.abs(y_act) ** 2))
    ls_post = float(np.mean(np.abs(y_res) ** 2))
    return h, ls_pre, ls_post


def estimate_fir_taps(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    num_taps: int,
    max_lag: int = _DEFAULT_MAX_LAG,
    max_ls_samples: int = _DEFAULT_MAX_LS_SAMPLES,
    max_samples: int | None = _DEFAULT_MAX_SAMPLES,
    lag_hint: int | None = _DEFAULT_LAG_HINT,
) -> EstimationStats:
    """Global lag search + single LS segment (legacy path)."""
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    n = min(tx.size, rx.size)
    if max_samples is not None:
        n = min(n, max_samples)
    tx = tx[:n].astype(np.complex128, copy=False)
    rx = rx[:n].astype(np.complex128, copy=False)

    print("Estimating coarse lag (global)...", flush=True)
    lag, lag_peak = find_coarse_lag(rx, tx, max_lag=max_lag, lag_hint=lag_hint)
    tx_aligned, rx_aligned = align_by_lag(rx, tx, lag)

    if tx_aligned.size > max_ls_samples + num_taps:
        tx_aligned = tx_aligned[: max_ls_samples + num_taps]
        rx_aligned = rx_aligned[: max_ls_samples + num_taps]

    print("Estimating FIR taps (LS)...", flush=True)
    h, ls_pre, ls_post = fit_ls_on_aligned(tx_aligned, rx_aligned, num_taps=num_taps)
    rt_pre, rt_post, _ = evaluate_cancellation_db(rx_aligned, tx_aligned, h)
    norm_corr = normalized_correlation(rx, tx, lag)

    return EstimationStats(
        h=h,
        lag=lag,
        ls_pre=ls_pre,
        ls_post=ls_post,
        rt_pre=rt_pre,
        rt_post=rt_post,
        lag_peak=lag_peak,
        samples_used=n,
        norm_corr=norm_corr,
        burst_align=False,
    )


def estimate_fir_taps_burst(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    num_taps: int,
    burst_len: int,
    max_lag: int = _DEFAULT_MAX_LAG,
    max_samples: int | None = _DEFAULT_MAX_SAMPLES,
    lag_hint: int | None = _DEFAULT_LAG_HINT,
    min_norm_corr: float = _MIN_NORM_CORR,
) -> EstimationStats:
    """Per-burst lag search, LS, and weighted merge of ``h``."""
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    n = min(tx.size, rx.size)
    if max_samples is not None:
        n = min(n, max_samples)
    tx = tx[:n].astype(np.complex128, copy=False)
    rx = rx[:n].astype(np.complex128, copy=False)

    print("Global coarse lag (pre-burst)...", flush=True)
    global_lag, global_lag_peak = find_coarse_lag(rx, tx, max_lag=max_lag, lag_hint=lag_hint)
    tx, rx = align_by_lag(rx, tx, global_lag)
    print(f"Global lag: {global_lag} samples (peak |corr|={global_lag_peak:.6e})", flush=True)

    burst_count = min(tx.size, rx.size) // burst_len
    if burst_count < 2:
        raise ValueError(f"need at least 2 bursts of length {burst_len}, got {burst_count}")

    per_burst_max_lag = min(256, burst_len // 4)
    burst_lags: list[int] = []
    h_accum = np.zeros(num_taps, dtype=np.complex128)
    weight_sum = 0.0
    bursts_used = 0

    print(f"Estimating per-burst lag + LS ({burst_count} bursts)...", flush=True)
    for b in range(burst_count):
        tx_b = tx[b * burst_len : (b + 1) * burst_len]
        rx_b = rx[b * burst_len : (b + 1) * burst_len]
        hint = burst_lags[-1] if burst_lags else 0
        lag_b, _ = find_coarse_lag(rx_b, tx_b, max_lag=per_burst_max_lag, lag_hint=hint)
        norm_corr_b = normalized_correlation(rx_b, tx_b, lag_b)
        if norm_corr_b < min_norm_corr:
            continue

        tx_a, rx_a = align_by_lag(rx_b, tx_b, lag_b)
        if tx_a.size < num_taps * 2:
            continue

        try:
            h_b, ls_pre_b, ls_post_b = fit_ls_on_aligned(tx_a, rx_a, num_taps=num_taps)
        except ValueError:
            continue

        if cancellation_db(ls_pre_b, ls_post_b) < 0.0:
            continue

        weight = ls_pre_b
        if weight <= 0.0:
            continue

        h_accum += weight * h_b.astype(np.complex128)
        weight_sum += weight
        burst_lags.append(lag_b)
        bursts_used += 1

    if bursts_used == 0 or weight_sum <= 0.0:
        raise ValueError(
            f"no usable bursts (need norm_corr>={min_norm_corr} and LS cancel>=0 dB); "
            f"checked {burst_count} bursts"
        )

    h = (h_accum / weight_sum).astype(_COMPLEX_DTYPE)
    lag = global_lag
    lag_std = float(np.std(burst_lags)) if len(burst_lags) > 1 else 0.0

    print("Evaluating merged taps on globally aligned segment...", flush=True)
    tx_aligned, rx_aligned = tx, rx
    if tx_aligned.size < num_taps * 2:
        raise ValueError("aligned segment too short after median lag")

    start = 0
    x_mat = build_toeplitz(tx_aligned, num_taps, start)
    y_vec = rx_aligned[start + num_taps - 1 : start + num_taps - 1 + x_mat.shape[0]]
    n_eval = min(50_000, y_vec.size)
    y_hat = x_mat[:n_eval] @ h
    y_act = y_vec[:n_eval]
    y_res = y_act - y_hat
    ls_pre = float(np.mean(np.abs(y_act) ** 2))
    ls_post = float(np.mean(np.abs(y_res) ** 2))

    rt_pre, rt_post, _ = evaluate_cancellation_db(rx_aligned, tx_aligned, h)
    norm_corr = normalized_correlation(rx_aligned, tx_aligned, 0)

    return EstimationStats(
        h=h,
        lag=lag,
        ls_pre=ls_pre,
        ls_post=ls_post,
        rt_pre=rt_pre,
        rt_post=rt_post,
        lag_peak=global_lag_peak,
        samples_used=min(tx.size, rx.size),
        norm_corr=norm_corr,
        burst_count=burst_count,
        bursts_used=bursts_used,
        burst_len=burst_len,
        lag_median=float(np.median(burst_lags)) if burst_lags else float(global_lag),
        lag_std=lag_std,
        burst_align=True,
    )


def print_stats(stats: EstimationStats, *, samp_rate: float) -> None:
    ls_cancel_db = cancellation_db(stats.ls_pre, stats.ls_post)
    rt_cancel_db = cancellation_db(stats.rt_pre, stats.rt_post)
    peak_tap = int(np.argmax(np.abs(stats.h)))
    est_duration_ms = stats.samples_used / samp_rate * 1e3

    print(f"Samples used for estimation: {stats.samples_used} ({est_duration_ms:.2f} ms)")
    if stats.burst_align:
        print(f"Burst count: {stats.burst_count} (burst_len={stats.burst_len})")
        print(f"Bursts used for LS: {stats.bursts_used} / {stats.burst_count}")
        print(f"Per-burst lag: median={stats.lag_median:.0f}, std={stats.lag_std:.1f}")
    print(f"Coarse lag: {stats.lag} samples (peak |corr|={stats.lag_peak:.6e})")
    print(f"Norm |corr| at best lag: {stats.norm_corr:.4f}")
    print(f"Mean |rx|^2 before (LS segment): {stats.ls_pre:.6e}")
    print(f"Mean |residual|^2 after (LS segment): {stats.ls_post:.6e}")
    print(f"Cancellation (LS segment): {ls_cancel_db:.1f} dB")
    print(f"Mean |rx|^2 before (runtime FIR): {stats.rt_pre:.6e}")
    print(f"Mean |residual|^2 after (runtime FIR): {stats.rt_post:.6e}")
    print(f"Cancellation (runtime FIR): {rt_cancel_db:.1f} dB")
    print(f"Peak |h|: {np.max(np.abs(stats.h)):.6e} at tap {peak_tap}")

    if stats.norm_corr < _MIN_NORM_CORR:
        print(
            "WARNING: Norm |corr| very low; TX/RX have little linear relationship. "
            "Re-record calibration (delete old .dat, steady-state 2-5 s, then enable Cal Record).",
            file=sys.stderr,
        )
    if stats.burst_align and stats.bursts_used < _MIN_BURSTS_USED:
        print(
            f"WARNING: only {stats.bursts_used} bursts passed quality checks "
            f"(need >= {_MIN_BURSTS_USED}). Re-record calibration.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx", type=Path, required=True, help="TX reference complex64 .dat")
    parser.add_argument("--rx", type=Path, required=True, help="RX IQ complex64 .dat")
    parser.add_argument(
        "--num-taps",
        type=int,
        default=64,
        help="FIR length (must match GRC sic_num_taps)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=_DEFAULT_MAX_LAG,
        help="Search range for coarse TX/RX alignment (samples)",
    )
    parser.add_argument(
        "--lag-hint",
        type=int,
        default=_DEFAULT_LAG_HINT,
        help=f"Expected lag center for search window (default: {_DEFAULT_LAG_HINT}, num_delay_samp)",
    )
    parser.add_argument(
        "--no-lag-hint",
        action="store_true",
        help="Search full [-max_lag, max_lag] instead of lag_hint +/- max_lag",
    )
    parser.add_argument(
        "--burst-len",
        type=int,
        default=_DEFAULT_BURST_LEN,
        help=f"CPI burst length in samples (default: {_DEFAULT_BURST_LEN})",
    )
    burst_group = parser.add_mutually_exclusive_group()
    burst_group.add_argument(
        "--burst-align",
        dest="burst_align",
        action="store_true",
        default=True,
        help="Enable per-burst lag + LS (default)",
    )
    burst_group.add_argument(
        "--no-burst-align",
        dest="burst_align",
        action="store_false",
        help="Use global lag search only (legacy)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_DEFAULT_MAX_SAMPLES,
        help=(
            "Max IQ samples to load and use for lag/LS estimation "
            f"(default: {_DEFAULT_MAX_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--max-ls-samples",
        type=int,
        default=_DEFAULT_MAX_LS_SAMPLES,
        help=f"Max aligned samples for global LS solve (default: {_DEFAULT_MAX_LS_SAMPLES})",
    )
    parser.add_argument(
        "--min-cancel-db",
        type=float,
        default=_DEFAULT_MIN_CANCEL_DB,
        help=(
            "Minimum runtime-style cancellation dB required before saving taps "
            f"(default: {_DEFAULT_MIN_CANCEL_DB})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path (default: alongside --rx ../sic_taps.npy)",
    )
    parser.add_argument(
        "--samp-rate",
        type=float,
        default=_DEFAULT_SAMP_RATE,
        help="Sample rate (Hz); used only for duration printout",
    )
    args = parser.parse_args(argv)

    tx, tx_total = read_complex64_iq(args.tx, max_samples=args.max_samples)
    rx, rx_total = read_complex64_iq(args.rx, max_samples=args.max_samples)
    output = args.output
    if output is None:
        output = args.rx.resolve().parent.parent / "sic_taps.npy"

    used = min(tx.size, rx.size)
    truncated = (tx_total > used) or (rx_total > used)
    duration_ms = used / args.samp_rate * 1e3
    lag_hint = None if args.no_lag_hint else args.lag_hint

    print(f"File samples (tx/rx): {tx_total} / {rx_total}")
    print(
        f"Using first {used} samples for estimation "
        f"({duration_ms:.2f} ms @ {args.samp_rate / 1e6:.3f} MS/s)"
    )
    if truncated:
        print(f"Truncated: yes (limit --max-samples={args.max_samples})")
    else:
        print("Truncated: no")

    tx_power = float(np.mean(np.abs(tx[:used].astype(np.complex128)) ** 2))
    rx_power = float(np.mean(np.abs(rx[:used].astype(np.complex128)) ** 2))
    power_ratio = rx_power / tx_power if tx_power > 0.0 else 0.0
    print(f"Mean |tx|^2: {tx_power:.6e}, Mean |rx|^2: {rx_power:.6e}, RX/TX ratio: {power_ratio:.6f}")
    if duration_ms < _MIN_DURATION_MS:
        print(
            f"WARNING: recording duration {duration_ms:.2f} ms < {_MIN_DURATION_MS:.0f} ms; "
            "re-record with Cal Record enabled for >=2 s after steady-state warmup.",
            file=sys.stderr,
        )

    use_burst = args.burst_align and used >= 2 * args.burst_len
    if args.burst_align and not use_burst:
        print(
            f"WARNING: too few samples for burst align (need >= {2 * args.burst_len}); "
            "falling back to global lag search.",
            file=sys.stderr,
        )

    try:
        if use_burst:
            stats = estimate_fir_taps_burst(
                tx,
                rx,
                num_taps=args.num_taps,
                burst_len=args.burst_len,
                max_lag=args.max_lag,
                max_samples=args.max_samples,
                lag_hint=lag_hint,
            )
        else:
            stats = estimate_fir_taps(
                tx,
                rx,
                num_taps=args.num_taps,
                max_lag=args.max_lag,
                max_ls_samples=args.max_ls_samples,
                max_samples=args.max_samples,
                lag_hint=lag_hint,
            )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_stats(stats, samp_rate=args.samp_rate)
    rt_cancel_db = cancellation_db(stats.rt_pre, stats.rt_post)

    if rt_cancel_db < args.min_cancel_db:
        print(
            f"WARNING: runtime FIR cancellation {rt_cancel_db:.1f} dB "
            f"< --min-cancel-db={args.min_cancel_db:.1f}; "
            "re-record calibration with matching GRC params or check TX/RX alignment.",
            file=sys.stderr,
        )
        print(f"NOT saving taps to {output}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, stats.h)
    print(f"FIR taps: {stats.h.size} -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
