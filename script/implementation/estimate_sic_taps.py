#!/usr/bin/env python3
"""Estimate fixed FIR taps for time-domain self-interference cancellation (SIC).

Reads raw complex64 TX reference and RX IQ captured by the calibration GRC
(``sic_tap_calibration`` → ``sic_cal_tx.dat`` / ``sic_cal_rx.dat``), finds coarse delay via cross-correlation,
then solves least-squares for ``h`` in ``rx ≈ conv(tx, h)``.

Output ``sic_taps.npy`` is loaded by ``range_profile_collection.py`` at flowgraph start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_COMPLEX_DTYPE = np.complex64


def read_complex64_iq(path: str | Path) -> np.ndarray:
    """Load a raw complex64 IQ file (no header)."""
    data = np.fromfile(path, dtype=_COMPLEX_DTYPE)
    if data.size == 0:
        raise ValueError(f"empty IQ file: {path}")
    return data


def find_coarse_lag(
    rx: np.ndarray,
    tx: np.ndarray,
    *,
    max_lag: int,
) -> int:
    """Return lag ``d`` that maximizes ``|corr(rx, tx delayed by d)|`` (positive = rx leads tx)."""
    n = min(rx.size, tx.size)
    rx = rx[:n]
    tx = tx[:n]
    corr = np.correlate(rx, tx, mode="full")
    lags = np.arange(-(n - 1), n, dtype=np.int64)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(mask):
        raise ValueError(f"max_lag={max_lag} too small for segment length {n}")
    idx = int(np.argmax(np.abs(corr[mask])))
    return int(lags[mask][idx])


def build_toeplitz(tx: np.ndarray, num_taps: int, start: int) -> np.ndarray:
    """Build ``X`` such that ``X @ h ≈ rx[start : start + n_rows]``."""
    n_rows = tx.size - start - num_taps + 1
    if n_rows <= num_taps:
        raise ValueError(
            f"not enough samples for LS (need > 2*num_taps); "
            f"tx_len={tx.size}, start={start}, num_taps={num_taps}"
        )
    cols = [tx[start + k : start + k + n_rows] for k in range(num_taps)]
    return np.column_stack(cols)


def estimate_fir_taps(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    num_taps: int,
    max_lag: int = 512,
    max_ls_samples: int = 200_000,
) -> tuple[np.ndarray, int, float, float]:
    """Return ``(h, coarse_lag, pre_power, post_power)`` on aligned segment."""
    n = min(tx.size, rx.size)
    tx = tx[:n].astype(np.complex128, copy=False)
    rx = rx[:n].astype(np.complex128, copy=False)

    lag = find_coarse_lag(rx, tx, max_lag=max_lag)
    if lag >= 0:
        tx_aligned = tx[lag:]
        rx_aligned = rx[: tx_aligned.size]
    else:
        shift = -lag
        rx_aligned = rx[shift:]
        tx_aligned = tx[: rx_aligned.size]

    m = min(tx_aligned.size, rx_aligned.size)
    tx_aligned = tx_aligned[:m]
    rx_aligned = rx_aligned[:m]

    if m > max_ls_samples + num_taps:
        tx_aligned = tx_aligned[: max_ls_samples + num_taps]
        rx_aligned = rx_aligned[: max_ls_samples + num_taps]

    start = 0
    x_mat = build_toeplitz(tx_aligned, num_taps, start)
    y_vec = rx_aligned[start + num_taps - 1 : start + num_taps - 1 + x_mat.shape[0]]
    h, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    h = h.astype(_COMPLEX_DTYPE)

    n_eval = min(50_000, y_vec.size)
    y_hat = x_mat[:n_eval] @ h
    y_act = y_vec[:n_eval]
    y_res = y_act - y_hat
    pre_power = float(np.mean(np.abs(y_act) ** 2))
    post_power = float(np.mean(np.abs(y_res) ** 2))
    return h, lag, pre_power, post_power


def cancellation_db(pre_power: float, post_power: float) -> float:
    if post_power <= 0.0 or pre_power <= 0.0:
        return float("inf")
    return 10.0 * np.log10(pre_power / post_power)


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
        default=512,
        help="Search range for coarse TX/RX alignment (samples)",
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
        default=122_880_000.0,
        help="Sample rate (Hz); used only for duration printout",
    )
    args = parser.parse_args(argv)

    tx = read_complex64_iq(args.tx)
    rx = read_complex64_iq(args.rx)
    output = args.output
    if output is None:
        output = args.rx.resolve().parent.parent / "sic_taps.npy"

    h, lag, pre_p, post_p = estimate_fir_taps(
        tx,
        rx,
        num_taps=args.num_taps,
        max_lag=args.max_lag,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, h)

    n = min(tx.size, rx.size)
    duration_s = n / args.samp_rate
    print(f"Samples used: {n} ({duration_s * 1e3:.2f} ms @ {args.samp_rate/1e6:.3f} MS/s)")
    print(f"Coarse lag: {lag} samples")
    print(f"FIR taps: {h.size} -> {output}")
    print(f"Mean |rx|^2 before: {pre_p:.6e}")
    print(f"Mean |residual|^2 after: {post_p:.6e}")
    print(f"Cancellation (LS segment): {cancellation_db(pre_p, post_p):.1f} dB")
    print(f"Peak |h|: {np.max(np.abs(h)):.6e} at tap {int(np.argmax(np.abs(h)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
