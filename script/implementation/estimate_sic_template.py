#!/usr/bin/env python3
"""Estimate Divide-domain SI template from ``sic_cal_divide.dat`` (OFDM Divide H(f) recording).

Reads complex64 vectors shaped ``(n_vectors, vlen)``, computes mean template ``t``,
validates range-profile cancellation dB, and saves ``sic_template.npy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gnuradio.fft import window

_COMPLEX_DTYPE = np.complex64
_DEFAULT_VLEN = 4096
_DEFAULT_TRANSPOSE_LEN = 2
_DEFAULT_MIN_CANCEL_DB = 3.0
_DEFAULT_MAX_VECTORS = 5000
_MIN_CPIS = 10


def read_divide_vectors(
    path: str | Path,
    *,
    vlen: int,
    max_vectors: int | None,
) -> tuple[np.ndarray, int]:
    path = Path(path)
    itemsize = np.dtype(_COMPLEX_DTYPE).itemsize
    total = path.stat().st_size // itemsize
    if total == 0:
        raise ValueError(f"empty divide file: {path}")
    if total % vlen != 0:
        raise ValueError(f"file size not multiple of vlen={vlen}: {path}")

    n_vectors_total = total // vlen
    n_vectors = n_vectors_total if max_vectors is None else min(n_vectors_total, max_vectors)
    flat = np.fromfile(path, dtype=_COMPLEX_DTYPE, count=n_vectors * vlen)
    return flat.reshape(n_vectors, vlen), n_vectors_total


def range_profile_power(h: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Non-coherent range power for one Divide vector (matches GR range FFT + |.|^2)."""
    spec = np.fft.fft(h.astype(np.complex128) * win, n=h.size)
    return np.abs(spec) ** 2


def mean_range_profile(
    vectors: np.ndarray,
    *,
    transpose_len: int,
    win: np.ndarray,
) -> np.ndarray:
    n_cpi = vectors.shape[0] // transpose_len
    if n_cpi == 0:
        raise ValueError("need at least one full CPI")
    prof = np.zeros(vectors.shape[1], dtype=np.float64)
    for c in range(n_cpi):
        base = c * transpose_len
        cpi_sum = np.zeros(vectors.shape[1], dtype=np.float64)
        for s in range(transpose_len):
            cpi_sum += range_profile_power(vectors[base + s], win)
        prof += cpi_sum / transpose_len
    return prof / n_cpi


def peak_db(prof: np.ndarray, *, peak_bin: int = 0, guard: int = 8) -> float:
    """Peak power at ``peak_bin`` relative to median of excluded guard region."""
    n = prof.size
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_bin - guard)
    hi = min(n, peak_bin + guard + 1)
    mask[lo:hi] = False
    ref = float(np.median(prof[mask])) if np.any(mask) else float(np.mean(prof))
    if ref <= 0.0:
        ref = 1e-30
    return float(10.0 * np.log10(max(prof[peak_bin], 1e-30) / ref))


def estimate_template(vectors: np.ndarray) -> np.ndarray:
    return np.mean(vectors.astype(np.complex128), axis=0).astype(_COMPLEX_DTYPE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate Divide-domain SIC template.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="sic_cal_divide.dat from SicDivideRecorder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sic_template.npy (default: alongside input ../sic_template.npy)",
    )
    parser.add_argument("--vlen", type=int, default=_DEFAULT_VLEN)
    parser.add_argument("--transpose-len", type=int, default=_DEFAULT_TRANSPOSE_LEN)
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=_DEFAULT_MAX_VECTORS,
        help=f"Max Divide vectors to load (default: {_DEFAULT_MAX_VECTORS})",
    )
    parser.add_argument(
        "--min-cancel-db",
        type=float,
        default=_DEFAULT_MIN_CANCEL_DB,
        help=f"Min 0 m peak drop dB to save template (default: {_DEFAULT_MIN_CANCEL_DB})",
    )
    args = parser.parse_args(argv)

    vectors, total = read_divide_vectors(
        args.input,
        vlen=args.vlen,
        max_vectors=args.max_vectors,
    )
    output = args.output
    if output is None:
        output = args.input.resolve().parent.parent / "sic_template.npy"

    n_cpi = vectors.shape[0] // args.transpose_len
    print(f"File vectors: {total}, using {vectors.shape[0]} (vlen={args.vlen})")
    print(f"CPIs used: {n_cpi} (transpose_len={args.transpose_len})")
    if n_cpi < _MIN_CPIS:
        print(
            f"WARNING: only {n_cpi} CPIs (< {_MIN_CPIS}); record longer calibration.",
            file=sys.stderr,
        )

    template = estimate_template(vectors)
    win = window.blackmanharris(args.vlen)

    prof_raw = mean_range_profile(vectors, transpose_len=args.transpose_len, win=win)
    cleaned = vectors - template[np.newaxis, :]
    prof_clean = mean_range_profile(cleaned, transpose_len=args.transpose_len, win=win)

    pre_db = peak_db(prof_raw)
    post_db = peak_db(prof_clean)
    cancel_db = pre_db - post_db

    print(f"0 m peak (raw):      {pre_db:.2f} dB above median")
    print(f"0 m peak (cleaned):  {post_db:.2f} dB above median")
    print(f"Cancellation (range profile): {cancel_db:.2f} dB")

    if cancel_db < args.min_cancel_db:
        print(
            f"ERROR: cancellation {cancel_db:.2f} dB < --min-cancel-db={args.min_cancel_db:.1f}; "
            "not saving template",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, template)
    print(f"Saved template shape {template.shape} -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
