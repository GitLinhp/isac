#!/usr/bin/env python3
"""发射端预计算各步骤 × 四后端时延基准（一次性启动墙钟，含 GR top_block 调度）。

步骤: build_freq_grid / fftshift / ifft / add_cp / ifft_cp / full_chain
后端: numpy / torch_cpu / torch_cuda / gr_cpp

full_chain = QPSK填栅格(含 fftshift) → IFFT+CP，对应完整预计算发射计算链路。

与实时稳态吞吐不可横比；公平 realtime 对比请用::

    PYTHONPATH=src python script/experiment/bench_tx_precompute_backends.py --mode realtime
    # 或
    PYTHONPATH=src python script/experiment/bench_tx_realtime_backends.py

示例::

    PYTHONPATH=src python script/experiment/bench_tx_precompute_backends.py
    PYTHONPATH=src python script/experiment/bench_tx_precompute_backends.py --fft-len 512 --repeat 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    step: str
    backend: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_err: float | None
    note: str = ""


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_call(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    cuda: bool = False,
) -> tuple[list[float], object]:
    """Return (times_ms, last_result)."""
    out: object = None
    for _ in range(max(0, warmup)):
        if cuda:
            _sync_cuda()
        out = fn()
        if cuda:
            _sync_cuda()
    times: list[float] = []
    for _ in range(repeat):
        if cuda:
            _sync_cuda()
        t0 = time.perf_counter()
        out = fn()
        if cuda:
            _sync_cuda()
        times.append((time.perf_counter() - t0) * 1e3)
    return times, out


def _stats(times: list[float]) -> tuple[float, float, float]:
    return statistics.median(times), float(statistics.fmean(times)), min(times)


def _max_err(ref: np.ndarray, other: np.ndarray) -> float:
    a = np.asarray(ref, dtype=np.complex128)
    b = np.asarray(other, dtype=np.complex128)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Shared OFDM helpers (NumPy / Torch)
# ---------------------------------------------------------------------------


def occupied_carrier_indices(fft_len: int) -> list[int]:
    """DC null + 两侧满占用：共 fft_len-2 个数据载波（未 shift 索引）。"""
    # 未 shift：DC=0；占用 1..N/2-1 与 N/2+1..N-1（跳过 Nyquist 也可，这里跳 DC 与最后一 guard）
    # 与流图 n_carriers=fft_len-2 一致：跳过 DC 与一个 guard（索引 fft_len//2）
    idxs = [i for i in range(fft_len) if i != 0 and i != fft_len // 2]
    assert len(idxs) == fft_len - 2
    return idxs


def qpsk_symbols_from_seed(n_sym: int, n_data: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_sym, n_data, 2), dtype=np.int8)
    # Gray QPSK: 00→+1+1j, 01→-1+1j, 11→-1-1j, 10→+1-1j  (scaled 1/sqrt(2))
    mI = 1 - 2 * bits[..., 0]
    mQ = 1 - 2 * bits[..., 1]
    return ((mI + 1j * mQ) / np.sqrt(2.0)).astype(np.complex64)


def map_to_grid_unshifted(
    symbols: np.ndarray, fft_len: int, occupied: list[int]
) -> np.ndarray:
    """symbols (n_sym, n_data) → grid (n_sym, fft_len) 未 fftshift。"""
    n_sym = symbols.shape[0]
    grid = np.zeros((n_sym, fft_len), dtype=np.complex64)
    grid[:, occupied] = symbols
    return grid


def numpy_ifft_batch(freq_shifted: np.ndarray) -> np.ndarray:
    """freq 已 fftshift；对齐 GR fft_vcc(IFFT, shift=True)。"""
    return np.fft.ifft(np.fft.ifftshift(freq_shifted, axes=-1), axis=-1, norm="forward").astype(
        np.complex64, copy=False
    )


def numpy_add_cp(td: np.ndarray, cp_len: int) -> np.ndarray:
    """td (n_sym, fft_len) → (n_sym, fft_len+cp_len)。"""
    return np.concatenate([td[:, -cp_len:], td], axis=-1).astype(np.complex64, copy=False)


def torch_ifft_batch(freq_shifted, device: str):
    import torch

    x = freq_shifted if isinstance(freq_shifted, torch.Tensor) else torch.as_tensor(
        freq_shifted, device=device
    )
    if x.device.type != device:
        x = x.to(device)
    # torch.fft.ifft default is norm="backward" (1/n); GR/FFTW unnormalized IFFT
    # ≡ numpy ifft(norm="forward") which has no 1/n on ifft... 
    # numpy: forward=ifft without 1/n, backward=ifft with 1/n
    # torch: forward=fft with 1/n, backward=ifft with 1/n, ortho=1/sqrt(n)
    # GR IFFT unnormalized = torch.fft.ifft(x, norm="forward")  # no 1/n on ifft
    y = torch.fft.ifft(torch.fft.ifftshift(x, dim=-1), dim=-1, norm="forward")
    return y


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def build_freq_grid_numpy(fft_len: int, n_sym: int, seed: int) -> np.ndarray:
    occ = occupied_carrier_indices(fft_len)
    syms = qpsk_symbols_from_seed(n_sym, len(occ), seed)
    unshifted = map_to_grid_unshifted(syms, fft_len, occ)
    return np.fft.fftshift(unshifted, axes=-1).astype(np.complex64, copy=False)


def build_freq_grid_torch(fft_len: int, n_sym: int, seed: int, device: str):
    import torch

    occ = occupied_carrier_indices(fft_len)
    # same RNG on CPU then move — fair bitgen cost on host; mapping on device
    syms = qpsk_symbols_from_seed(n_sym, len(occ), seed)
    grid = torch.zeros((n_sym, fft_len), dtype=torch.complex64, device=device)
    idx = torch.tensor(occ, device=device, dtype=torch.long)
    grid[:, idx] = torch.as_tensor(syms, device=device)
    return torch.fft.fftshift(grid, dim=-1)


def build_freq_grid_gr(
    fft_len: int, n_sym: int, seed: int
) -> np.ndarray:
    """C++: chunks_to_symbols + ofdm_carrier_allocator (output_is_shifted=True)."""
    from gnuradio import blocks, digital, gr

    occ = occupied_carrier_indices(fft_len)
    # allocator 索引：output_is_shifted=True 时常用“自然频域”占用表再输出 shift
    # 使用未 shift 占用表，让 allocator 输出与 fftshift(grid) 同形
    occupied = (tuple(occ),)
    pilots: tuple = ((),)
    pilot_sym: tuple = ((),)
    sync: tuple = ()

    # QPSK constellation (same scale)
    const = digital.constellation_qpsk().base()
    # bits from same seed → pack to bytes for chunks_to_symbols via bytes? 
    # Use complex vector source of QPSK symbols directly into allocator via
    # ofdm_carrier_allocator expecting complex symbols on occupied carriers.
    # Actually allocator input is complex stream of data symbols in order.
    syms = qpsk_symbols_from_seed(n_sym, len(occ), seed).reshape(-1)

    src = blocks.vector_source_c(syms.tolist(), False, 1)
    # Tag packet length in symbols for each OFDM symbol = n_data
    n_data = len(occ)
    # tagged stream: length tag every n_data samples
    tagger = blocks.stream_to_tagged_stream(
        gr.sizeof_gr_complex, 1, n_data, "packet_len"
    )
    alloc = digital.ofdm_carrier_allocator_cvc(
        fft_len,
        occupied,
        pilots,
        pilot_sym,
        sync,
        "packet_len",
        True,
    )
    snk = blocks.vector_sink_c(fft_len)
    tb = gr.top_block()
    tb.connect(src, tagger, alloc, snk)
    tb.run()
    out = np.asarray(snk.data(), dtype=np.complex64)
    return out.reshape(n_sym, fft_len)


def fftshift_numpy(freq_unshifted: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(freq_unshifted, axes=-1).astype(np.complex64, copy=False)


def fftshift_torch(freq_unshifted, device: str):
    import torch

    x = freq_unshifted if isinstance(freq_unshifted, torch.Tensor) else torch.as_tensor(
        freq_unshifted, dtype=torch.complex64, device=device
    )
    if x.device.type != ("cuda" if device.startswith("cuda") else "cpu"):
        x = x.to(device)
    return torch.fft.fftshift(x, dim=-1)


def fftshift_gr(freq_unshifted: np.ndarray) -> np.ndarray:
    """C++: vector↔stream + keep_m_in_n + stream_mux 半谱对调。"""
    from gnuradio import blocks, gr

    n_sym, n = freq_unshifted.shape
    half = n // 2
    flat = freq_unshifted.astype(np.complex64).reshape(-1)
    src = blocks.vector_source_c(flat.tolist(), False, n)
    v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, n)
    keep_hi = blocks.keep_m_in_n(gr.sizeof_gr_complex, half, n, half)
    keep_lo = blocks.keep_m_in_n(gr.sizeof_gr_complex, half, n, 0)
    mux = blocks.stream_mux(gr.sizeof_gr_complex, [half, half])
    s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, n)
    snk = blocks.vector_sink_c(n)
    tb = gr.top_block()
    tb.connect(src, v2s)
    tb.connect(v2s, keep_hi)
    tb.connect(v2s, keep_lo)
    tb.connect(keep_hi, (mux, 0))
    tb.connect(keep_lo, (mux, 1))
    tb.connect(mux, s2v, snk)
    tb.run()
    return np.asarray(snk.data(), dtype=np.complex64).reshape(n_sym, n)


def ifft_numpy(freq_shifted: np.ndarray) -> np.ndarray:
    return numpy_ifft_batch(freq_shifted)


def ifft_torch(freq_shifted, device: str):
    return torch_ifft_batch(freq_shifted, device)


def ifft_gr(freq_shifted: np.ndarray) -> np.ndarray:
    from gnuradio import blocks, fft, gr

    n_sym, n = freq_shifted.shape
    flat = freq_shifted.astype(np.complex64).reshape(-1)
    src = blocks.vector_source_c(flat.tolist(), False, n)
    iff = fft.fft_vcc(n, False, (), True, 1)
    snk = blocks.vector_sink_c(n)
    tb = gr.top_block()
    tb.connect(src, iff, snk)
    tb.run()
    return np.asarray(snk.data(), dtype=np.complex64).reshape(n_sym, n)


def add_cp_numpy(td: np.ndarray, cp_len: int) -> np.ndarray:
    return numpy_add_cp(td, cp_len)


def add_cp_torch(td, cp_len: int, device: str):
    import torch

    x = td if isinstance(td, torch.Tensor) else torch.as_tensor(td, device=device)
    if x.device.type != ("cuda" if device.startswith("cuda") else "cpu"):
        x = x.to(device)
    return torch.cat([x[..., -cp_len:], x], dim=-1)


def add_cp_gr(td: np.ndarray, cp_len: int) -> np.ndarray:
    from gnuradio import blocks, digital, gr

    n_sym, fft_len = td.shape
    sym_len = fft_len + cp_len
    flat = td.astype(np.complex64).reshape(-1)
    # 每符号单独 packet_len=1，避免整帧输出缓冲过大
    tags = [
        gr.tag_utils.python_to_tag(
            (i, pmt_intern_packet_len(), pmt_from_long(1), pmt_intern_src())
        )
        for i in range(n_sym)
    ]
    src = blocks.vector_source_c(flat.tolist(), False, fft_len, tags)
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "packet_len")
    cp.set_min_output_buffer(max(sym_len * 4, 8192))
    snk = blocks.vector_sink_c()
    tb = gr.top_block()
    tb.connect(src, cp, snk)
    tb.run()
    out = np.asarray(snk.data(), dtype=np.complex64)
    return out.reshape(n_sym, sym_len)


def pmt_intern_packet_len():
    import pmt

    return pmt.intern("packet_len")


def pmt_from_long(v: int):
    import pmt

    return pmt.from_long(int(v))


def pmt_intern_src():
    import pmt

    return pmt.intern("bench")


def ifft_cp_numpy(freq_shifted: np.ndarray, cp_len: int) -> np.ndarray:
    return numpy_add_cp(numpy_ifft_batch(freq_shifted), cp_len)


def ifft_cp_torch(freq_shifted, cp_len: int, device: str):
    td = torch_ifft_batch(freq_shifted, device)
    return add_cp_torch(td, cp_len, device)


def ifft_cp_gr(freq_shifted: np.ndarray, cp_len: int) -> np.ndarray:
    from gnuradio import blocks, digital, fft, gr

    n_sym, fft_len = freq_shifted.shape
    sym_len = fft_len + cp_len
    flat = freq_shifted.astype(np.complex64).reshape(-1)
    tags = [
        gr.tag_utils.python_to_tag(
            (i, pmt_intern_packet_len(), pmt_from_long(1), pmt_intern_src())
        )
        for i in range(n_sym)
    ]
    src = blocks.vector_source_c(flat.tolist(), False, fft_len, tags)
    iff = fft.fft_vcc(fft_len, False, (), True, 1)
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "packet_len")
    cp.set_min_output_buffer(max(sym_len * 4, 8192))
    snk = blocks.vector_sink_c()
    tb = gr.top_block()
    tb.connect(src, iff, cp, snk)
    tb.run()
    return np.asarray(snk.data(), dtype=np.complex64).reshape(n_sym, sym_len)


def full_chain_numpy(fft_len: int, n_sym: int, cp_len: int, seed: int) -> np.ndarray:
    """完整预计算链路: QPSK 填栅格(含 fftshift) → IFFT+CP。"""
    return ifft_cp_numpy(build_freq_grid_numpy(fft_len, n_sym, seed), cp_len)


def full_chain_torch(fft_len: int, n_sym: int, cp_len: int, seed: int, device: str):
    return ifft_cp_torch(build_freq_grid_torch(fft_len, n_sym, seed, device), cp_len, device)


def full_chain_gr(fft_len: int, n_sym: int, cp_len: int, seed: int) -> np.ndarray:
    """单次 top_block: symbols → allocator → fft_vcc → cyclic_prefixer。"""
    from gnuradio import blocks, digital, fft, gr

    occ = occupied_carrier_indices(fft_len)
    occupied = (tuple(occ),)
    pilots: tuple = ((),)
    pilot_sym: tuple = ((),)
    sync: tuple = ()
    n_data = len(occ)
    sym_len = fft_len + cp_len
    syms = qpsk_symbols_from_seed(n_sym, n_data, seed).reshape(-1)

    src = blocks.vector_source_c(syms.tolist(), False, 1)
    tagger = blocks.stream_to_tagged_stream(
        gr.sizeof_gr_complex, 1, n_data, "packet_len"
    )
    alloc = digital.ofdm_carrier_allocator_cvc(
        fft_len, occupied, pilots, pilot_sym, sync, "packet_len", True
    )
    iff = fft.fft_vcc(fft_len, False, (), True, 1)
    # empty len_tag_key: one vector in → one OFDM symbol with CP out
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "")
    cp.set_min_output_buffer(max(sym_len * 4, 8192))
    snk = blocks.vector_sink_c()
    tb = gr.top_block()
    tb.connect(src, tagger, alloc, iff, cp, snk)
    tb.run()
    out = np.asarray(snk.data(), dtype=np.complex64)
    return out.reshape(n_sym, sym_len)


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_matrix(
    *,
    fft_len: int,
    n_sym: int,
    cp_len: int,
    seed: int,
    warmup: int,
    repeat: int,
) -> list[BenchResult]:
    import torch

    has_cuda = torch.cuda.is_available()
    results: list[BenchResult] = []

    def add(
        step: str,
        backend: str,
        times: list[float] | None,
        err: float | None,
        note: str = "",
    ) -> None:
        if times is None:
            results.append(
                BenchResult(step, backend, float("nan"), float("nan"), float("nan"), err, note)
            )
            return
        med, mean, mn = _stats(times)
        results.append(BenchResult(step, backend, med, mean, mn, err, note))

    # ----- build_freq_grid -----
    # Reference: numpy
    t, ref_grid = bench_call(
        lambda: build_freq_grid_numpy(fft_len, n_sym, seed), warmup=warmup, repeat=repeat
    )
    ref_grid_np = to_numpy(ref_grid)
    add("build_freq_grid", "numpy", t, 0.0)

    t, out = bench_call(
        lambda: build_freq_grid_torch(fft_len, n_sym, seed, "cpu"),
        warmup=warmup,
        repeat=repeat,
    )
    add("build_freq_grid", "torch_cpu", t, _max_err(ref_grid_np, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: build_freq_grid_torch(fft_len, n_sym, seed, "cuda"),
            warmup=warmup,
            repeat=repeat,
            cuda=True,
        )
        add("build_freq_grid", "torch_cuda", t, _max_err(ref_grid_np, to_numpy(out)))
    else:
        add("build_freq_grid", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(
        lambda: build_freq_grid_gr(fft_len, n_sym, seed), warmup=warmup, repeat=repeat
    )
    add(
        "build_freq_grid",
        "gr_cpp",
        t,
        _max_err(ref_grid_np, to_numpy(out)),
        "incl. top_block sched",
    )

    # Unshifted grids for fftshift step (same symbols)
    occ = occupied_carrier_indices(fft_len)
    syms = qpsk_symbols_from_seed(n_sym, len(occ), seed)
    unshifted = map_to_grid_unshifted(syms, fft_len, occ)
    ref_shifted = fftshift_numpy(unshifted)

    # ----- fftshift -----
    t, out = bench_call(lambda: fftshift_numpy(unshifted), warmup=warmup, repeat=repeat)
    add("fftshift", "numpy", t, _max_err(ref_shifted, to_numpy(out)))

    t, out = bench_call(
        lambda: fftshift_torch(unshifted, "cpu"), warmup=warmup, repeat=repeat
    )
    add("fftshift", "torch_cpu", t, _max_err(ref_shifted, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: fftshift_torch(unshifted, "cuda"),
            warmup=warmup,
            repeat=repeat,
            cuda=True,
        )
        add("fftshift", "torch_cuda", t, _max_err(ref_shifted, to_numpy(out)))
    else:
        add("fftshift", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(lambda: fftshift_gr(unshifted), warmup=warmup, repeat=repeat)
    add("fftshift", "gr_cpp", t, _max_err(ref_shifted, to_numpy(out)), "incl. top_block sched")

    # ----- ifft -----
    freq = ref_shifted
    ref_td = ifft_numpy(freq)

    t, out = bench_call(lambda: ifft_numpy(freq), warmup=warmup, repeat=repeat)
    add("ifft", "numpy", t, _max_err(ref_td, to_numpy(out)))

    t, out = bench_call(lambda: ifft_torch(freq, "cpu"), warmup=warmup, repeat=repeat)
    add("ifft", "torch_cpu", t, _max_err(ref_td, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: ifft_torch(freq, "cuda"), warmup=warmup, repeat=repeat, cuda=True
        )
        add("ifft", "torch_cuda", t, _max_err(ref_td, to_numpy(out)))
    else:
        add("ifft", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(lambda: ifft_gr(freq), warmup=warmup, repeat=repeat)
    add("ifft", "gr_cpp", t, _max_err(ref_td, to_numpy(out)), "incl. top_block sched")

    # ----- add_cp -----
    ref_cp = add_cp_numpy(ref_td, cp_len)

    t, out = bench_call(lambda: add_cp_numpy(ref_td, cp_len), warmup=warmup, repeat=repeat)
    add("add_cp", "numpy", t, _max_err(ref_cp, to_numpy(out)))

    t, out = bench_call(
        lambda: add_cp_torch(ref_td, cp_len, "cpu"), warmup=warmup, repeat=repeat
    )
    add("add_cp", "torch_cpu", t, _max_err(ref_cp, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: add_cp_torch(ref_td, cp_len, "cuda"),
            warmup=warmup,
            repeat=repeat,
            cuda=True,
        )
        add("add_cp", "torch_cuda", t, _max_err(ref_cp, to_numpy(out)))
    else:
        add("add_cp", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(lambda: add_cp_gr(ref_td, cp_len), warmup=warmup, repeat=repeat)
    add("add_cp", "gr_cpp", t, _max_err(ref_cp, to_numpy(out)), "incl. top_block sched")

    # ----- ifft_cp -----
    ref_ic = ifft_cp_numpy(freq, cp_len)

    t, out = bench_call(
        lambda: ifft_cp_numpy(freq, cp_len), warmup=warmup, repeat=repeat
    )
    add("ifft_cp", "numpy", t, _max_err(ref_ic, to_numpy(out)))

    t, out = bench_call(
        lambda: ifft_cp_torch(freq, cp_len, "cpu"), warmup=warmup, repeat=repeat
    )
    add("ifft_cp", "torch_cpu", t, _max_err(ref_ic, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: ifft_cp_torch(freq, cp_len, "cuda"),
            warmup=warmup,
            repeat=repeat,
            cuda=True,
        )
        add("ifft_cp", "torch_cuda", t, _max_err(ref_ic, to_numpy(out)))
    else:
        add("ifft_cp", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(
        lambda: ifft_cp_gr(freq, cp_len), warmup=warmup, repeat=repeat
    )
    add("ifft_cp", "gr_cpp", t, _max_err(ref_ic, to_numpy(out)), "incl. top_block sched")

    # ----- full_chain: build_freq_grid + ifft_cp（完整预计算发射计算链路）-----
    ref_full = full_chain_numpy(fft_len, n_sym, cp_len, seed)

    t, out = bench_call(
        lambda: full_chain_numpy(fft_len, n_sym, cp_len, seed),
        warmup=warmup,
        repeat=repeat,
    )
    add("full_chain", "numpy", t, _max_err(ref_full, to_numpy(out)))

    t, out = bench_call(
        lambda: full_chain_torch(fft_len, n_sym, cp_len, seed, "cpu"),
        warmup=warmup,
        repeat=repeat,
    )
    add("full_chain", "torch_cpu", t, _max_err(ref_full, to_numpy(out)))

    if has_cuda:
        t, out = bench_call(
            lambda: full_chain_torch(fft_len, n_sym, cp_len, seed, "cuda"),
            warmup=warmup,
            repeat=repeat,
            cuda=True,
        )
        add("full_chain", "torch_cuda", t, _max_err(ref_full, to_numpy(out)))
    else:
        add("full_chain", "torch_cuda", None, None, "skipped: no CUDA")

    t, out = bench_call(
        lambda: full_chain_gr(fft_len, n_sym, cp_len, seed),
        warmup=warmup,
        repeat=repeat,
    )
    add(
        "full_chain",
        "gr_cpp",
        t,
        _max_err(ref_full, to_numpy(out)),
        "incl. top_block sched; one TB alloc→ifft→cp",
    )

    return results


def print_table(results: list[BenchResult]) -> None:
    hdr = f"{'step':<18} {'backend':<12} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} {'max_err':>12}  note"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.note.startswith("skipped"):
            print(
                f"{r.step:<18} {r.backend:<12} {'—':>10} {'—':>10} {'—':>10} {'—':>12}  {r.note}"
            )
            continue
        err = "n/a" if r.max_err is None or (isinstance(r.max_err, float) and np.isnan(r.max_err)) else f"{r.max_err:.3e}"
        print(
            f"{r.step:<18} {r.backend:<12} {r.median_ms:10.3f} {r.mean_ms:10.3f} {r.min_ms:10.3f} {err:>12}  {r.note}"
        )


def argument_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TX 预计算步骤 × 四后端时延基准")
    p.add_argument(
        "--mode",
        choices=("precompute", "realtime"),
        default="precompute",
        help="precompute=一次性墙钟；realtime=常驻算子稳态吞吐（见 bench_tx_realtime_backends）",
    )
    p.add_argument("--fft-len", type=int, default=2048)
    p.add_argument("--num-symbols", type=int, default=4)
    p.add_argument("--cp-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=15)
    # realtime-only (ignored in precompute)
    p.add_argument("--samp-rate", type=float, default=122.88e6)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--warmup-cpi", type=int, default=50)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--scenario",
        choices=("all", "cache_replay", "ifft_cp", "full_chain"),
        default="all",
    )
    return p.parse_args()


def main() -> None:
    args = argument_parser()
    if args.cp_len <= 0 or args.fft_len % 2 != 0:
        raise SystemExit("require fft_len even and cp_len > 0")

    if args.mode == "realtime":
        from bench_tx_realtime_backends import main as realtime_main

        realtime_main(
            [
                "--fft-len",
                str(args.fft_len),
                "--num-symbols",
                str(args.num_symbols),
                "--cp-len",
                str(args.cp_len),
                "--seed",
                str(args.seed),
                "--samp-rate",
                str(args.samp_rate),
                "--duration",
                str(args.duration),
                "--warmup-cpi",
                str(args.warmup_cpi),
                "--threads",
                str(args.threads),
                "--scenario",
                str(args.scenario),
            ]
        )
        return

    print(
        f"mode=precompute (one-shot wall; NOT comparable to realtime)\n"
        f"fft_len={args.fft_len} num_symbols={args.num_symbols} "
        f"cp_len={args.cp_len} seed={args.seed} warmup={args.warmup} repeat={args.repeat}"
    )
    import torch

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    results = run_matrix(
        fft_len=args.fft_len,
        n_sym=args.num_symbols,
        cp_len=args.cp_len,
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print_table(results)


if __name__ == "__main__":
    main()
