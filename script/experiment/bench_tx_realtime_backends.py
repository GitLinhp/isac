#!/usr/bin/env python3
"""公平比较实时发射算子稳态吞吐/时延（常驻算子）。

场景:
  A  cache_replay  — 预计算 CPI 缓冲重放（memcpy），对应现网 ofdm_burst_tx_source 运行期
  B  ifft_cp       — 每 CPI 做 IFFT+CP（实时调制）
  C  full_chain    — 每 CPI：QPSK 填栅格(fftshift) → IFFT+CP（完整链路）

后端:
  numpy / torch_cpu / torch_cuda_kernel / torch_cuda_xfer / gr_cpp
  （full_chain 展示 torch_cpu / torch_cuda_* / gr_cpp）

计时边界（full_chain）:
  - 不计：算子/图初始化、top_block start、缓冲分配、CUDA 冷启动（warmup）
  - 计入：稳态计算 + 环内实际发生的设备/类型转换（如 D2H）

说明:
  - 不测每次重建 top_block；GR 使用 repeat 源 + 常驻块 + probe_rate
  - torch_cuda_kernel: 数据常驻 GPU，环内无 H2D/D2H
  - torch_cuda_xfer: 输入常驻 GPU；每 CPI 仅 D2H 结果到主机（贴近喂 USRP）
  - 与 bench_tx_precompute_backends.py 的一次性墙钟不可横比

也可经预计算脚本入口::

    PYTHONPATH=src python script/experiment/bench_tx_precompute_backends.py --mode realtime

示例::

    PYTHONPATH=src python script/experiment/bench_tx_realtime_backends.py
    PYTHONPATH=src python script/experiment/bench_tx_realtime_backends.py --scenario full_chain
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bench_tx_precompute_backends import (  # noqa: E402
    build_freq_grid_numpy,
    numpy_add_cp,
    numpy_ifft_batch,
    occupied_carrier_indices,
    qpsk_symbols_from_seed,
)


def _configure_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


@dataclass
class RealtimeResult:
    scenario: str
    backend: str
    samples_per_s: float
    cpi_per_s: float
    vs_samp_rate: float
    realtime_ok: bool
    note: str = ""

    @property
    def ms_per_cpi(self) -> float:
        if self.cpi_per_s <= 0:
            return float("nan")
        return 1e3 / self.cpi_per_s


def _ifft_cp_numpy_into(freq: np.ndarray, cp_len: int, out: np.ndarray) -> None:
    """Write one CPI (n_sym, fft+cp) into preallocated out."""
    td = numpy_ifft_batch(freq)
    out[:, :cp_len] = td[:, -cp_len:]
    out[:, cp_len:] = td


def _ifft_cp_torch(freq_t, cp_len: int):
    import torch

    td = torch.fft.ifft(torch.fft.ifftshift(freq_t, dim=-1), dim=-1, norm="forward")
    return torch.cat([td[..., -cp_len:], td], dim=-1)


def _full_chain_torch_once(syms, occ_idx, grid, cp_len: int):
    """Fill grid from resident symbols → fftshift → IFFT+CP（无设备间拷贝）。"""
    import torch

    grid.zero_()
    grid[:, occ_idx] = syms
    freq = torch.fft.fftshift(grid, dim=-1)
    return _ifft_cp_torch(freq, cp_len)


def bench_numpy_replay(cache: np.ndarray, duration_s: float, warmup_cpi: int) -> tuple[float, float]:
    """Scenario A: memcpy replay. Returns (samples_per_s, cpi_per_s)."""
    burst = int(cache.size)
    out = np.empty_like(cache)
    for _ in range(warmup_cpi):
        np.copyto(out, cache)
    n_cpi = 0
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    while time.perf_counter() < t_end:
        np.copyto(out, cache)
        n_cpi += 1
    elapsed = time.perf_counter() - t0
    return (n_cpi * burst) / elapsed, n_cpi / elapsed


def bench_torch_replay(
    cache: np.ndarray, duration_s: float, warmup_cpi: int, device: str, xfer: bool
) -> tuple[float, float]:
    import torch

    burst = int(cache.size)
    if device.startswith("cuda") and not xfer:
        src = torch.as_tensor(cache, device=device)
        dst = torch.empty_like(src)
        for _ in range(warmup_cpi):
            dst.copy_(src)
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            dst.copy_(src)
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    src_h = np.ascontiguousarray(cache)
    if device.startswith("cuda") and xfer:
        # Cache resident on GPU; per-CPI D2H only
        src = torch.as_tensor(cache, device=device)
        dst_h = np.empty(cache.shape, dtype=np.complex64)
        for _ in range(warmup_cpi):
            dst_h[:] = src.detach().cpu().numpy()
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            dst_h[:] = src.detach().cpu().numpy()
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    src_t = torch.as_tensor(src_h, device="cpu")
    dst_t = torch.empty_like(src_t)
    for _ in range(warmup_cpi):
        dst_t.copy_(src_t)
    n_cpi = 0
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    while time.perf_counter() < t_end:
        dst_t.copy_(src_t)
        n_cpi += 1
    elapsed = time.perf_counter() - t0
    return (n_cpi * burst) / elapsed, n_cpi / elapsed


def bench_gr_replay(cache: np.ndarray, duration_s: float) -> tuple[float, float]:
    """Scenario A GR: resident repeating vector_source -> probe_rate."""
    from gnuradio import blocks, gr

    burst = int(cache.size)
    src = blocks.vector_source_c(cache.reshape(-1).tolist(), True, 1)
    probe = blocks.probe_rate(gr.sizeof_gr_complex, 0.001, 0.15)
    snk = blocks.null_sink(gr.sizeof_gr_complex)
    tb = gr.top_block()
    tb.connect(src, probe)
    tb.connect(src, snk)
    tb.start()
    time.sleep(max(0.3, duration_s * 0.25))
    time.sleep(duration_s)
    rate = float(probe.rate())
    tb.stop()
    tb.wait()
    cpi_s = rate / burst if burst > 0 else 0.0
    return rate, cpi_s


def bench_numpy_ifft_cp(
    freq: np.ndarray, cp_len: int, duration_s: float, warmup_cpi: int
) -> tuple[float, float]:
    n_sym, fft_len = freq.shape
    burst = n_sym * (fft_len + cp_len)
    out = np.empty((n_sym, fft_len + cp_len), dtype=np.complex64)
    for _ in range(warmup_cpi):
        _ifft_cp_numpy_into(freq, cp_len, out)
    n_cpi = 0
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    while time.perf_counter() < t_end:
        _ifft_cp_numpy_into(freq, cp_len, out)
        n_cpi += 1
    elapsed = time.perf_counter() - t0
    return (n_cpi * burst) / elapsed, n_cpi / elapsed


def bench_torch_ifft_cp(
    freq: np.ndarray,
    cp_len: int,
    duration_s: float,
    warmup_cpi: int,
    device: str,
    xfer: bool,
) -> tuple[float, float]:
    import torch

    n_sym, fft_len = freq.shape
    burst = n_sym * (fft_len + cp_len)

    if device.startswith("cuda") and not xfer:
        freq_t = torch.as_tensor(freq, device=device)
        for _ in range(warmup_cpi):
            _ = _ifft_cp_torch(freq_t, cp_len)
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            _ = _ifft_cp_torch(freq_t, cp_len)
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    if device.startswith("cuda") and xfer:
        # Freq resident on GPU; per-CPI kernel + D2H only
        freq_t = torch.as_tensor(freq, device=device)
        for _ in range(warmup_cpi):
            out = _ifft_cp_torch(freq_t, cp_len)
            _ = out.detach().cpu().numpy()
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            out = _ifft_cp_torch(freq_t, cp_len)
            _ = out.detach().cpu().numpy()
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    freq_t = torch.as_tensor(freq, device="cpu")
    for _ in range(warmup_cpi):
        _ = _ifft_cp_torch(freq_t, cp_len)
    n_cpi = 0
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    while time.perf_counter() < t_end:
        _ = _ifft_cp_torch(freq_t, cp_len)
        n_cpi += 1
    elapsed = time.perf_counter() - t0
    return (n_cpi * burst) / elapsed, n_cpi / elapsed


def bench_gr_ifft_cp(
    freq: np.ndarray, cp_len: int, duration_s: float
) -> tuple[float, float]:
    """Resident fft_vcc -> ofdm_cyclic_prefixer (untagged per-symbol) + probe_rate."""
    from gnuradio import blocks, digital, fft, gr

    n_sym, fft_len = freq.shape
    sym_len = fft_len + cp_len
    burst = n_sym * sym_len
    flat = freq.astype(np.complex64).reshape(-1)
    src = blocks.vector_source_c(flat.tolist(), True, fft_len)
    iff = fft.fft_vcc(fft_len, False, (), True, 1)
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "")
    cp.set_min_output_buffer(max(sym_len * 8, 8192))
    probe = blocks.probe_rate(gr.sizeof_gr_complex, 0.001, 0.15)
    snk = blocks.null_sink(gr.sizeof_gr_complex)
    tb = gr.top_block()
    tb.connect(src, iff, cp)
    tb.connect(cp, probe)
    tb.connect(cp, snk)
    tb.start()
    time.sleep(max(0.3, duration_s * 0.25))
    time.sleep(duration_s)
    rate = float(probe.rate())
    tb.stop()
    tb.wait()
    return rate, rate / burst if burst else 0.0


def bench_torch_full_chain(
    *,
    fft_len: int,
    n_sym: int,
    cp_len: int,
    seed: int,
    duration_s: float,
    warmup_cpi: int,
    device: str,
    xfer: bool,
) -> tuple[float, float]:
    """Resident tensors; init/warmup outside timing. xfer=True: D2H result only (no per-CPI H2D)."""
    import torch

    occ = occupied_carrier_indices(fft_len)
    syms_np = qpsk_symbols_from_seed(n_sym, len(occ), seed)
    burst = n_sym * (fft_len + cp_len)
    use_cuda = device.startswith("cuda")

    if use_cuda and not xfer:
        syms = torch.as_tensor(syms_np, device=device, dtype=torch.complex64)
        occ_idx = torch.tensor(occ, device=device, dtype=torch.long)
        grid = torch.zeros((n_sym, fft_len), dtype=torch.complex64, device=device)
        for _ in range(warmup_cpi):
            _ = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            _ = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    if use_cuda and xfer:
        # Inputs resident on GPU; per-CPI kernel + D2H only
        syms = torch.as_tensor(syms_np, device=device, dtype=torch.complex64)
        occ_idx = torch.tensor(occ, device=device, dtype=torch.long)
        grid = torch.zeros((n_sym, fft_len), dtype=torch.complex64, device=device)
        for _ in range(warmup_cpi):
            out = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
            _ = out.detach().cpu().numpy()
            torch.cuda.synchronize()
        n_cpi = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_end = t0 + duration_s
        while time.perf_counter() < t_end:
            out = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
            _ = out.detach().cpu().numpy()
            n_cpi += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (n_cpi * burst) / elapsed, n_cpi / elapsed

    syms = torch.as_tensor(syms_np, device="cpu", dtype=torch.complex64)
    occ_idx = torch.tensor(occ, device="cpu", dtype=torch.long)
    grid = torch.zeros((n_sym, fft_len), dtype=torch.complex64, device="cpu")
    for _ in range(warmup_cpi):
        _ = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
    n_cpi = 0
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    while time.perf_counter() < t_end:
        _ = _full_chain_torch_once(syms, occ_idx, grid, cp_len)
        n_cpi += 1
    elapsed = time.perf_counter() - t0
    return (n_cpi * burst) / elapsed, n_cpi / elapsed


def bench_gr_full_chain(
    *,
    fft_len: int,
    n_sym: int,
    cp_len: int,
    seed: int,
    duration_s: float,
) -> tuple[float, float]:
    """Resident alloc→ifft→cp；搭建/start/tolist 在计时外。"""
    from gnuradio import blocks, digital, fft, gr

    occ = occupied_carrier_indices(fft_len)
    occupied = (tuple(occ),)
    pilots: tuple = ((),)
    pilot_sym: tuple = ((),)
    sync: tuple = ()
    n_data = len(occ)
    sym_len = fft_len + cp_len
    burst = n_sym * sym_len
    syms = qpsk_symbols_from_seed(n_sym, n_data, seed).reshape(-1)

    src = blocks.vector_source_c(syms.tolist(), True, 1)
    tagger = blocks.stream_to_tagged_stream(
        gr.sizeof_gr_complex, 1, n_data, "packet_len"
    )
    alloc = digital.ofdm_carrier_allocator_cvc(
        fft_len, occupied, pilots, pilot_sym, sync, "packet_len", True
    )
    iff = fft.fft_vcc(fft_len, False, (), True, 1)
    cp = digital.ofdm_cyclic_prefixer(fft_len, fft_len + cp_len, 0, "")
    cp.set_min_output_buffer(max(sym_len * 8, 8192))
    probe = blocks.probe_rate(gr.sizeof_gr_complex, 0.001, 0.15)
    snk = blocks.null_sink(gr.sizeof_gr_complex)
    tb = gr.top_block()
    tb.connect(src, tagger, alloc, iff, cp)
    tb.connect(cp, probe)
    tb.connect(cp, snk)
    tb.start()
    time.sleep(max(0.3, duration_s * 0.25))
    time.sleep(duration_s)
    rate = float(probe.rate())
    tb.stop()
    tb.wait()
    return rate, rate / burst if burst else 0.0


def _row(
    scenario: str,
    backend: str,
    sps: float,
    cps: float,
    samp_rate: float,
    note: str = "",
) -> RealtimeResult:
    vs = sps / samp_rate if samp_rate > 0 else float("nan")
    return RealtimeResult(
        scenario=scenario,
        backend=backend,
        samples_per_s=sps,
        cpi_per_s=cps,
        vs_samp_rate=vs,
        realtime_ok=bool(sps >= samp_rate),
        note=note,
    )


def run_realtime(
    *,
    fft_len: int,
    n_sym: int,
    cp_len: int,
    seed: int,
    samp_rate: float,
    duration_s: float,
    warmup_cpi: int,
    threads: int,
    scenario: str = "all",
) -> list[RealtimeResult]:
    import torch

    _configure_threads(threads)
    has_cuda = torch.cuda.is_available()
    freq = build_freq_grid_numpy(fft_len, n_sym, seed)
    cache = numpy_add_cp(numpy_ifft_batch(freq), cp_len).reshape(-1)
    results: list[RealtimeResult] = []
    want = {scenario} if scenario != "all" else {"cache_replay", "ifft_cp", "full_chain"}

    if "cache_replay" in want:
        sps, cps = bench_numpy_replay(cache, duration_s, warmup_cpi)
        results.append(_row("cache_replay", "numpy", sps, cps, samp_rate))

        sps, cps = bench_torch_replay(cache, duration_s, warmup_cpi, "cpu", xfer=False)
        results.append(_row("cache_replay", "torch_cpu", sps, cps, samp_rate))

        if has_cuda:
            sps, cps = bench_torch_replay(cache, duration_s, warmup_cpi, "cuda", xfer=False)
            results.append(
                _row(
                    "cache_replay",
                    "torch_cuda_kernel",
                    sps,
                    cps,
                    samp_rate,
                    "device-resident copy",
                )
            )
            sps, cps = bench_torch_replay(cache, duration_s, warmup_cpi, "cuda", xfer=True)
            results.append(
                _row(
                    "cache_replay",
                    "torch_cuda_xfer",
                    sps,
                    cps,
                    samp_rate,
                    "D2H only each CPI",
                )
            )
        else:
            results.append(
                _row(
                    "cache_replay",
                    "torch_cuda_kernel",
                    0.0,
                    0.0,
                    samp_rate,
                    "skipped: no CUDA",
                )
            )
            results.append(
                _row(
                    "cache_replay",
                    "torch_cuda_xfer",
                    0.0,
                    0.0,
                    samp_rate,
                    "skipped: no CUDA",
                )
            )

        sps, cps = bench_gr_replay(cache, duration_s)
        results.append(
            _row(
                "cache_replay",
                "gr_cpp",
                sps,
                cps,
                samp_rate,
                "resident vector_source+probe_rate",
            )
        )

    if "ifft_cp" in want:
        sps, cps = bench_numpy_ifft_cp(freq, cp_len, duration_s, warmup_cpi)
        results.append(_row("ifft_cp", "numpy", sps, cps, samp_rate))

        sps, cps = bench_torch_ifft_cp(freq, cp_len, duration_s, warmup_cpi, "cpu", False)
        results.append(_row("ifft_cp", "torch_cpu", sps, cps, samp_rate))

        if has_cuda:
            sps, cps = bench_torch_ifft_cp(
                freq, cp_len, duration_s, warmup_cpi, "cuda", False
            )
            results.append(
                _row(
                    "ifft_cp",
                    "torch_cuda_kernel",
                    sps,
                    cps,
                    samp_rate,
                    "freq resident on GPU",
                )
            )
            sps, cps = bench_torch_ifft_cp(
                freq, cp_len, duration_s, warmup_cpi, "cuda", True
            )
            results.append(
                _row(
                    "ifft_cp",
                    "torch_cuda_xfer",
                    sps,
                    cps,
                    samp_rate,
                    "kernel + D2H only each CPI",
                )
            )
        else:
            results.append(
                _row(
                    "ifft_cp", "torch_cuda_kernel", 0.0, 0.0, samp_rate, "skipped: no CUDA"
                )
            )
            results.append(
                _row(
                    "ifft_cp", "torch_cuda_xfer", 0.0, 0.0, samp_rate, "skipped: no CUDA"
                )
            )

        sps, cps = bench_gr_ifft_cp(freq, cp_len, duration_s)
        results.append(
            _row(
                "ifft_cp",
                "gr_cpp",
                sps,
                cps,
                samp_rate,
                "resident fft_vcc+cp+probe_rate",
            )
        )

    if "full_chain" in want:
        sps, cps = bench_torch_full_chain(
            fft_len=fft_len,
            n_sym=n_sym,
            cp_len=cp_len,
            seed=seed,
            duration_s=duration_s,
            warmup_cpi=warmup_cpi,
            device="cpu",
            xfer=False,
        )
        results.append(
            _row(
                "full_chain",
                "torch_cpu",
                sps,
                cps,
                samp_rate,
                "init excluded; resident CPU tensors",
            )
        )

        if has_cuda:
            sps, cps = bench_torch_full_chain(
                fft_len=fft_len,
                n_sym=n_sym,
                cp_len=cp_len,
                seed=seed,
                duration_s=duration_s,
                warmup_cpi=warmup_cpi,
                device="cuda",
                xfer=False,
            )
            results.append(
                _row(
                    "full_chain",
                    "torch_cuda_kernel",
                    sps,
                    cps,
                    samp_rate,
                    "init excluded; no per-CPI H2D/D2H",
                )
            )
            sps, cps = bench_torch_full_chain(
                fft_len=fft_len,
                n_sym=n_sym,
                cp_len=cp_len,
                seed=seed,
                duration_s=duration_s,
                warmup_cpi=warmup_cpi,
                device="cuda",
                xfer=True,
            )
            results.append(
                _row(
                    "full_chain",
                    "torch_cuda_xfer",
                    sps,
                    cps,
                    samp_rate,
                    "init excluded; D2H only each CPI",
                )
            )
        else:
            results.append(
                _row(
                    "full_chain",
                    "torch_cuda_kernel",
                    0.0,
                    0.0,
                    samp_rate,
                    "skipped: no CUDA",
                )
            )
            results.append(
                _row(
                    "full_chain",
                    "torch_cuda_xfer",
                    0.0,
                    0.0,
                    samp_rate,
                    "skipped: no CUDA",
                )
            )

        sps, cps = bench_gr_full_chain(
            fft_len=fft_len,
            n_sym=n_sym,
            cp_len=cp_len,
            seed=seed,
            duration_s=duration_s,
        )
        results.append(
            _row(
                "full_chain",
                "gr_cpp",
                sps,
                cps,
                samp_rate,
                "init/start excluded; resident alloc→ifft→cp",
            )
        )

    return results


def print_table(results: list[RealtimeResult], samp_rate: float) -> None:
    print(
        f"\nTarget samp_rate={samp_rate:.3e}  "
        f"(realtime_ok iff samples/s >= samp_rate)\n"
        "NOTE: not comparable to precompute one-shot wall times.\n"
        "full_chain: operator init excluded; device/type conversion included.\n"
    )
    hdr = (
        f"{'scenario':<14} {'backend':<20} {'ms/CPI':>10} {'samples/s':>12} {'CPI/s':>10} "
        f"{'vs_rate':>8} {'ok':>4}  note"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.note.startswith("skipped"):
            print(
                f"{r.scenario:<14} {r.backend:<20} {'—':>10} {'—':>12} {'—':>10} "
                f"{'—':>8} {'—':>4}  {r.note}"
            )
            continue
        ok = "Y" if r.realtime_ok else "N"
        print(
            f"{r.scenario:<14} {r.backend:<20} {r.ms_per_cpi:10.4f} {r.samples_per_s:12.3e} "
            f"{r.cpi_per_s:10.1f} {r.vs_samp_rate:8.2f} {ok:>4}  {r.note}"
        )


def argument_parser(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时发射算子稳态吞吐基准（公平对比）")
    p.add_argument("--fft-len", type=int, default=2048)
    p.add_argument("--num-symbols", type=int, default=4)
    p.add_argument("--cp-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samp-rate", type=float, default=122.88e6)
    p.add_argument("--duration", type=float, default=2.0, help="稳态测量时长（秒）")
    p.add_argument("--warmup-cpi", type=int, default=50)
    p.add_argument("--threads", type=int, default=1, help="CPU 线程数（公平固定）")
    p.add_argument(
        "--scenario",
        choices=("all", "cache_replay", "ifft_cp", "full_chain"),
        default="all",
        help="只跑指定场景（默认全部）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = argument_parser(argv)
    if args.cp_len <= 0 or args.fft_len % 2 != 0:
        raise SystemExit("require fft_len even and cp_len > 0")

    import torch

    print(
        f"fft_len={args.fft_len} num_symbols={args.num_symbols} cp_len={args.cp_len} "
        f"seed={args.seed} duration={args.duration}s threads={args.threads} "
        f"scenario={args.scenario}"
    )
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")

    results = run_realtime(
        fft_len=args.fft_len,
        n_sym=args.num_symbols,
        cp_len=args.cp_len,
        seed=args.seed,
        samp_rate=args.samp_rate,
        duration_s=args.duration,
        warmup_cpi=args.warmup_cpi,
        threads=args.threads,
        scenario=args.scenario,
    )
    print_table(results, args.samp_rate)


if __name__ == "__main__":
    main()
