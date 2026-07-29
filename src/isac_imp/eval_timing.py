"""评测循环计时：写出每样本平均耗时 JSON（支持算法核 / 设备同步）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch


TIMING_SCOPE_ALGO_CORE = "algo_core"
DEFAULT_WARMUP_FRAMES = 8


def timing_json_path_for_csv(csv_path: Path) -> Path:
    """``foo_rmse.csv`` → ``foo_rmse_timing.json``。"""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_timing.json")


def mean_ms_per_sample(eval_s: float, n_samples: int) -> float:
    if n_samples <= 0:
        return float("nan")
    return 1000.0 * float(eval_s) / float(n_samples)


def sync_device(device: torch.device | str | None) -> None:
    """CUDA 设备上 ``synchronize``；CPU / None 为空操作。"""
    if device is None:
        return
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.synchronize(device_t)


def write_eval_timing_json(
    path: Path,
    *,
    method: str,
    eval_s: float,
    n_samples: int,
    device: str | None = None,
    timing_scope: str = TIMING_SCOPE_ALGO_CORE,
) -> dict[str, float | int | str]:
    """写出 timing JSON 并返回 payload；同时打印一行摘要。"""
    payload: dict[str, float | int | str] = {
        "method": str(method),
        "eval_s": float(eval_s),
        "n_samples": int(n_samples),
        "mean_ms_per_sample": mean_ms_per_sample(eval_s, n_samples),
        "timing_scope": str(timing_scope),
        "device": str(device) if device is not None else "cpu",
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"timing: scope={payload['timing_scope']} device={payload['device']} "
        f"eval_s={payload['eval_s']:.4f} "
        f"n={payload['n_samples']} "
        f"mean_ms_per_sample={payload['mean_ms_per_sample']:.4f}",
        flush=True,
    )
    print(f"output timing json: {out}", flush=True)
    return payload


def read_eval_timing_json(path: Path) -> dict[str, float | int | str] | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "method": str(data.get("method", "")),
        "eval_s": float(data["eval_s"]),
        "n_samples": int(data["n_samples"]),
        "mean_ms_per_sample": float(data["mean_ms_per_sample"]),
        "timing_scope": str(data.get("timing_scope", "")),
        "device": str(data.get("device", "")),
    }


def run_algo_core_timed(
    n_total: int,
    *,
    device: torch.device | str | None,
    warmup: int = DEFAULT_WARMUP_FRAMES,
    run_one: Callable[[int], Any],
) -> tuple[float, int]:
    """对 ``0..n_total-1``：先 warmup，再对剩余帧计时（含 CUDA sync）。

    ``run_one(i)`` 执行单帧算法核。返回 ``(eval_s, n_timed)``。
    """
    if n_total <= 0:
        return 0.0, 0
    n_warm = min(int(warmup), n_total)
    for i in range(n_warm):
        run_one(i)
    sync_device(device)

    n_timed = n_total - n_warm
    if n_timed <= 0:
        sync_device(device)
        t0 = time.perf_counter()
        for i in range(n_total):
            run_one(i)
        sync_device(device)
        return time.perf_counter() - t0, n_total

    sync_device(device)
    t0 = time.perf_counter()
    for i in range(n_warm, n_total):
        run_one(i)
    sync_device(device)
    return time.perf_counter() - t0, n_timed
