"""离线加载成对距离谱数据集（``shard_*.npz`` + ``meta.json``）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch optional
    torch = None
    Dataset = object  # type: ignore[misc, assignment]


def _shard_paths(session_dir: Path) -> list[Path]:
    shards = sorted(session_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz under {session_dir}")
    return shards


def load_meta(session_dir: str | Path) -> dict[str, Any]:
    """读取 ``meta.json``。"""
    path = Path(session_dir) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing meta.json: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_session(
    session_dir: str | Path,
    *,
    mmap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """加载完整会话，拼接所有分片。

    Returns
    -------
    profiles_dev0, profiles_dev1, range_m, meta
        dev0/dev1 shape ``(N, vlen)``；``range_m`` 为距离轴 (m)。
    """
    session_dir = Path(session_dir)
    meta = load_meta(session_dir)
    vlen = int(meta["vlen"])
    range_step = float(meta.get("range_bin_step", 0.0))

    dev0_parts: list[np.ndarray] = []
    dev1_parts: list[np.ndarray] = []
    for shard in _shard_paths(session_dir):
        if mmap:
            data = np.load(shard, mmap_mode="r")
        else:
            data = np.load(shard)
        dev0_parts.append(np.asarray(data["profiles_dev0"], dtype=np.float32))
        dev1_parts.append(np.asarray(data["profiles_dev1"], dtype=np.float32))

    profiles_dev0 = np.concatenate(dev0_parts, axis=0)
    profiles_dev1 = np.concatenate(dev1_parts, axis=0)
    if profiles_dev0.shape != profiles_dev1.shape:
        raise ValueError(
            f"dev0/dev1 shape mismatch: {profiles_dev0.shape} vs {profiles_dev1.shape}"
        )
    if profiles_dev0.shape[1] != vlen:
        raise ValueError(f"expected vlen={vlen}, got {profiles_dev0.shape[1]}")

    range_m = np.arange(vlen, dtype=np.float64) * range_step
    return profiles_dev0, profiles_dev1, range_m, meta


def summarize_session(session_dir: str | Path) -> None:
    """打印会话摘要（帧数、shape、标签）。"""
    profiles_dev0, profiles_dev1, range_m, meta = load_session(session_dir)
    print(f"session: {Path(session_dir).resolve()}")
    print(f"  label: {meta.get('label', '')!r}")
    print(f"  frames: {profiles_dev0.shape[0]}")
    print(f"  vlen: {profiles_dev0.shape[1]}")
    print(f"  range_m: [{range_m[0]:.3f}, {range_m[-1]:.3f}] m")
    print(f"  dev0 peak bin: {int(profiles_dev0[0].argmax())}")
    print(f"  dev1 peak bin: {int(profiles_dev1[0].argmax())}")


class PairedRangeDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """PyTorch Dataset：每样本 ``(dev0, dev1)`` float32 向量。"""

    def __init__(
        self,
        session_dir: str | Path,
        *,
        mmap: bool = False,
        return_range_axis: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for PairedRangeDataset")
        dev0, dev1, range_m, meta = load_session(session_dir, mmap=mmap)
        self.profiles_dev0 = torch.from_numpy(dev0)
        self.profiles_dev1 = torch.from_numpy(dev1)
        self.range_m = torch.from_numpy(range_m.astype(np.float32))
        self.label = str(meta.get("label", ""))
        self.meta = meta
        self.return_range_axis = return_range_axis

    def __len__(self) -> int:
        return int(self.profiles_dev0.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        dev0 = self.profiles_dev0[index]
        dev1 = self.profiles_dev1[index]
        if self.return_range_axis:
            return dev0, dev1, self.range_m, self.label
        return dev0, dev1, self.label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize paired range profile session")
    parser.add_argument("session_dir", help="directory with meta.json and shard_*.npz")
    args = parser.parse_args()
    summarize_session(args.session_dir)
