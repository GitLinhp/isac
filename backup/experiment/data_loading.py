from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_COMPLEX_DTYPE = np.complex64


def count_range_profile_frames(path: str | Path, *, vlen: int = 4096) -> int:
    """Return the number of CPI frames in a raw complex64 range_profiles file."""
    item_bytes = vlen * np.dtype(_COMPLEX_DTYPE).itemsize
    size = os.path.getsize(path)
    if size % item_bytes != 0:
        raise ValueError(
            f"file size {size} is not a multiple of frame size {item_bytes}; "
            "file may be legacy float/file_meta format or recording ended abnormally"
        )
    return size // item_bytes


def iter_range_profile_frames(
    path: str | Path,
    *,
    vlen: int = 4096,
) -> Iterator[np.ndarray]:
    """Yield one CPI coherently-integrated complex range-profile frame per CPI."""
    item_bytes = vlen * np.dtype(_COMPLEX_DTYPE).itemsize
    size = os.path.getsize(path)
    if size % item_bytes != 0:
        raise ValueError(
            f"file size {size} is not a multiple of frame size {item_bytes}; "
            "file may be legacy float/file_meta format or recording ended abnormally"
        )

    with open(path, "rb") as f:
        while True:
            chunk = f.read(item_bytes)
            if not chunk:
                break
            if len(chunk) < item_bytes:
                raise ValueError(
                    f"truncated final frame: got {len(chunk)} bytes, expected {item_bytes}"
                )
            yield np.frombuffer(chunk, dtype=_COMPLEX_DTYPE).copy()


def frame_power_db(profile: np.ndarray) -> np.ndarray:
    """Convert one complex CPI frame to power (dB), matching playback mag2+nlog10."""
    return 10.0 * np.log10(np.maximum(np.abs(profile) ** 2, 1e-30))


if __name__ == "__main__":
    path = "/home/caict/Desktop/isac/gnuradio/tests/data_collection/usrp_ofdm_echotimer_dd_data_collection_test/dataset/run_001/range_profiles"

    vlen = 4096
    frame_rate_hz = 12000.0

    n_frames = count_range_profile_frames(path, vlen=vlen)
    print(f"帧数: {n_frames}, 时长约: {n_frames / frame_rate_hz:.2f} s")

    for frame_idx, profile in enumerate(iter_range_profile_frames(path, vlen=vlen)):
        if frame_idx == 0:
            print(f"第 0 帧前 5 个 bin (complex): {profile[:5]}")
            print(f"第 0 帧前 5 个 bin (|z|^2 dB): {frame_power_db(profile)[:5]}")
        break
