"""Load fixed FIR SIC taps from offline calibration (.npy) into a GRC top block."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np


def load_sic_taps(top_block: Any) -> None:
    """Load ``sic_taps.npy`` into ``top_block`` if path and tap count are valid."""
    path = getattr(top_block, "sic_taps_path", "") or ""
    if not path or not os.path.isfile(path):
        print(f"[SIC] taps file missing: {path!r}, using zeros", file=sys.stderr)
        return

    h = np.load(path)
    expected = int(getattr(top_block, "sic_num_taps", h.size))
    if h.size != expected:
        print(
            f"[SIC] tap count mismatch: file={h.size} expected={expected}",
            file=sys.stderr,
        )
        return

    top_block.set_sic_taps(h.astype(np.complex128).tolist())
    print(f"[SIC] loaded {h.size} taps from {path}")
