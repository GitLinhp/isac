"""Load fixed Divide-domain SIC template from offline calibration (.npy) into a GRC top block."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from isac_imp.sic_divide_subtract import _load_template_array


def load_sic_template(top_block: Any) -> None:
    """Load ``sic_template.npy`` into ``top_block.sic_divide_subtract_0`` if valid."""
    path = getattr(top_block, "sic_template_path", "") or ""
    block = getattr(top_block, "sic_divide_subtract_0", None)
    if block is None:
        print("[SIC] sic_divide_subtract_0 block missing on top block", file=sys.stderr)
        return

    if hasattr(block, "set_template_path"):
        block.set_template_path(path)
        return

    expected = int(getattr(top_block, "sic_template_vlen", 0) or 0)
    t = _load_template_array(path, vlen=expected or np.load(path).size)
    if t is None:
        return
    block.set_template_list(t.astype(np.complex128).tolist())
    print(f"[SIC] loaded template ({t.size},) from {path}")
