"""GNU Radio block: subtract offline SI template from OFDM Divide H(f) vectors."""

from __future__ import annotations

import os
import sys

import numpy as np
from gnuradio import gr

_COMPLEX_DTYPE = np.complex64


def _load_template_array(path: str, *, vlen: int) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        print(f"[SIC] template file missing: {path!r}, using zeros", file=sys.stderr)
        return None
    arr = np.load(path).reshape(-1)
    if arr.size != vlen:
        print(
            f"[SIC] template length mismatch: file={arr.size} expected={vlen}",
            file=sys.stderr,
        )
        return None
    return arr.astype(_COMPLEX_DTYPE, copy=False)


class SicDivideSubtract(gr.sync_block):
    """``out = in - template`` when enabled; 1:1 vector passthrough with tag propagation."""

    def __init__(
        self,
        vlen: int = 4096,
        sic_enable: bool = True,
        template_path: str = "",
    ) -> None:
        self._vlen = int(vlen)
        gr.sync_block.__init__(
            self,
            name="SicDivideSubtract",
            in_sig=[(np.complex64, self._vlen)],
            out_sig=[(np.complex64, self._vlen)],
        )
        self._sic_enable = bool(sic_enable)
        self._template_path = str(template_path)
        self._template = np.zeros(self._vlen, dtype=_COMPLEX_DTYPE)
        if self._template_path:
            self._reload_template()

    @property
    def sic_enable(self) -> bool:
        return self._sic_enable

    @sic_enable.setter
    def sic_enable(self, value: bool) -> None:
        self._sic_enable = bool(value)

    @property
    def template(self) -> np.ndarray:
        return self._template.copy()

    @property
    def template_path(self) -> str:
        return self._template_path

    @template_path.setter
    def template_path(self, value: str) -> None:
        self._template_path = str(value)
        self._reload_template()

    def set_template_path(self, value: str) -> None:
        self.template_path = value

    @template.setter
    def template(self, value) -> None:
        arr = np.asarray(value, dtype=np.complex128).reshape(-1)
        if arr.size != self._vlen:
            raise ValueError(f"template length {arr.size} != vlen {self._vlen}")
        self._template = arr.astype(_COMPLEX_DTYPE, copy=False)

    def set_template_list(self, value: list) -> None:
        self.template = value

    def _reload_template(self) -> None:
        loaded = _load_template_array(self._template_path, vlen=self._vlen)
        if loaded is not None:
            self._template = loaded
            print(
                f"[SIC] loaded template ({loaded.size},) from {self._template_path}",
                flush=True,
            )

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = min(len(inp), len(out))
        if self._sic_enable:
            for i in range(n):
                out[i][:] = inp[i] - self._template
        else:
            for i in range(n):
                out[i][:] = inp[i]
        return n
