"""Python OFDM 调制 epy 块：频域符号向量 → 时域+CP 标量流。

替换 GNU Radio ``fft_vxx`` (IFFT, shift=True) + ``digital_ofdm_cyclic_prefixer``。
IFFT 使用 ``norm="forward"``（无 1/N），与 GR ``fft_vcc`` / FFTW 一致。
"""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import TPP_DONT


class SionnaOfdmModulatorBlock(gr.basic_block):
    """fftshift 频域 OFDM 符号 (vlen=fft_len) → 时域+CP 标量流。"""

    _FORECAST_MAX = 8191

    def __init__(
        self,
        fft_len: int = 2048,
        cp_len: int = 512,
        burst_len_samples: int = 10240,
        length_tag_key: str = "packet_len",
        **_ignored,
    ) -> None:
        del _ignored
        self._fft_len = int(fft_len)
        self._cp_len = int(cp_len)
        self._sym_len = self._fft_len + self._cp_len
        self._burst_len_samples = int(burst_len_samples)

        gr.basic_block.__init__(
            self,
            name="Sionna OFDM Modulator",
            in_sig=[(np.complex64, self._fft_len)],
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._srcid = pmt.intern("sionna_ofdm_modulator")

        self._sym_buf: np.ndarray | None = None
        self._sym_pos = 0
        self._cpi_out_len = 0
        self._cpi_tag_pending = False

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._burst_len_samples)

    @property
    def burst_len_samples(self) -> int:
        return self._burst_len_samples

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len_samples = int(value)

    def start(self) -> bool:
        self._sym_buf = None
        self._sym_pos = 0
        self._cpi_tag_pending = False
        return True

    def _tag_to_int(self, value: pmt.pmt) -> int:
        try:
            return int(pmt.to_long(value))
        except Exception:
            return int(pmt.to_python(value))

    def _modulate_symbol(self, freq_shifted: np.ndarray) -> np.ndarray:
        td = np.fft.ifft(np.fft.ifftshift(freq_shifted), norm="forward")
        return np.concatenate((td[-self._cp_len :], td)).astype(np.complex64, copy=False)

    def forecast(self, noutput_items: int, ninputs: list) -> list:
        del ninputs
        if self._sym_buf is not None:
            need = self._sym_len - self._sym_pos
        else:
            need = max(1, (noutput_items + self._sym_len - 1) // self._sym_len)
        return [min(max(1, need), self._FORECAST_MAX)]

    def general_work(self, input_items, output_items) -> int:
        inp = input_items[0]
        out = output_items[0]
        if len(out) <= 0:
            return 0

        in_base = self.nitems_read(0)
        write_base = self.nitems_written(0)
        in_consumed = 0
        out_produced = 0

        while out_produced < len(out):
            if self._sym_buf is None or self._sym_pos >= self._sym_len:
                if in_consumed >= len(inp):
                    break
                abs_in = in_base + in_consumed
                for tag in self.get_tags_in_range(0, abs_in, abs_in + 1):
                    if pmt.eq(tag.key, self._length_tag_key):
                        n_syms = self._tag_to_int(tag.value)
                        if n_syms > 0:
                            self._cpi_out_len = n_syms * self._sym_len
                            self._cpi_tag_pending = True
                self._sym_buf = self._modulate_symbol(inp[in_consumed])
                self._sym_pos = 0
                in_consumed += 1

            if self._cpi_tag_pending and self._sym_pos == 0:
                self.add_item_tag(
                    0,
                    write_base + out_produced,
                    self._length_tag_key,
                    pmt.from_long(self._cpi_out_len),
                    self._srcid,
                )
                self._cpi_tag_pending = False

            n_copy = min(len(out) - out_produced, self._sym_len - self._sym_pos)
            out[out_produced : out_produced + n_copy] = self._sym_buf[
                self._sym_pos : self._sym_pos + n_copy
            ]
            self._sym_pos += n_copy
            out_produced += n_copy
            if self._sym_pos >= self._sym_len:
                self._sym_buf = None

        if in_consumed > 0:
            self.consume(0, in_consumed)
        if out_produced > 0:
            return out_produced
        if in_consumed > 0:
            return gr.WORK_CALLED_PRODUCE
        return 0


blk = SionnaOfdmModulatorBlock
