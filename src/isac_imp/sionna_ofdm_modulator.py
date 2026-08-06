"""Python OFDM 调制 epy 块：频域符号向量 → 时域+CP 标量流。

替换 GNU Radio ``fft_vxx`` (IFFT, shift=True) + ``digital_ofdm_cyclic_prefixer``。
IFFT 使用 ``1/sqrt(fft_len)`` 与 GR ``fft_vxx`` 一致；``start()`` 再校准 CPI 峰值 ≈ ``target_peak``。
"""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import TPP_DONT
from isac_imp.sionna_resource_grid_tx import _build_freq_grid


class SionnaOfdmModulatorBlock(gr.basic_block):
    """fftshift 频域 OFDM 符号 (vlen=fft_len) → 时域+CP 标量流。"""

    _FORECAST_MAX = 8191

    def __init__(
        self,
        fft_len: int = 2048,
        cp_len: int = 512,
        burst_len_samples: int = 10240,
        transpose_len: int = 4,
        subcarrier_spacing: float = 60e3,
        num_bits_per_symbol: int = 2,
        seed: int = 42,
        target_peak: float = 1.0,
        length_tag_key: str = "packet_len",
        **_ignored,
    ) -> None:
        del _ignored
        self._fft_len = int(fft_len)
        self._cp_len = int(cp_len)
        self._sym_len = self._fft_len + self._cp_len
        self._burst_len_samples = int(burst_len_samples)
        self._transpose_len = int(transpose_len)
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._seed = int(seed)
        self._target_peak = float(target_peak)
        self._ifft_scale = 1.0 / np.sqrt(float(self._fft_len))
        self._amp_scale = 1.0

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
        freq, _, _ = _build_freq_grid(
            fft_len=self._fft_len,
            transpose_len=self._transpose_len,
            subcarrier_spacing=self._subcarrier_spacing,
            cp_len=self._cp_len,
            num_bits_per_symbol=self._num_bits_per_symbol,
            device="cpu",
            seed=self._seed,
        )
        cpi = np.concatenate(
            [self._modulate_symbol_raw(freq[i]) for i in range(self._transpose_len)]
        )
        peak = float(np.max(np.abs(cpi)))
        self._amp_scale = self._target_peak / peak if peak > 0 else 1.0
        self._sym_buf = None
        self._sym_pos = 0
        self._cpi_tag_pending = False
        # #region agent log
        from isac_imp.agent_debug_log import agent_log

        agent_log(
            "sionna_ofdm_modulator.py:start",
            "amplitude calibration",
            {
                "raw_peak": peak,
                "amp_scale": self._amp_scale,
                "ifft_scale": self._ifft_scale,
                "cal_peak": float(np.max(np.abs(cpi * self._amp_scale))),
            },
            hypothesis_id="H5",
            run_id="post-fix-v3",
        )
        # #endregion
        return True

    def _tag_to_int(self, value: pmt.pmt) -> int:
        try:
            return int(pmt.to_long(value))
        except Exception:
            return int(pmt.to_python(value))

    def _modulate_symbol_raw(self, freq_shifted: np.ndarray) -> np.ndarray:
        td = np.fft.ifft(np.fft.ifftshift(freq_shifted), norm=None) * self._ifft_scale
        cp = td[-self._cp_len :]
        return np.concatenate((cp, td))

    def _modulate_symbol(self, freq_shifted: np.ndarray) -> np.ndarray:
        return (self._modulate_symbol_raw(freq_shifted) * self._amp_scale).astype(
            np.complex64, copy=False
        )

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
                if not hasattr(self, "_dbg_cpi_out"):
                    self._dbg_cpi_out = 0
                self._dbg_cpi_out += 1
                if self._dbg_cpi_out <= 3:
                    # #region agent log
                    from isac_imp.agent_debug_log import agent_log

                    agent_log(
                        "sionna_ofdm_modulator.py:general_work",
                        "CPI tag emitted",
                        {
                            "count": self._dbg_cpi_out,
                            "burst_len": self._cpi_out_len,
                            "sym_peak": float(np.max(np.abs(self._sym_buf))),
                        },
                        hypothesis_id="H5",
                        run_id="post-fix-v3",
                    )
                    # #endregion

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
