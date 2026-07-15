"""
Embedded Python Block: Style 1 Sionna PHY TX (GR-aligned, UHD hardware burst)

Generate one OFDM frame at block construction, then reload that same waveform
before each UHD burst (tx_sob / tx_time / tx_eob). Requires USRP Sink
len_tag_name empty and sync=pc_clock with set_time_now().
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pmt
import sionna as sn
import torch
from gnuradio import gr

_UHD_TEST_DIR = "/home/caict/Desktop/gunradio_test/uhd_test"
if _UHD_TEST_DIR not in sys.path:
    sys.path.insert(0, _UHD_TEST_DIR)

from ofdm_loopback_phy_context import PhyContext
from isac_imp.burst_pack import TAG_TX_EOB, TAG_TX_SOB, TAG_TX_TIME


class blk(gr.basic_block):
    """Generate one frame at init; reload and replay on repeat_period_ms."""

    def __init__(
        self,
        device="cpu",
        tx_amp=0.05,
        repeat_period_ms=50.0,
        fft_len=2048,
        subcarrier_spacing=15e3,
        n_carriers=512,
        ofdm_syms_per_tag=32,
        time_lead_s=0.05,
    ):
        gr.basic_block.__init__(
            self,
            name="Style1 Sionna PHY TX",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._device = str(device)
        self._tx_amp = float(tx_amp)
        self._repeat_period_ms = float(repeat_period_ms)
        self._time_lead_s = float(time_lead_s)
        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_time = 0.0
        self._frame: np.ndarray | None = None
        self._burst_buffer: np.ndarray | None = None
        self._phy = PhyContext.from_grc(
            fft_len, subcarrier_spacing, n_carriers, ofdm_syms_per_tag
        )
        sn.phy.config.device = self._device
        torch.set_num_threads(1)
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self._initialize_frame()

    @property
    def phy(self) -> PhyContext:
        return self._phy

    @property
    def device(self):
        return self._device

    @property
    def tx_amp(self):
        return self._tx_amp

    @property
    def repeat_period_ms(self):
        return self._repeat_period_ms

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @property
    def fft_len(self) -> int:
        return self._phy.fft_len

    @property
    def subcarrier_spacing(self) -> float:
        return self._phy.subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, value):
        self.reconfigure_phy(subcarrier_spacing=float(value))

    @property
    def n_carriers(self) -> int:
        return self._phy.n_carriers

    @n_carriers.setter
    def n_carriers(self, value):
        self.reconfigure_phy(n_carriers=int(value))

    @property
    def ofdm_syms_per_tag(self) -> int:
        return self._phy.ofdm_syms_data

    @ofdm_syms_per_tag.setter
    def ofdm_syms_per_tag(self, value):
        self.reconfigure_phy(ofdm_syms_per_tag=int(value))

    @tx_amp.setter
    def tx_amp(self, value):
        self._tx_amp = float(value)

    @repeat_period_ms.setter
    def repeat_period_ms(self, value):
        self._repeat_period_ms = float(value)

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

    @device.setter
    def device(self, device):
        self._device = str(device)
        sn.phy.config.device = self._device
        self._rebuild_frame()

    def reconfigure_phy(
        self,
        fft_len=None,
        subcarrier_spacing=None,
        n_carriers=None,
        ofdm_syms_per_tag=None,
    ) -> None:
        self._phy = PhyContext.from_grc(
            fft_len if fft_len is not None else self._phy.fft_len,
            subcarrier_spacing
            if subcarrier_spacing is not None
            else self._phy.subcarrier_spacing,
            n_carriers if n_carriers is not None else self._phy.n_carriers,
            ofdm_syms_per_tag
            if ofdm_syms_per_tag is not None
            else self._phy.ofdm_syms_data,
        )
        self._rebuild_frame()

    def forecast(self, noutput_items, ninput_items):
        del noutput_items, ninput_items
        return []

    def _idle_gap_seconds(self) -> float:
        assert self._frame is not None
        frame_s = self._frame.size / self._phy.samp_rate
        period_s = max(self._repeat_period_ms / 1000.0, frame_s)
        return max(0.0, period_s - frame_s)

    def _tx_time_pmt(self):
        t = time.time() + self._time_lead_s
        sec = int(t)
        frac = t - sec
        return pmt.make_tuple(pmt.from_uint64(sec), pmt.from_double(frac))

    def _reset_burst_schedule(self, *, immediate: bool) -> None:
        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_time = 0.0 if immediate else time.monotonic() + self._idle_gap_seconds()

    def _initialize_frame(self) -> None:
        """Generate random payload and OFDM waveform once at startup."""
        self._frame = self._build_frame_once()
        self._load_burst_buffer()
        self._reset_burst_schedule(immediate=True)
        self.set_min_output_buffer(int(self._frame.size) * 2)

    def _rebuild_frame(self) -> None:
        """Regenerate waveform only when PHY dimensions or device change."""
        self._initialize_frame()

    def _load_burst_buffer(self) -> None:
        """Load the cached frame before each burst (same data, scaled by tx_amp)."""
        assert self._frame is not None
        self._burst_buffer = (self._frame * self._tx_amp).astype(
            np.complex64, copy=True
        )

    def _build_frame_once(self) -> np.ndarray:
        p = self._phy
        with torch.inference_mode():
            binary_source = sn.phy.mapping.BinarySource(device=self._device)
            mapper = sn.phy.mapping.Mapper("qam", 2, device=self._device)
            modulator = sn.phy.ofdm.OFDMModulator(
                cyclic_prefix_length=p.cp_len, device=self._device
            )
            bits = binary_source([p.bits_per_tag])
            syms = mapper(bits).reshape(p.ofdm_syms_data, p.n_carriers)
            grid = p.build_tx_freq_grid_torch(syms, device=self._device)
            x = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            y = modulator(x)
            if self._device.startswith("cuda"):
                torch.cuda.synchronize()
            frame = y.squeeze()
            if self._device != "cpu":
                frame = frame.cpu()
            return frame.detach().numpy().astype(np.complex64, copy=False)

    def general_work(self, input_items, output_items):
        del input_items
        out = output_items[0]
        assert self._frame is not None and self._burst_buffer is not None

        if not self._burst_active:
            if time.monotonic() < self._next_burst_time:
                return 0
            self._load_burst_buffer()
            self._burst_active = True
            self._burst_idx = 0

        frame_len = self._frame.size
        n_remain = frame_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        out[:n] = self._burst_buffer[self._burst_idx : self._burst_idx + n]
        abs_out = self.nitems_written(0)

        if self._burst_idx == 0:
            self.add_item_tag(0, abs_out, TAG_TX_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, TAG_TX_TIME, self._tx_time_pmt())

        self._burst_idx += n
        if self._burst_idx >= frame_len:
            self.add_item_tag(0, abs_out + n - 1, TAG_TX_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_time = time.monotonic() + self._idle_gap_seconds()

        return n
