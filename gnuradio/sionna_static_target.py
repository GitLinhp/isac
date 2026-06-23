"""Sionna Torch 点目标信道：替代 gr-radar static_target_simulator_cc。"""
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import pmt
import torch
from gnuradio import gr

_GRC = Path(__file__).resolve().parent
_REPO = _GRC.parent
_SRC = _REPO / "src"
for _p in (_GRC, str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from isac.channel import StaticTargetParams, STChannel


class SionnaStaticTarget(gr.basic_block):
    """Torch STChannel 的 GNU Radio 流块（packet_len tag 门控）。"""

    def __init__(
        self,
        range_m: float = 100.0,
        velocity_mps: float = 5.0,
        rcs: float = 1e25,
        azimuth_deg: float = 0.0,
        position_rx_m: float = 0.0,
        samp_rate: int = 30_720_000,
        center_freq: float = 6e9,
        self_coupling_db: float = -10.0,
        rndm_phaseshift: bool = True,
        self_coupling: bool = True,
        length_tag_key: str = "packet_len",
        device: str = "cuda:0",
        fft_len: int = 2048,
        ofdm_symbols: int = 512,
        cp_len: int = 512,
        burst_mode: bool = False,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna Static Target",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

        self._device = str(device)
        self._tag_key = pmt.intern(length_tag_key)
        gr_cp = max(int(cp_len), 1)
        self._default_packet_len = int(ofdm_symbols) * (int(fft_len) + gr_cp)
        self._packet_len = self._default_packet_len

        self._in_buf: List[np.complex64] = []
        self._out_buf: List[np.complex64] = []
        self._out_packet_sample_idx = 0
        self._burst_mode = bool(burst_mode)
        self._burst_armed = not self._burst_mode

        self._sim = STChannel(
            StaticTargetParams(
                range_m=range_m,
                velocity_mps=velocity_mps,
                rcs=rcs,
                azimuth_deg=azimuth_deg,
                position_rx_m=position_rx_m,
                samp_rate=int(samp_rate),
                center_freq=float(center_freq),
                self_coupling_db=float(self_coupling_db),
                rndm_phaseshift=bool(rndm_phaseshift),
                self_coupling=bool(self_coupling),
            ),
        )

        self.set_min_output_buffer(max(4096, int(ofdm_symbols)))

    def setup_targets(
        self,
        range_m: float,
        velocity_mps: float,
        rcs: float,
        azimuth_deg: float,
        position_rx_m: float,
        samp_rate: int,
        center_freq: float,
        self_coupling_db: float,
        rndm_phaseshift: bool,
        self_coupling: bool,
    ) -> None:
        """GRC 滑块回调：运行时更新目标参数。"""
        self._sim = STChannel(
            StaticTargetParams(
                range_m=range_m,
                velocity_mps=velocity_mps,
                rcs=rcs,
                azimuth_deg=azimuth_deg,
                position_rx_m=position_rx_m,
                samp_rate=int(samp_rate),
                center_freq=float(center_freq),
                self_coupling_db=float(self_coupling_db),
                rndm_phaseshift=bool(rndm_phaseshift),
                self_coupling=bool(self_coupling),
            ),
        )

    def forecast(self, noutput_items: int, ninputs) -> List[int]:
        del noutput_items, ninputs
        return [1]

    def _process_packet(self, tx: np.ndarray) -> np.ndarray:
        dev = self._device
        ctx = (
            torch.cuda.device(dev)
            if dev.startswith("cuda") and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            tx_t = torch.from_numpy(tx).to(device=dev, dtype=torch.complex64)
            rx_t = self._sim(tx_t)
        return np.asarray(rx_t.detach().cpu().numpy(), dtype=np.complex64).reshape(-1)

    def _flush_input_packet(self) -> None:
        if not self._in_buf:
            return
        tx = np.ascontiguousarray(self._in_buf, dtype=np.complex64)
        self._out_buf.extend(self._process_packet(tx).tolist())
        self._out_packet_sample_idx = 0
        self._in_buf.clear()

    def _handle_packet_len_tag(self) -> None:
        if self._in_buf:
            self._flush_input_packet()
        self._in_buf.clear()
        self._out_packet_sample_idx = 0

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)
        n_out = len(out)
        consumed = 0
        produced = 0

        while produced < n_out and self._out_buf:
            if self._out_packet_sample_idx == 0:
                self.add_item_tag(
                    0,
                    self.nitems_written(0) + produced,
                    self._tag_key,
                    pmt.from_long(self._packet_len),
                )
            out[produced] = self._out_buf.pop(0)
            self._out_packet_sample_idx += 1
            if self._out_packet_sample_idx >= self._packet_len:
                self._out_packet_sample_idx = 0
            produced += 1

        while consumed < n_in and produced < n_out:
            abs_idx = self.nitems_read(0) + consumed
            for tag in self.get_tags_in_window(0, abs_idx, 1):
                if tag.key == self._tag_key:
                    self._handle_packet_len_tag()
                    self._packet_len = int(pmt.to_long(tag.value))
                    if self._burst_mode:
                        self._burst_armed = True

            if self._burst_mode and not self._burst_armed:
                out[produced] = inp[consumed]
                consumed += 1
                produced += 1
                continue

            self._in_buf.append(inp[consumed])
            consumed += 1

            if len(self._in_buf) >= self._packet_len:
                self._flush_input_packet()
                if self._burst_mode:
                    self._burst_armed = False
                while produced < n_out and self._out_buf:
                    if self._out_packet_sample_idx == 0:
                        self.add_item_tag(
                            0,
                            self.nitems_written(0) + produced,
                            self._tag_key,
                            pmt.from_long(self._packet_len),
                        )
                    out[produced] = self._out_buf.pop(0)
                    self._out_packet_sample_idx += 1
                    if self._out_packet_sample_idx >= self._packet_len:
                        self._out_packet_sample_idx = 0
                    produced += 1

        while consumed < n_in:
            abs_idx = self.nitems_read(0) + consumed
            for tag in self.get_tags_in_window(0, abs_idx, 1):
                if tag.key == self._tag_key:
                    self._handle_packet_len_tag()
                    self._packet_len = int(pmt.to_long(tag.value))
                    if self._burst_mode:
                        self._burst_armed = True

            if self._burst_mode and not self._burst_armed:
                consumed += 1
                continue

            self._in_buf.append(inp[consumed])
            consumed += 1

            if len(self._in_buf) >= self._packet_len:
                self._flush_input_packet()
                if self._burst_mode:
                    self._burst_armed = False

        if consumed:
            self.consume(0, consumed)
        if produced:
            self.produce(0, produced)
        if consumed or produced:
            return gr.WORK_CALLED_PRODUCE
        return gr.WORK_DONE
