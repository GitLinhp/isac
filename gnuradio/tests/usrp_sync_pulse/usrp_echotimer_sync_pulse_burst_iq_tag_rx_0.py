import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):

    def __init__(self, burst_len=3200, tag_key='packet_len', phase_offset=0):
        gr.sync_block.__init__(
            self,
            name='Burst IQ Tag RX',
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._tag_key = pmt.intern(tag_key)
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self._phase = 0
        self.set_burst_len(burst_len)
        self.set_phase_offset(phase_offset)

    def set_burst_len(self, burst_len):
        n = int(burst_len)
        if n <= 0:
            raise ValueError('burst_len must be positive')
        self._burst_len = n
        self._phase = int(self._phase) % self._burst_len

    def set_phase_offset(self, phase_offset):
        self._phase = int(phase_offset) % self._burst_len

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        out[:n] = inp
        abs_base = self.nitems_written(0)
        phase = self._phase
        blen = self._burst_len
        for i in range(n):
            if phase == 0:
                self.add_item_tag(
                    0, abs_base + i, self._tag_key, pmt.from_long(blen),
                )
            phase += 1
            if phase >= blen:
                phase = 0
        self._phase = phase
        return n
