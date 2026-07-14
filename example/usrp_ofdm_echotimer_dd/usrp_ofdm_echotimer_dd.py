#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Usrp Ofdm Echotimer Dd
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import radar
import numpy as np
import threading



class usrp_ofdm_echotimer_dd(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Usrp Ofdm Echotimer Dd", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Usrp Ofdm Echotimer Dd")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_echotimer_dd")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.packet_len = packet_len = 2**9
        self.occupied_carriers_all = occupied_carriers_all = list((range(-26,27),))
        self.wait_to_start = wait_to_start = 0.03
        self.transpose_len = transpose_len = int(np.ceil(packet_len*4.0/len(occupied_carriers_all[0])))
        self.samp_rate = samp_rate = 5120000
        self.fft_len = fft_len = 2**6
        self.burst_pri_s = burst_pri_s = wait_to_start + transpose_len*(fft_len+fft_len//4)/samp_rate
        self.zeropadding_fac = zeropadding_fac = 2
        self.v_max = v_max = 2000
        self.uhd_timeout_s = uhd_timeout_s = 1.0
        self.uhd_dev_args = uhd_dev_args = "type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2"
        self.tx_gain = tx_gain = 20
        self.spectrogram_interval = spectrogram_interval = max(5000, int(1000 * (wait_to_start + 2*transpose_len*(fft_len+fft_len//4)/samp_rate + 2.0)))
        self.rx_gain = rx_gain = 40
        self.payload_mod = payload_mod = digital.constellation_qpsk()
        self.num_delay_samp = num_delay_samp = 0
        self.min_out_buf_val = min_out_buf_val = int(2*transpose_len*(fft_len+fft_len/4))
        self.length_tag_key = length_tag_key = "packet_len"
        self.discarded_carriers = discarded_carriers = []
        self.center_freq = center_freq = 6000000000
        self.burst_byte_rate = burst_byte_rate = packet_len/burst_pri_s
        self.R_max = R_max = 3e8/2/samp_rate*fft_len

        ##################################################
        # Blocks
        ##################################################

        self._tx_gain_range = qtgui.Range(0, 100, 1, 20, 200)
        self._tx_gain_win = qtgui.RangeWidget(self._tx_gain_range, self.set_tx_gain, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tx_gain_win, 0, 4, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(4, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rx_gain_range = qtgui.Range(0, 100, 1, 40, 200)
        self._rx_gain_win = qtgui.RangeWidget(self._rx_gain_range, self.set_rx_gain, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rx_gain_win, 0, 3, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._num_delay_samp_range = qtgui.Range(0, packet_len, 1, 0, 200)
        self._num_delay_samp_win = qtgui.RangeWidget(self._num_delay_samp_range, self.set_num_delay_samp, "num_delay_samp", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._num_delay_samp_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.radar_usrp_echotimer_cc_0 = radar.usrp_echotimer_cc(samp_rate, center_freq, int(num_delay_samp), uhd_dev_args, 0, '', 'internal', 'internal', 'TX/RX', tx_gain, uhd_timeout_s, wait_to_start, 0, uhd_dev_args, 0, '', 'internal', 'internal', 'RX1', rx_gain, uhd_timeout_s, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0.set_min_output_buffer(min_out_buf_val)
        self.radar_transpose_matrix_vcvc_0_0 = radar.transpose_matrix_vcvc(transpose_len, (fft_len*zeropadding_fac), "packet_len")
        self.radar_transpose_matrix_vcvc_0_0.set_min_output_buffer((2*transpose_len))
        self.radar_transpose_matrix_vcvc_0 = radar.transpose_matrix_vcvc((fft_len*zeropadding_fac), transpose_len, "packet_len")
        self.radar_transpose_matrix_vcvc_0.set_min_output_buffer((2*fft_len*zeropadding_fac))
        self.radar_qtgui_spectrogram_plot_0 = radar.qtgui_spectrogram_plot((fft_len*zeropadding_fac), spectrogram_interval, 'value_range', 'Velocity', 'OFDM Radar', [0,R_max], [0,v_max], [-15,-12], True, "packet_len")
        self.radar_print_results_0 = radar.print_results(False, "")
        self.radar_os_cfar_2d_vc_0 = radar.os_cfar_2d_vc((fft_len*zeropadding_fac), [10,10], [0,0], 0.78, 30, "packet_len")
        self.radar_ofdm_divide_vcvc_0 = radar.ofdm_divide_vcvc(fft_len, ((fft_len-len(discarded_carriers))*zeropadding_fac), (), 0, "packet_len")
        self.radar_ofdm_divide_vcvc_0.set_min_output_buffer((2*transpose_len))
        self.radar_ofdm_cyclic_prefix_remover_cvc_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0.set_min_output_buffer((2*transpose_len))
        self.radar_estimator_ofdm_0 = radar.estimator_ofdm('range', (fft_len*zeropadding_fac), [0,R_max], 'velocity', transpose_len, [0,v_max,-v_max,0], True)
        self.fft_vxx_0_1_0 = fft.fft_vcc(transpose_len, False, window.blackmanharris(transpose_len), False, 1)
        self.fft_vxx_0_1 = fft.fft_vcc((fft_len*zeropadding_fac), True, window.blackmanharris(fft_len*zeropadding_fac), False, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0.set_min_output_buffer((2*transpose_len))
        self.fft_vxx_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + fft_len//4,
            0,
            length_tag_key)
        self.digital_ofdm_cyclic_prefixer_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers_all, ((),), ((),), (), length_tag_key, True)
        self.digital_ofdm_carrier_allocator_cvc_0.set_min_output_buffer((2*transpose_len))
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.digital_chunks_to_symbols_xx_0_0.set_min_output_buffer((2*packet_len*4))
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, burst_byte_rate,True)
        self.blocks_tag_debug_rx_0 = blocks.tag_debug(gr.sizeof_gr_complex*1, 'RX echotimer', "")
        self.blocks_tag_debug_rx_0.set_display(True)
        self.blocks_tag_debug_dd_0 = blocks.tag_debug(gr.sizeof_float*(fft_len*zeropadding_fac), 'DD output', "")
        self.blocks_tag_debug_dd_0.set_display(True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*(fft_len*zeropadding_fac))
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(1, (fft_len*zeropadding_fac), 0)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared((fft_len*zeropadding_fac))
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.radar_estimator_ofdm_0, 'Msg out'), (self.radar_print_results_0, 'Msg in'))
        self.msg_connect((self.radar_os_cfar_2d_vc_0, 'Msg out'), (self.radar_estimator_ofdm_0, 'Msg in'))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_tag_debug_dd_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.radar_qtgui_spectrogram_plot_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.radar_ofdm_divide_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.radar_usrp_echotimer_cc_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.radar_ofdm_divide_vcvc_0, 1))
        self.connect((self.fft_vxx_0_1, 0), (self.radar_transpose_matrix_vcvc_0, 0))
        self.connect((self.fft_vxx_0_1_0, 0), (self.radar_transpose_matrix_vcvc_0_0, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.radar_ofdm_divide_vcvc_0, 0), (self.fft_vxx_0_1, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0, 0), (self.fft_vxx_0_1_0, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0_0, 0), (self.radar_os_cfar_2d_vc_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.blocks_tag_debug_rx_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_echotimer_dd")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_transpose_len(int(np.ceil(self.packet_len*4.0/len(self.occupied_carriers_all[0]))))
        self.set_burst_byte_rate(self.packet_len/self.burst_pri_s)
        self.blocks_stream_to_tagged_stream_0.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0.set_packet_len_pmt(self.packet_len)

    def get_occupied_carriers_all(self):
        return self.occupied_carriers_all

    def set_occupied_carriers_all(self, occupied_carriers_all):
        self.occupied_carriers_all = occupied_carriers_all
        self.set_transpose_len(int(np.ceil(self.packet_len*4.0/len(self.occupied_carriers_all[0]))))

    def get_wait_to_start(self):
        return self.wait_to_start

    def set_wait_to_start(self, wait_to_start):
        self.wait_to_start = wait_to_start
        self.set_spectrogram_interval(max(5000, int(1000 * (self.wait_to_start + 2*self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate + 2.0))))
        self.set_burst_pri_s(self.wait_to_start + self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate)

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.set_min_out_buf_val(int(2*self.transpose_len*(self.fft_len+self.fft_len/4)))
        self.set_spectrogram_interval(max(5000, int(1000 * (self.wait_to_start + 2*self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate + 2.0))))
        self.set_burst_pri_s(self.wait_to_start + self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate)
        self.fft_vxx_0_1_0.set_window(window.blackmanharris(self.transpose_len))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_spectrogram_interval(max(5000, int(1000 * (self.wait_to_start + 2*self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate + 2.0))))
        self.set_burst_pri_s(self.wait_to_start + self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_min_out_buf_val(int(2*self.transpose_len*(self.fft_len+self.fft_len/4)))
        self.set_spectrogram_interval(max(5000, int(1000 * (self.wait_to_start + 2*self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate + 2.0))))
        self.set_burst_pri_s(self.wait_to_start + self.transpose_len*(self.fft_len+self.fft_len//4)/self.samp_rate)
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_burst_pri_s(self):
        return self.burst_pri_s

    def set_burst_pri_s(self, burst_pri_s):
        self.burst_pri_s = burst_pri_s
        self.set_burst_byte_rate(self.packet_len/self.burst_pri_s)

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_v_max(self):
        return self.v_max

    def set_v_max(self, v_max):
        self.v_max = v_max

    def get_uhd_timeout_s(self):
        return self.uhd_timeout_s

    def set_uhd_timeout_s(self, uhd_timeout_s):
        self.uhd_timeout_s = uhd_timeout_s

    def get_uhd_dev_args(self):
        return self.uhd_dev_args

    def set_uhd_dev_args(self, uhd_dev_args):
        self.uhd_dev_args = uhd_dev_args

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.radar_usrp_echotimer_cc_0.set_tx_gain(self.tx_gain)

    def get_spectrogram_interval(self):
        return self.spectrogram_interval

    def set_spectrogram_interval(self, spectrogram_interval):
        self.spectrogram_interval = spectrogram_interval

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.radar_usrp_echotimer_cc_0.set_rx_gain(self.rx_gain)

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_num_delay_samp(self):
        return self.num_delay_samp

    def set_num_delay_samp(self, num_delay_samp):
        self.num_delay_samp = num_delay_samp
        self.radar_usrp_echotimer_cc_0.set_num_delay_samps(int(self.num_delay_samp))

    def get_min_out_buf_val(self):
        return self.min_out_buf_val

    def set_min_out_buf_val(self, min_out_buf_val):
        self.min_out_buf_val = min_out_buf_val

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key

    def get_discarded_carriers(self):
        return self.discarded_carriers

    def set_discarded_carriers(self, discarded_carriers):
        self.discarded_carriers = discarded_carriers

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq

    def get_burst_byte_rate(self):
        return self.burst_byte_rate

    def set_burst_byte_rate(self, burst_byte_rate):
        self.burst_byte_rate = burst_byte_rate
        self.blocks_throttle_0.set_sample_rate(self.burst_byte_rate)

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max




def main(top_block_cls=usrp_ofdm_echotimer_dd, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
