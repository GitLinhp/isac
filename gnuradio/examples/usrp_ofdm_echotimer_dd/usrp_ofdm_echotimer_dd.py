#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Usrp Ofdm Echotimer Dd
# Description: USRP Sink/Source OFDM radar DD (no echotimer, no Throttle). RX packet_len tags via burst_iq_tag_rx EPY.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
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
        self.samp_rate = samp_rate = 5e6
        self.packet_len = packet_len = 2**9
        self.occupied_carriers_all = occupied_carriers_all = list((range(-26,27),))
        self.fft_len = fft_len = 2**6
        self.center_freq = center_freq = 6.0e9
        self.zeropadding_fac = zeropadding_fac = 2
        self.velocity = velocity = 500
        self.value_range = value_range = 100
        self.v_max = v_max = 3e8/(4*center_freq*(fft_len+fft_len/4)/samp_rate)
        self.transpose_len = transpose_len = int(np.ceil(packet_len*4.0/len(occupied_carriers_all[0])))
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = [(-21, -7, 7, 21,),]
        self.payload_mod = payload_mod = digital.constellation_qpsk()
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) +list( range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.length_tag_key = length_tag_key = "packet_len"
        self.discarded_carriers = discarded_carriers = []
        self.R_max = R_max = 3e8/2/samp_rate*fft_len

        ##################################################
        # Blocks
        ##################################################

        self._velocity_range = qtgui.Range(-v_max, v_max, 1, 500, 200)
        self._velocity_win = qtgui.RangeWidget(self._velocity_range, self.set_velocity, "Velocity", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._velocity_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._value_range_range = qtgui.Range(0.1, R_max, 1, 100, 200)
        self._value_range_win = qtgui.RangeWidget(self._value_range_range, self.set_value_range, "range", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._value_range_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.radar_transpose_matrix_vcvc_0_0 = radar.transpose_matrix_vcvc(transpose_len, (fft_len*zeropadding_fac), "packet_len")
        self.radar_transpose_matrix_vcvc_0_0.set_min_output_buffer((2*transpose_len))
        self.radar_transpose_matrix_vcvc_0 = radar.transpose_matrix_vcvc((fft_len*zeropadding_fac), transpose_len, "packet_len")
        self.radar_transpose_matrix_vcvc_0.set_min_output_buffer((2*fft_len*zeropadding_fac))
        self.radar_static_target_simulator_cc_0 = radar.static_target_simulator_cc([value_range], [velocity], [1e25], [0], [0], int(samp_rate), center_freq, -10, True, True, "packet_len")
        self.radar_static_target_simulator_cc_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.radar_qtgui_spectrogram_plot_0 = radar.qtgui_spectrogram_plot((fft_len*zeropadding_fac), 500, 'value_range', 'Velocity', 'OFDM Radar', [0,R_max], [0,v_max], [-15,-12], True, "packet_len")
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
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, samp_rate,True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*(fft_len*zeropadding_fac))
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(1, (fft_len*zeropadding_fac), 0)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared((fft_len*zeropadding_fac))
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.1, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.radar_estimator_ofdm_0, 'Msg out'), (self.radar_print_results_0, 'Msg in'))
        self.msg_connect((self.radar_os_cfar_2d_vc_0, 'Msg out'), (self.radar_estimator_ofdm_0, 'Msg in'))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.radar_qtgui_spectrogram_plot_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.radar_ofdm_divide_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.radar_static_target_simulator_cc_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.radar_ofdm_divide_vcvc_0, 1))
        self.connect((self.fft_vxx_0_1, 0), (self.radar_transpose_matrix_vcvc_0, 0))
        self.connect((self.fft_vxx_0_1_0, 0), (self.radar_transpose_matrix_vcvc_0_0, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.radar_ofdm_divide_vcvc_0, 0), (self.fft_vxx_0_1, 0))
        self.connect((self.radar_static_target_simulator_cc_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0, 0), (self.fft_vxx_0_1_0, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.radar_transpose_matrix_vcvc_0_0, 0), (self.radar_os_cfar_2d_vc_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_echotimer_dd")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_v_max(3e8/(4*self.center_freq*(self.fft_len+self.fft_len/4)/self.samp_rate))
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], int(self.samp_rate), self.center_freq, -10, True, True)

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_transpose_len(int(np.ceil(self.packet_len*4.0/len(self.occupied_carriers_all[0]))))
        self.blocks_stream_to_tagged_stream_0.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0.set_packet_len_pmt(self.packet_len)

    def get_occupied_carriers_all(self):
        return self.occupied_carriers_all

    def set_occupied_carriers_all(self, occupied_carriers_all):
        self.occupied_carriers_all = occupied_carriers_all
        self.set_transpose_len(int(np.ceil(self.packet_len*4.0/len(self.occupied_carriers_all[0]))))

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_v_max(3e8/(4*self.center_freq*(self.fft_len+self.fft_len/4)/self.samp_rate))
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.set_v_max(3e8/(4*self.center_freq*(self.fft_len+self.fft_len/4)/self.samp_rate))
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], int(self.samp_rate), self.center_freq, -10, True, True)

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = velocity
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], int(self.samp_rate), self.center_freq, -10, True, True)

    def get_value_range(self):
        return self.value_range

    def set_value_range(self, value_range):
        self.value_range = value_range
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], int(self.samp_rate), self.center_freq, -10, True, True)

    def get_v_max(self):
        return self.v_max

    def set_v_max(self, v_max):
        self.v_max = v_max

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.fft_vxx_0_1_0.set_window(window.blackmanharris(self.transpose_len))

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key

    def get_discarded_carriers(self):
        return self.discarded_carriers

    def set_discarded_carriers(self, discarded_carriers):
        self.discarded_carriers = discarded_carriers

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
