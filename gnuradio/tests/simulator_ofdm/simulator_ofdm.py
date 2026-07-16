#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Simulator Ofdm
# Description: OFDM radar simulator with zero-Doppler range profile (qtgui_vector_sink_f).
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
import sip
import threading



class simulator_ofdm(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Simulator Ofdm", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Simulator Ofdm")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "simulator_ofdm")

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
        self.samp_rate = samp_rate = 5000000
        self.fft_len = fft_len = 2**6
        self.zeropadding_fac = zeropadding_fac = 2
        self.R_max = R_max = 3e8/2/samp_rate*fft_len
        self.range_bin_step = range_bin_step = R_max/(fft_len*zeropadding_fac)
        self.packet_len = packet_len = 2**9
        self.occupied_carriers_all = occupied_carriers_all = list((range(-26,27),))
        self.R_display = R_display = 100
        self.velocity = velocity = 0
        self.value_range = value_range = 100
        self.v_max = v_max = 2000
        self.transpose_len = transpose_len = int(np.ceil(packet_len*4.0/len(occupied_carriers_all[0])))
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = [(-21, -7, 7, 21,),]
        self.payload_mod = payload_mod = digital.constellation_qpsk()
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) +list( range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.length_tag_key = length_tag_key = "packet_len"
        self.display_bins = display_bins = min(fft_len*zeropadding_fac, int((R_display + range_bin_step - 1) / range_bin_step))
        self.discarded_carriers = discarded_carriers = []
        self.center_freq = center_freq = 2.45e9

        ##################################################
        # Blocks
        ##################################################

        self._velocity_range = qtgui.Range(-v_max, v_max, 1, 0, 200)
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
        self.radar_static_target_simulator_cc_0 = radar.static_target_simulator_cc([value_range], [velocity], [1e25], [0], [0], samp_rate, center_freq, -10, True, True, "packet_len")
        self.radar_static_target_simulator_cc_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.radar_ofdm_divide_vcvc_0 = radar.ofdm_divide_vcvc(fft_len, ((fft_len-len(discarded_carriers))*zeropadding_fac), (), 0, "packet_len")
        self.radar_ofdm_divide_vcvc_0.set_min_output_buffer((2*transpose_len))
        self.radar_ofdm_cyclic_prefix_remover_cvc_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0.set_min_output_buffer((2*transpose_len))
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            display_bins,
            0,
            range_bin_step,
            "Range",
            "Power (dB)",
            "Range Profile",
            1, # Number of inputs
            None # parent
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0.set_y_axis((-40), 0)
        self.qtgui_vector_sink_f_0.enable_autoscale(True)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        self.qtgui_vector_sink_f_0.set_x_axis_units("m")
        self.qtgui_vector_sink_f_0.set_y_axis_units("dB")
        self.qtgui_vector_sink_f_0.set_ref_level(0)


        labels = ['Range Profile', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
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
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_float*1, (fft_len*zeropadding_fac))
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_float*1, display_bins)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, (fft_len*zeropadding_fac), 0)
        self.blocks_keep_m_in_n_0 = blocks.keep_m_in_n(gr.sizeof_float, display_bins, (fft_len*zeropadding_fac), 0)
        self.blocks_integrate_xx_0 = blocks.integrate_ff(transpose_len, (fft_len*zeropadding_fac))
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared((fft_len*zeropadding_fac))
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.1, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0, 0))
        self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_keep_m_in_n_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_vector_to_stream_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_vector_to_stream_0, 0), (self.blocks_keep_m_in_n_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.radar_ofdm_divide_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.radar_static_target_simulator_cc_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.radar_ofdm_divide_vcvc_0, 1))
        self.connect((self.fft_vxx_0_1, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.radar_ofdm_divide_vcvc_0, 0), (self.fft_vxx_0_1, 0))
        self.connect((self.radar_static_target_simulator_cc_0, 0), (self.blocks_add_xx_0, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "simulator_ofdm")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], self.samp_rate, self.center_freq, -10, True, True)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))
        self.set_display_bins(min(self.fft_len*self.zeropadding_fac, int((self.R_display + self.range_bin_step - 1) / self.range_bin_step)))
        self.blocks_keep_m_in_n_0.set_n((self.fft_len*self.zeropadding_fac))
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))
        self.set_display_bins(min(self.fft_len*self.zeropadding_fac, int((self.R_display + self.range_bin_step - 1) / self.range_bin_step)))
        self.blocks_keep_m_in_n_0.set_n((self.fft_len*self.zeropadding_fac))
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))

    def get_range_bin_step(self):
        return self.range_bin_step

    def set_range_bin_step(self, range_bin_step):
        self.range_bin_step = range_bin_step
        self.set_display_bins(min(self.fft_len*self.zeropadding_fac, int((self.R_display + self.range_bin_step - 1) / self.range_bin_step)))
        self.qtgui_vector_sink_f_0.set_x_axis(0, self.range_bin_step)

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

    def get_R_display(self):
        return self.R_display

    def set_R_display(self, R_display):
        self.R_display = R_display
        self.set_display_bins(min(self.fft_len*self.zeropadding_fac, int((self.R_display + self.range_bin_step - 1) / self.range_bin_step)))

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = velocity
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], self.samp_rate, self.center_freq, -10, True, True)

    def get_value_range(self):
        return self.value_range

    def set_value_range(self, value_range):
        self.value_range = value_range
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], self.samp_rate, self.center_freq, -10, True, True)

    def get_v_max(self):
        return self.v_max

    def set_v_max(self, v_max):
        self.v_max = v_max

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len

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

    def get_display_bins(self):
        return self.display_bins

    def set_display_bins(self, display_bins):
        self.display_bins = display_bins
        self.blocks_keep_m_in_n_0.set_m(self.display_bins)

    def get_discarded_carriers(self):
        return self.discarded_carriers

    def set_discarded_carriers(self, discarded_carriers):
        self.discarded_carriers = discarded_carriers

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.radar_static_target_simulator_cc_0.setup_targets([self.value_range], [self.velocity], [1e25], [0], [0], self.samp_rate, self.center_freq, -10, True, True)




def main(top_block_cls=simulator_ofdm, options=None):

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
