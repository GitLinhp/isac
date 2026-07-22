#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Range Profile Collection
# Description: Load calibrated SIC FIR taps and record CPI complex range profiles with targets present.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import radar
import sip
import threading



class range_profile_collection(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Range Profile Collection", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Range Profile Collection")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "range_profile_collection")

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
        self.subcarrier_spacing = subcarrier_spacing = 120e3
        self.fft_len = fft_len = 2048
        self.samp_rate = samp_rate = int(fft_len * subcarrier_spacing)
        self.zeropadding_fac = zeropadding_fac = 2
        self.transpose_len = transpose_len = 1
        self.sic_num_taps = sic_num_taps = 64
        self.sic_enable = sic_enable = True
        self.record_enable = record_enable = False
        self.n_carriers = n_carriers = fft_len - 2
        self.R_max = R_max = 3e8/2/samp_rate*fft_len
        self.wait_to_start = wait_to_start = 0.03
        self.uhd_dev_args = uhd_dev_args = "type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=external,time_source=external"
        self.sic_taps_path = sic_taps_path = "/home/caict/Desktop/isac/gnuradio/tests/data_collection/sic_tap_calibration/dataset/run_001/sic_taps.npy"
        self.sic_taps = sic_taps = [0j]*int(sic_num_taps)
        self.sic_input_index = sic_input_index = 0 if sic_enable else 1
        self.record_output_index = record_output_index = 0 if record_enable else 1
        self.record_file_path = record_file_path = "/home/caict/Desktop/isac/gnuradio/tests/data_collection/range_profile_collection/dataset/run_001/range_profiles"
        self.range_bin_step = range_bin_step = R_max/(fft_len*zeropadding_fac)
        self.qpsk_symbols_per_packet = qpsk_symbols_per_packet = transpose_len * n_carriers
        self.payload_mod = payload_mod = digital.constellation_qpsk()
        self.packet_len = packet_len = transpose_len * n_carriers // 4
        self.occupied_carriers = occupied_carriers = list((list(range(-n_carriers//2, 0)) + list(range(1, n_carriers//2 + 1)),))
        self.num_delay_samp = num_delay_samp = 276
        self.min_out_buf_val = min_out_buf_val = int(2*transpose_len*(fft_len+fft_len/4))
        self.length_tag_key = length_tag_key = "packet_len"
        self.freq = freq = 6.0e9
        self.frame_rate_hz = frame_rate_hz = samp_rate / (transpose_len * (fft_len + fft_len // 4))
        self.factor = factor = 0.004
        self.TX_gain = TX_gain = 30
        self.RX_gain = RX_gain = 30

        ##################################################
        # Blocks
        ##################################################

        self._num_delay_samp_range = qtgui.Range(0, packet_len, 1, 276, 200)
        self._num_delay_samp_win = qtgui.RangeWidget(self._num_delay_samp_range, self.set_num_delay_samp, "Number of delayed samples", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._num_delay_samp_win)
        self._factor_range = qtgui.Range(0, 1, 0.001, 0.004, 200)
        self._factor_win = qtgui.RangeWidget(self._factor_range, self.set_factor, "'factor'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._factor_win)
        self._TX_gain_range = qtgui.Range(0, 50, 1, 30, 200)
        self._TX_gain_win = qtgui.RangeWidget(self._TX_gain_range, self.set_TX_gain, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._TX_gain_win)
        self._RX_gain_range = qtgui.Range(0, 50, 1, 30, 200)
        self._RX_gain_win = qtgui.RangeWidget(self._RX_gain_range, self.set_RX_gain, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._RX_gain_win)
        _sic_enable_check_box = Qt.QCheckBox("SIC Enable")
        self._sic_enable_choices = {True: True, False: False}
        self._sic_enable_choices_inv = dict((v,k) for k,v in self._sic_enable_choices.items())
        self._sic_enable_callback = lambda i: Qt.QMetaObject.invokeMethod(_sic_enable_check_box, "setChecked", Qt.Q_ARG("bool", self._sic_enable_choices_inv[i]))
        self._sic_enable_callback(self.sic_enable)
        _sic_enable_check_box.stateChanged.connect(lambda i: self.set_sic_enable(self._sic_enable_choices[bool(i)]))
        self.top_grid_layout.addWidget(_sic_enable_check_box, 1, 6, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(6, 7):
            self.top_grid_layout.setColumnStretch(c, 1)
        _record_enable_check_box = Qt.QCheckBox("Record Enable")
        self._record_enable_choices = {True: True, False: False}
        self._record_enable_choices_inv = dict((v,k) for k,v in self._record_enable_choices.items())
        self._record_enable_callback = lambda i: Qt.QMetaObject.invokeMethod(_record_enable_check_box, "setChecked", Qt.Q_ARG("bool", self._record_enable_choices_inv[i]))
        self._record_enable_callback(self.record_enable)
        _record_enable_check_box.stateChanged.connect(lambda i: self.set_record_enable(self._record_enable_choices[bool(i)]))
        self.top_grid_layout.addWidget(_record_enable_check_box, 1, 4, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(4, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.radar_usrp_echotimer_cc_0 = radar.usrp_echotimer_cc(int(samp_rate), freq, int(num_delay_samp), uhd_dev_args, 0, '', 'external', 'external', 'TX/RX', TX_gain, 0.2, wait_to_start, 0, uhd_dev_args, 0, '', 'external', 'external', 'RX1', RX_gain, 0.2, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0.set_min_output_buffer(min_out_buf_val)
        self.radar_ofdm_divide_vcvc_0 = radar.ofdm_divide_vcvc(fft_len, ((fft_len)*zeropadding_fac), (), 0, "packet_len")
        self.radar_ofdm_divide_vcvc_0.set_min_output_buffer((2*transpose_len))
        self.radar_ofdm_cyclic_prefix_remover_cvc_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0.set_min_output_buffer((2*transpose_len))
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            (fft_len*zeropadding_fac),
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
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            (fft_len + fft_len//4), #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            fft_len, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.filter_fir_filter_ccc_sic_0 = filter.fir_filter_ccc(1, sic_taps)
        self.filter_fir_filter_ccc_sic_0.declare_sample_delay(0)
        self.filter_fir_filter_ccc_sic_0.set_min_output_buffer(min_out_buf_val)
        self.fft_vxx_0_1 = fft.fft_vcc((fft_len*zeropadding_fac), True, window.blackmanharris(fft_len*zeropadding_fac), False, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0.set_min_output_buffer((2*transpose_len))
        self.fft_vxx_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fft_vxx_0.set_min_output_buffer((2*transpose_len))
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + fft_len//4,
            0,
            length_tag_key)
        self.digital_ofdm_cyclic_prefixer_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, ((),), ((),), (), length_tag_key, True)
        self.digital_ofdm_carrier_allocator_cvc_0.set_min_output_buffer((4*transpose_len))
        self.digital_chunks_to_symbols_xx_0_0 = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.digital_chunks_to_symbols_xx_0_0.set_min_output_buffer((2*qpsk_symbols_per_packet))
        self.blocks_sub_cc_sic_0 = blocks.sub_cc(1)
        self.blocks_sub_cc_sic_0.set_min_output_buffer(min_out_buf_val)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_stream_to_tagged_stream_0.set_min_output_buffer((2*qpsk_symbols_per_packet))
        self.blocks_selector_sic_0 = blocks.selector(gr.sizeof_gr_complex*1,sic_input_index,0)
        self.blocks_selector_sic_0.set_enabled(True)
        self.blocks_selector_sic_0.set_min_output_buffer(min_out_buf_val)
        self.blocks_selector_0 = blocks.selector(gr.sizeof_gr_complex*(fft_len*zeropadding_fac),0,record_output_index)
        self.blocks_selector_0.set_enabled(True)
        self.blocks_repack_bits_bb_0 = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0.set_min_output_buffer((2*qpsk_symbols_per_packet))
        self.blocks_null_sink_rec_0 = blocks.null_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac))
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, (fft_len*zeropadding_fac), 0)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(factor)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.blocks_integrate_xx_0_cx = blocks.integrate_cc(transpose_len, (fft_len*zeropadding_fac))
        self.blocks_integrate_xx_0 = blocks.integrate_ff(transpose_len, (fft_len*zeropadding_fac))
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac), record_file_path, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared((fft_len*zeropadding_fac))
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, (packet_len*8)))), True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0, 0))
        self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_integrate_xx_0_cx, 0), (self.blocks_selector_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.filter_fir_filter_ccc_sic_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.radar_usrp_echotimer_cc_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_repack_bits_bb_0, 0), (self.digital_chunks_to_symbols_xx_0_0, 0))
        self.connect((self.blocks_selector_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_selector_0, 1), (self.blocks_null_sink_rec_0, 0))
        self.connect((self.blocks_selector_sic_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_repack_bits_bb_0, 0))
        self.connect((self.blocks_sub_cc_sic_0, 0), (self.blocks_selector_sic_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.radar_ofdm_divide_vcvc_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.radar_ofdm_divide_vcvc_0, 1))
        self.connect((self.fft_vxx_0_1, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.fft_vxx_0_1, 0), (self.blocks_integrate_xx_0_cx, 0))
        self.connect((self.filter_fir_filter_ccc_sic_0, 0), (self.blocks_sub_cc_sic_0, 1))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.radar_ofdm_divide_vcvc_0, 0), (self.fft_vxx_0_1, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.blocks_selector_sic_0, 1))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.blocks_sub_cc_sic_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.qtgui_freq_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "range_profile_collection")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_subcarrier_spacing(self):
        return self.subcarrier_spacing

    def set_subcarrier_spacing(self, subcarrier_spacing):
        self.subcarrier_spacing = subcarrier_spacing
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_frame_rate_hz(self.samp_rate / (self.transpose_len * (self.fft_len + self.fft_len // 4)))
        self.set_min_out_buf_val(int(2*self.transpose_len*(self.fft_len+self.fft_len/4)))
        self.set_n_carriers(self.fft_len - 2)
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_frame_rate_hz(self.samp_rate / (self.transpose_len * (self.fft_len + self.fft_len // 4)))
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))
        self.fft_vxx_0_1.set_window(window.blackmanharris(self.fft_len*self.zeropadding_fac))

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.set_frame_rate_hz(self.samp_rate / (self.transpose_len * (self.fft_len + self.fft_len // 4)))
        self.set_min_out_buf_val(int(2*self.transpose_len*(self.fft_len+self.fft_len/4)))
        self.set_packet_len(self.transpose_len * self.n_carriers // 4)
        self.set_qpsk_symbols_per_packet(self.transpose_len * self.n_carriers)

    def get_sic_num_taps(self):
        return self.sic_num_taps

    def set_sic_num_taps(self, sic_num_taps):
        self.sic_num_taps = sic_num_taps
        self.set_sic_taps([0j]*int(self.sic_num_taps))

    def get_sic_enable(self):
        return self.sic_enable

    def set_sic_enable(self, sic_enable):
        self.sic_enable = sic_enable
        self._sic_enable_callback(self.sic_enable)
        self.set_sic_input_index(0 if self.sic_enable else 1)

    def get_record_enable(self):
        return self.record_enable

    def set_record_enable(self, record_enable):
        self.record_enable = record_enable
        self._record_enable_callback(self.record_enable)
        self.set_record_output_index(0 if self.record_enable else 1)

    def get_n_carriers(self):
        return self.n_carriers

    def set_n_carriers(self, n_carriers):
        self.n_carriers = n_carriers
        self.set_occupied_carriers(list((list(range(-self.n_carriers//2, 0)) + list(range(1, self.n_carriers//2 + 1)),)))
        self.set_packet_len(self.transpose_len * self.n_carriers // 4)
        self.set_qpsk_symbols_per_packet(self.transpose_len * self.n_carriers)

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))

    def get_wait_to_start(self):
        return self.wait_to_start

    def set_wait_to_start(self, wait_to_start):
        self.wait_to_start = wait_to_start

    def get_uhd_dev_args(self):
        return self.uhd_dev_args

    def set_uhd_dev_args(self, uhd_dev_args):
        self.uhd_dev_args = uhd_dev_args

    def get_sic_taps_path(self):
        return self.sic_taps_path

    def set_sic_taps_path(self, sic_taps_path):
        self.sic_taps_path = sic_taps_path

    def get_sic_taps(self):
        return self.sic_taps

    def set_sic_taps(self, sic_taps):
        self.sic_taps = sic_taps
        self.filter_fir_filter_ccc_sic_0.set_taps(self.sic_taps)

    def get_sic_input_index(self):
        return self.sic_input_index

    def set_sic_input_index(self, sic_input_index):
        self.sic_input_index = sic_input_index
        self.blocks_selector_sic_0.set_input_index(self.sic_input_index)

    def get_record_output_index(self):
        return self.record_output_index

    def set_record_output_index(self, record_output_index):
        self.record_output_index = record_output_index
        self.blocks_selector_0.set_output_index(self.record_output_index)

    def get_record_file_path(self):
        return self.record_file_path

    def set_record_file_path(self, record_file_path):
        self.record_file_path = record_file_path
        self.blocks_file_sink_0.open(self.record_file_path)

    def get_range_bin_step(self):
        return self.range_bin_step

    def set_range_bin_step(self, range_bin_step):
        self.range_bin_step = range_bin_step
        self.qtgui_vector_sink_f_0.set_x_axis(0, self.range_bin_step)

    def get_qpsk_symbols_per_packet(self):
        return self.qpsk_symbols_per_packet

    def set_qpsk_symbols_per_packet(self, qpsk_symbols_per_packet):
        self.qpsk_symbols_per_packet = qpsk_symbols_per_packet

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.blocks_stream_to_tagged_stream_0.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0.set_packet_len_pmt(self.packet_len)

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers

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

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_frame_rate_hz(self):
        return self.frame_rate_hz

    def set_frame_rate_hz(self, frame_rate_hz):
        self.frame_rate_hz = frame_rate_hz

    def get_factor(self):
        return self.factor

    def set_factor(self, factor):
        self.factor = factor
        self.blocks_multiply_const_vxx_0.set_k(self.factor)

    def get_TX_gain(self):
        return self.TX_gain

    def set_TX_gain(self, TX_gain):
        self.TX_gain = TX_gain
        self.radar_usrp_echotimer_cc_0.set_tx_gain(self.TX_gain)

    def get_RX_gain(self):
        return self.RX_gain

    def set_RX_gain(self, RX_gain):
        self.RX_gain = RX_gain
        self.radar_usrp_echotimer_cc_0.set_rx_gain(self.RX_gain)




def main(top_block_cls=range_profile_collection, options=None):

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
