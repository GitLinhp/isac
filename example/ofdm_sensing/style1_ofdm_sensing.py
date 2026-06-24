#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Style1 Burst OFDM Sensing (X410)
# Author: Lin Haopeng
# Copyright: Caict
# Description: Style1 burst OFDM TX (Sionna) + GR OFDM RX sensing (X410 OTA)
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import sip
import style1_ofdm_sensing_epy_fft_batch_hdr as epy_fft_batch_hdr  # embedded python block
import style1_ofdm_sensing_epy_fft_batch_rx as epy_fft_batch_rx  # embedded python block
import style1_ofdm_sensing_epy_sionna_phy_tx as epy_sionna_phy_tx  # embedded python block
import threading



class style1_ofdm_sensing(gr.top_block, Qt.QWidget):

    def __init__(self, address="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2", freq=6.0e9):
        gr.top_block.__init__(self, "Style1 Burst OFDM Sensing (X410)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Style1 Burst OFDM Sensing (X410)")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "style1_ofdm_sensing")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.address = address
        self.freq = freq

        ##################################################
        # Variables
        ##################################################
        self.ofdm_syms_per_tag = ofdm_syms_per_tag = 32
        self.n_carriers = n_carriers = 512
        self.fft_len = fft_len = 2048
        self.subcarrier_spacing = subcarrier_spacing = 15e3
        self.pilot_symbols = pilot_symbols = ((-1,1,-1,1),)
        self.pilot_carriers = pilot_carriers = (tuple(range(-n_carriers//2-2, -n_carriers//2)) + tuple(range(n_carriers//2+1, n_carriers//2+3)),)
        self.payload_mod = payload_mod = digital.constellation_qpsk()
        self.occupied_carriers = occupied_carriers = (tuple(range(-n_carriers//2, 0)) + tuple(range(1, n_carriers//2+1)),)
        self.hdr_payload_syms = hdr_payload_syms = ofdm_syms_per_tag
        self.cp_len = cp_len = fft_len//4
        self.tx_repeat_period_ms = tx_repeat_period_ms = 1000
        self.tx_amp = tx_amp = 50e-3
        self.sync_word2 = sync_word2 = list(digital.ofdm_txrx._make_sync_word2(fft_len, occupied_carriers, pilot_carriers))
        self.sync_word1 = sync_word1 = list(digital.ofdm_txrx._make_sync_word1(fft_len, occupied_carriers, pilot_carriers))
        self.spectrum_decim = spectrum_decim = 4
        self.sionna_device = sionna_device = "cuda:0"
        self.samp_rate = samp_rate = subcarrier_spacing * fft_len
        self.rx_vlen_buf = rx_vlen_buf = int((2 + 1 + ofdm_syms_per_tag) * fft_len * 16)
        self.rx_chain_min_buf = rx_chain_min_buf = max(65536, int((2 + 1 + ofdm_syms_per_tag) * (fft_len + cp_len)))
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, payload_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols, 1, 0.1)
        self.hdr_ack_msg = hdr_ack_msg = pmt.dict_add(pmt.make_dict(), pmt.intern("frame_len"), pmt.from_long(hdr_payload_syms))
        self.gui_update_period = gui_update_period = 0.10
        self.frame_len_tag = frame_len_tag = "frame_len"
        self.TX_gain = TX_gain = 20
        self.RX_gain = RX_gain = 20

        ##################################################
        # Blocks
        ##################################################

        self._tx_repeat_period_ms_range = qtgui.Range(20, 2000, 1, 1000, 200)
        self._tx_repeat_period_ms_win = qtgui.RangeWidget(self._tx_repeat_period_ms_range, self.set_tx_repeat_period_ms, "tx_repeat_ms", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_repeat_period_ms_win)
        self._tx_amp_range = qtgui.Range(0, 200e-3, 1e-3, 50e-3, 200)
        self._tx_amp_win = qtgui.RangeWidget(self._tx_amp_range, self.set_tx_amp, "'tx_amp'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_amp_win)
        self._TX_gain_range = qtgui.Range(0, 40, 1, 20, 200)
        self._TX_gain_win = qtgui.RangeWidget(self._TX_gain_range, self.set_TX_gain, "tx_gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._TX_gain_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._RX_gain_range = qtgui.Range(0, 40, 1, 20, 200)
        self._RX_gain_win = qtgui.RangeWidget(self._RX_gain_range, self.set_RX_gain, "rx_gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._RX_gain_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='num_recv_frames=512,recv_buff_size=25000000',
                channels=[2],
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("RX1", 0)
        self.uhd_usrp_source_0.set_gain(RX_gain, 0)
        self.uhd_usrp_source_0.set_min_output_buffer(262144)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=[0],
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(TX_gain, 0)
        self.qtgui_freq_sink_x_0_0 = qtgui.freq_sink_c(
            fft_len, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Rx Spectrum', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0_0.set_update_time(gui_update_period)
        self.qtgui_freq_sink_x_0_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_0.enable_autoscale(True)
        self.qtgui_freq_sink_x_0_0.enable_grid(False)
        self.qtgui_freq_sink_x_0_0.set_fft_average(0.2)
        self.qtgui_freq_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_0.set_fft_window_normalized(True)



        labels = ['Rx Spectrum', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_0_win)
        self.qtgui_freq_sink_x_0_0.set_max_output_buffer(8192)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            16384, #size
            "OFDM RX symbols", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(gui_update_period)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.epy_sionna_phy_tx = epy_sionna_phy_tx.blk(device=sionna_device, tx_amp=tx_amp, repeat_period_ms=tx_repeat_period_ms, fft_len=fft_len, subcarrier_spacing=subcarrier_spacing, n_carriers=n_carriers, ofdm_syms_per_tag=ofdm_syms_per_tag, time_lead_s=0.05)
        self.epy_sionna_phy_tx.set_min_output_buffer((max(65536, int((2 + 1 + ofdm_syms_per_tag) * (fft_len + cp_len) * 2))))
        self.epy_fft_batch_rx = epy_fft_batch_rx.blk(fft_len=fft_len, device=sionna_device)
        self.epy_fft_batch_rx.set_min_output_buffer((int(rx_vlen_buf // 4)))
        self.epy_fft_batch_hdr = epy_fft_batch_hdr.blk(fft_len=fft_len, device=sionna_device)
        self.epy_fft_batch_hdr.set_min_output_buffer((int(rx_vlen_buf // 4)))
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, cp_len, False, 0.6)
        self.digital_ofdm_sync_sc_cfb_0.set_min_output_buffer(rx_chain_min_buf)
        self.digital_ofdm_serializer_vcc_0 = digital.ofdm_serializer_vcc(fft_len, occupied_carriers, frame_len_tag, "", 0, "", True)
        self.digital_ofdm_serializer_vcc_0.set_min_output_buffer((int(n_carriers * ofdm_syms_per_tag * 16)))
        self.digital_ofdm_frame_equalizer_vcvc_0 = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), cp_len, frame_len_tag, True, 0)
        self.digital_ofdm_frame_equalizer_vcvc_0.set_min_output_buffer((int(rx_vlen_buf // 4)))
        self.digital_ofdm_chanest_vcvc_0 = digital.ofdm_chanest_vcvc(sync_word1, sync_word2, 1, 0, 3, False)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            3,
            fft_len,
            cp_len,
            frame_len_tag,
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            samp_rate,
            (),
            0)
        self.digital_header_payload_demux_0.set_min_output_buffer(rx_chain_min_buf)
        self.blocks_null_sink_hdr = blocks.null_sink(gr.sizeof_gr_complex*fft_len)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0.set_min_output_buffer(rx_chain_min_buf)
        self.blocks_message_strobe_0 = blocks.message_strobe(hdr_ack_msg, int(tx_repeat_period_ms))
        self.blocks_keep_one_in_n_spectrum = blocks.keep_one_in_n(gr.sizeof_gr_complex*1, spectrum_decim)
        self.blocks_keep_one_in_n_spectrum.set_min_output_buffer(8192)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, (fft_len+cp_len))
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc((-2.0/fft_len))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.digital_header_payload_demux_0, 'header_data'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_keep_one_in_n_spectrum, 0), (self.qtgui_freq_sink_x_0_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.epy_fft_batch_hdr, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.epy_fft_batch_rx, 0))
        self.connect((self.digital_ofdm_chanest_vcvc_0, 0), (self.blocks_null_sink_hdr, 0))
        self.connect((self.digital_ofdm_frame_equalizer_vcvc_0, 0), (self.digital_ofdm_serializer_vcc_0, 0))
        self.connect((self.digital_ofdm_serializer_vcc_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.epy_fft_batch_hdr, 0), (self.digital_ofdm_chanest_vcvc_0, 0))
        self.connect((self.epy_fft_batch_rx, 0), (self.digital_ofdm_frame_equalizer_vcvc_0, 0))
        self.connect((self.epy_sionna_phy_tx, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_keep_one_in_n_spectrum, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "style1_ofdm_sensing")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_address(self):
        return self.address

    def set_address(self, address):
        self.address = address

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_ofdm_syms_per_tag(self):
        return self.ofdm_syms_per_tag

    def set_ofdm_syms_per_tag(self, ofdm_syms_per_tag):
        self.ofdm_syms_per_tag = ofdm_syms_per_tag
        self.set_hdr_payload_syms(self.ofdm_syms_per_tag)
        self.set_rx_chain_min_buf(max(65536, int((2 + 1 + self.ofdm_syms_per_tag) * (self.fft_len + self.cp_len))))
        self.set_rx_vlen_buf(int((2 + 1 + self.ofdm_syms_per_tag) * self.fft_len * 16))
        self.epy_sionna_phy_tx.ofdm_syms_per_tag = self.ofdm_syms_per_tag

    def get_n_carriers(self):
        return self.n_carriers

    def set_n_carriers(self, n_carriers):
        self.n_carriers = n_carriers
        self.set_occupied_carriers((tuple(range(-self.n_carriers//2, 0)) + tuple(range(1, self.n_carriers//2+1)),))
        self.set_pilot_carriers((tuple(range(-self.n_carriers//2-2, -self.n_carriers//2)) + tuple(range(self.n_carriers//2+1, self.n_carriers//2+3)),))
        self.epy_sionna_phy_tx.n_carriers = self.n_carriers

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_cp_len(self.fft_len//4)
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1, 0.1))
        self.set_rx_chain_min_buf(max(65536, int((2 + 1 + self.ofdm_syms_per_tag) * (self.fft_len + self.cp_len))))
        self.set_rx_vlen_buf(int((2 + 1 + self.ofdm_syms_per_tag) * self.fft_len * 16))
        self.set_samp_rate(self.subcarrier_spacing * self.fft_len)
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.analog_frequency_modulator_fc_0.set_sensitivity((-2.0/self.fft_len))
        self.blocks_delay_0.set_dly(int((self.fft_len+self.cp_len)))

    def get_subcarrier_spacing(self):
        return self.subcarrier_spacing

    def set_subcarrier_spacing(self, subcarrier_spacing):
        self.subcarrier_spacing = subcarrier_spacing
        self.set_samp_rate(self.subcarrier_spacing * self.fft_len)
        self.epy_sionna_phy_tx.subcarrier_spacing = self.subcarrier_spacing

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1, 0.1))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1, 0.1))
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_payload_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, payload_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1, 0.1))
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))

    def get_hdr_payload_syms(self):
        return self.hdr_payload_syms

    def set_hdr_payload_syms(self, hdr_payload_syms):
        self.hdr_payload_syms = hdr_payload_syms
        self.set_hdr_ack_msg(pmt.dict_add(pmt.make_dict(), pmt.intern("frame_len"), pmt.from_long(self.hdr_payload_syms)))

    def get_cp_len(self):
        return self.cp_len

    def set_cp_len(self, cp_len):
        self.cp_len = cp_len
        self.set_rx_chain_min_buf(max(65536, int((2 + 1 + self.ofdm_syms_per_tag) * (self.fft_len + self.cp_len))))
        self.blocks_delay_0.set_dly(int((self.fft_len+self.cp_len)))

    def get_tx_repeat_period_ms(self):
        return self.tx_repeat_period_ms

    def set_tx_repeat_period_ms(self, tx_repeat_period_ms):
        self.tx_repeat_period_ms = tx_repeat_period_ms
        self.blocks_message_strobe_0.set_period(int(self.tx_repeat_period_ms))
        self.epy_sionna_phy_tx.repeat_period_ms = self.tx_repeat_period_ms

    def get_tx_amp(self):
        return self.tx_amp

    def set_tx_amp(self, tx_amp):
        self.tx_amp = tx_amp
        self.epy_sionna_phy_tx.tx_amp = self.tx_amp

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_spectrum_decim(self):
        return self.spectrum_decim

    def set_spectrum_decim(self, spectrum_decim):
        self.spectrum_decim = spectrum_decim
        self.blocks_keep_one_in_n_spectrum.set_n(self.spectrum_decim)

    def get_sionna_device(self):
        return self.sionna_device

    def set_sionna_device(self, sionna_device):
        self.sionna_device = sionna_device
        self.epy_fft_batch_hdr.device = self.sionna_device
        self.epy_fft_batch_rx.device = self.sionna_device
        self.epy_sionna_phy_tx.device = self.sionna_device

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0_0.set_frequency_range(0, self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_vlen_buf(self):
        return self.rx_vlen_buf

    def set_rx_vlen_buf(self, rx_vlen_buf):
        self.rx_vlen_buf = rx_vlen_buf

    def get_rx_chain_min_buf(self):
        return self.rx_chain_min_buf

    def set_rx_chain_min_buf(self, rx_chain_min_buf):
        self.rx_chain_min_buf = rx_chain_min_buf

    def get_payload_equalizer(self):
        return self.payload_equalizer

    def set_payload_equalizer(self, payload_equalizer):
        self.payload_equalizer = payload_equalizer

    def get_hdr_ack_msg(self):
        return self.hdr_ack_msg

    def set_hdr_ack_msg(self, hdr_ack_msg):
        self.hdr_ack_msg = hdr_ack_msg
        self.blocks_message_strobe_0.set_msg(self.hdr_ack_msg)

    def get_gui_update_period(self):
        return self.gui_update_period

    def set_gui_update_period(self, gui_update_period):
        self.gui_update_period = gui_update_period
        self.qtgui_const_sink_x_0.set_update_time(self.gui_update_period)
        self.qtgui_freq_sink_x_0_0.set_update_time(self.gui_update_period)

    def get_frame_len_tag(self):
        return self.frame_len_tag

    def set_frame_len_tag(self, frame_len_tag):
        self.frame_len_tag = frame_len_tag

    def get_TX_gain(self):
        return self.TX_gain

    def set_TX_gain(self, TX_gain):
        self.TX_gain = TX_gain
        self.uhd_usrp_sink_0.set_gain(self.TX_gain, 0)

    def get_RX_gain(self):
        return self.RX_gain

    def set_RX_gain(self, RX_gain):
        self.RX_gain = RX_gain
        self.uhd_usrp_source_0.set_gain(self.RX_gain, 0)



def argument_parser():
    description = 'Style1 burst OFDM TX (Sionna) + GR OFDM RX sensing (X410 OTA)'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--address", dest="address", type=str, default="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2",
        help="Set UHD dev args [default=%(default)r]")
    parser.add_argument(
        "-f", "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(6.0e9)),
        help="Set Default Frequency [default=%(default)r]")
    return parser


def main(top_block_cls=style1_ofdm_sensing, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(address=options.address, freq=options.freq)

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
