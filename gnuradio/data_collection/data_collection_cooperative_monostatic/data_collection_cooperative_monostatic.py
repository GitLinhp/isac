#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Cooperative monostatic sensing data collection
# Description: Dual-USRP cooperative monostatic OFDM range profile collection; both devices TX+RX simultaneously (aligned with usrp_ofdm_echotimer_dd_seperate).
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
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
from isac_imp.mics_test_record_flow import install_mics_test_record_flow
from isac_imp.record_paths import repo_data_dir
import data_collection_cooperative_monostatic_ofdm_range_profile_dev0 as ofdm_range_profile_dev0  # embedded python block
import data_collection_cooperative_monostatic_ofdm_range_profile_dev1 as ofdm_range_profile_dev1  # embedded python block
import data_collection_cooperative_monostatic_range_music_block_dev0 as range_music_block_dev0  # embedded python block
import data_collection_cooperative_monostatic_range_music_block_dev1 as range_music_block_dev1  # embedded python block
import data_collection_cooperative_monostatic_range_profile_plot_dev0 as range_profile_plot_dev0  # embedded python block
import data_collection_cooperative_monostatic_range_profile_plot_dev1 as range_profile_plot_dev1  # embedded python block
import data_collection_cooperative_monostatic_range_profile_record_limiter_dev0 as range_profile_record_limiter_dev0  # embedded python block
import data_collection_cooperative_monostatic_range_profile_record_limiter_dev1 as range_profile_record_limiter_dev1  # embedded python block
import data_collection_cooperative_monostatic_sionna_resource_grid_tx as sionna_resource_grid_tx  # embedded python block
import sip
import threading


def snipfcn_snippet_install_record_flow_0(self):
    install_mics_test_record_flow(self)


def snippets_main_after_init(tb):
    snipfcn_snippet_install_record_flow_0(tb)

class data_collection_cooperative_monostatic(gr.top_block, Qt.QWidget):

    def __init__(self, address0='type=x4xx,serial=33ABFDE,mgmt_addr=192.168.1.101,addr=192.168.11.2,clock_source=external,time_source=external', address1='type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=external,time_source=external'):
        gr.top_block.__init__(self, "Cooperative monostatic sensing data collection", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Cooperative monostatic sensing data collection")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "data_collection_cooperative_monostatic")

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
        self.address0 = address0
        self.address1 = address1

        ##################################################
        # Variables
        ##################################################
        self.fft_len = fft_len = 2048
        self.transpose_len = transpose_len = 4
        self.subcarrier_spacing = subcarrier_spacing = 120e3
        self.n_carriers = n_carriers = fft_len - 2
        self.zeropadding_fac = zeropadding_fac = 2
        self.samp_rate = samp_rate = int(fft_len * subcarrier_spacing)
        self.record_enable = record_enable = False
        self.packet_len = packet_len = transpose_len * n_carriers // 4
        self.wait_to_start = wait_to_start = 0.03
        self.record_output_index_dev1 = record_output_index_dev1 = 0 if record_enable else 1
        self.record_output_index_dev0 = record_output_index_dev0 = 0 if record_enable else 1
        self.record_output_dir_dev1 = record_output_dir_dev1 = repo_data_dir("data", "experiment", "cooperative_monostatic", "dev1")
        self.record_output_dir_dev0 = record_output_dir_dev0 = repo_data_dir("data", "experiment", "cooperative_monostatic", "dev0")
        self.record_max_frames = record_max_frames = 100
        self.record_file_path_dev1 = record_file_path_dev1 = "/dev/null"
        self.record_file_path_dev0 = record_file_path_dev0 = "/dev/null"
        self.range_roi = range_roi = (0.0, 30.0)
        self.range_bin_step = range_bin_step = 3e8/(2*int(fft_len*subcarrier_spacing)*zeropadding_fac)
        self.num_delay_samp1 = num_delay_samp1 = 282
        self.num_delay_samp0 = num_delay_samp0 = 282
        self.music_enable = music_enable = False
        self.min_out_buf_val = min_out_buf_val = packet_len*2
        self.length_tag_key = length_tag_key = "packet_len"
        self.freq1 = freq1 = 5.97e9
        self.freq0 = freq0 = 6.03e9
        self.factor = factor = 0.008
        self.device = device = "cpu"
        self.burst_len_samples = burst_len_samples = transpose_len * (fft_len + fft_len//4)
        self.TX_gain1 = TX_gain1 = 20
        self.TX_gain0 = TX_gain0 = 20
        self.R_max = R_max = 3e8/2/samp_rate*fft_len
        self.RX_gain1 = RX_gain1 = 20
        self.RX_gain0 = RX_gain0 = 20

        ##################################################
        # Blocks
        ##################################################

        self._num_delay_samp1_range = qtgui.Range(0, packet_len, 1, 282, 200)
        self._num_delay_samp1_win = qtgui.RangeWidget(self._num_delay_samp1_range, self.set_num_delay_samp1, "Number of delayed samples", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._num_delay_samp1_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._num_delay_samp0_range = qtgui.Range(0, packet_len, 1, 282, 200)
        self._num_delay_samp0_win = qtgui.RangeWidget(self._num_delay_samp0_range, self.set_num_delay_samp0, "Number of delayed samples", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._num_delay_samp0_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        _music_enable_check_box = Qt.QCheckBox("MUSIC Enable")
        self._music_enable_choices = {True: True, False: False}
        self._music_enable_choices_inv = dict((v,k) for k,v in self._music_enable_choices.items())
        self._music_enable_callback = lambda i: Qt.QMetaObject.invokeMethod(_music_enable_check_box, "setChecked", Qt.Q_ARG("bool", self._music_enable_choices_inv[i]))
        self._music_enable_callback(self.music_enable)
        _music_enable_check_box.stateChanged.connect(lambda i: self.set_music_enable(self._music_enable_choices[bool(i)]))
        self.top_grid_layout.addWidget(_music_enable_check_box, 0, 4, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(4, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._TX_gain1_range = qtgui.Range(0, packet_len, 1, 20, 200)
        self._TX_gain1_win = qtgui.RangeWidget(self._TX_gain1_range, self.set_TX_gain1, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._TX_gain1_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._TX_gain0_range = qtgui.Range(0, packet_len, 1, 20, 200)
        self._TX_gain0_win = qtgui.RangeWidget(self._TX_gain0_range, self.set_TX_gain0, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._TX_gain0_win, 3, 2, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._RX_gain1_range = qtgui.Range(0, packet_len, 1, 20, 200)
        self._RX_gain1_win = qtgui.RangeWidget(self._RX_gain1_range, self.set_RX_gain1, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._RX_gain1_win, 0, 3, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._RX_gain0_range = qtgui.Range(0, packet_len, 1, 20, 200)
        self._RX_gain0_win = qtgui.RangeWidget(self._RX_gain0_range, self.set_RX_gain0, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._RX_gain0_win, 3, 3, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.sionna_resource_grid_tx = sionna_resource_grid_tx.SionnaResourceGridTxBlock(fft_len=fft_len, transpose_len=transpose_len, subcarrier_spacing=subcarrier_spacing, cp_len=fft_len//4, length_tag_key=length_tag_key, num_bits_per_symbol=2, device=device, seed=42)
        self.sionna_resource_grid_tx.set_min_output_buffer((4*transpose_len))
        self.range_profile_record_limiter_dev1 = range_profile_record_limiter_dev1.RangeProfileRecordLimiter(vlen_in=fft_len*zeropadding_fac, record_enable=record_enable, record_max_frames=int(record_max_frames), record_output_dir_override=record_output_dir_dev1, file_sink_attr="blocks_file_sink_dev1", record_file_path_attr="record_file_path_dev1")
        self.range_profile_record_limiter_dev0 = range_profile_record_limiter_dev0.RangeProfileRecordLimiter(vlen_in=fft_len*zeropadding_fac, record_enable=record_enable, record_max_frames=int(record_max_frames), record_output_dir_override=record_output_dir_dev0, file_sink_attr="blocks_file_sink_dev0", record_file_path_attr="record_file_path_dev0")
        self.range_profile_plot_dev1 = range_profile_plot_dev1.RangeProfilePlotBlock(vlen_in=fft_len*zeropadding_fac, range_roi=range_roi, range_bin_step=range_bin_step)
        self.range_profile_plot_dev0 = range_profile_plot_dev0.RangeProfilePlotBlock(vlen_in=fft_len*zeropadding_fac, range_roi=range_roi, range_bin_step=range_bin_step)
        self.range_music_block_dev1 = range_music_block_dev1.RangeMusicBlock(vlen_in=fft_len*zeropadding_fac, range_bin_step=range_bin_step, range_roi=range_roi, num_sources=1, music_enable=music_enable, subarray_size=16, threshold=0.1)
        self.range_music_block_dev0 = range_music_block_dev0.RangeMusicBlock(vlen_in=fft_len*zeropadding_fac, range_bin_step=range_bin_step, range_roi=range_roi, num_sources=1, music_enable=music_enable, subarray_size=16, threshold=0.1)
        self.radar_usrp_echotimer_cc_0_0 = radar.usrp_echotimer_cc(int(samp_rate), freq0, int(num_delay_samp0), address0, 0, '', 'external', 'external', 'TX/RX', TX_gain0, 0.2, wait_to_start, 0, address0, 0, '', 'external', 'external', 'RX1', RX_gain0, 0.2, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0_0.set_min_output_buffer(min_out_buf_val)
        self.radar_usrp_echotimer_cc_0 = radar.usrp_echotimer_cc(int(samp_rate), freq1, int(num_delay_samp1), address1, 0, '', 'external', 'external', 'TX/RX', TX_gain1, 0.2, wait_to_start, 0, address1, 0, '', 'external', 'external', 'RX1', RX_gain1, 0.2, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0.set_min_output_buffer(min_out_buf_val)
        self.radar_ofdm_cyclic_prefix_remover_cvc_0_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0_0.set_min_output_buffer((2*transpose_len))
        self.radar_ofdm_cyclic_prefix_remover_cvc_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0.set_min_output_buffer((2*transpose_len))
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            (fft_len + fft_len//4), #size
            samp_rate, #samp_rate
            '', #name
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
        self.qtgui_freq_sink_x_dev1 = qtgui.freq_sink_c(
            fft_len, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            '', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_dev1.set_update_time(0.10)
        self.qtgui_freq_sink_x_dev1.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_dev1.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_dev1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_dev1.enable_autoscale(False)
        self.qtgui_freq_sink_x_dev1.enable_grid(False)
        self.qtgui_freq_sink_x_dev1.set_fft_average(1.0)
        self.qtgui_freq_sink_x_dev1.enable_axis_labels(True)
        self.qtgui_freq_sink_x_dev1.enable_control_panel(False)
        self.qtgui_freq_sink_x_dev1.set_fft_window_normalized(False)



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
                self.qtgui_freq_sink_x_dev1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_dev1.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_dev1.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_dev1.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_dev1.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_dev1_win = sip.wrapinstance(self.qtgui_freq_sink_x_dev1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_dev1_win)
        self.qtgui_freq_sink_x_dev0 = qtgui.freq_sink_c(
            fft_len, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            '', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_dev0.set_update_time(0.10)
        self.qtgui_freq_sink_x_dev0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_dev0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_dev0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_dev0.enable_autoscale(False)
        self.qtgui_freq_sink_x_dev0.enable_grid(False)
        self.qtgui_freq_sink_x_dev0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_dev0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_dev0.enable_control_panel(False)
        self.qtgui_freq_sink_x_dev0.set_fft_window_normalized(False)



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
                self.qtgui_freq_sink_x_dev0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_dev0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_dev0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_dev0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_dev0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_dev0_win = sip.wrapinstance(self.qtgui_freq_sink_x_dev0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_dev0_win)
        self.ofdm_range_profile_dev1 = ofdm_range_profile_dev1.OfdmRangeProfileBlock(fft_len=fft_len, zeropadding_fac=zeropadding_fac, transpose_len=transpose_len)
        self.ofdm_range_profile_dev0 = ofdm_range_profile_dev0.OfdmRangeProfileBlock(fft_len=fft_len, zeropadding_fac=zeropadding_fac, transpose_len=transpose_len)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fft_vxx_0.set_min_output_buffer((2*transpose_len))
        self.fft_rx_dev1 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_rx_dev1.set_min_output_buffer((2*transpose_len))
        self.fft_rx_dev0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_rx_dev0.set_min_output_buffer((2*transpose_len))
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + fft_len//4,
            0,
            length_tag_key)
        self.digital_ofdm_cyclic_prefixer_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.blocks_selector_dev1 = blocks.selector(gr.sizeof_gr_complex*(fft_len*zeropadding_fac),0,record_output_index_dev1)
        self.blocks_selector_dev1.set_enabled(True)
        self.blocks_selector_dev0 = blocks.selector(gr.sizeof_gr_complex*(fft_len*zeropadding_fac),0,record_output_index_dev0)
        self.blocks_selector_dev0.set_enabled(True)
        self.blocks_null_sink_rec_dev1 = blocks.null_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac))
        self.blocks_null_sink_rec_dev0 = blocks.null_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(factor)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))
        self.blocks_file_sink_dev1 = blocks.file_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac), record_file_path_dev1, False)
        self.blocks_file_sink_dev1.set_unbuffered(False)
        self.blocks_file_sink_dev0 = blocks.file_sink(gr.sizeof_gr_complex*(fft_len*zeropadding_fac), record_file_path_dev0, False)
        self.blocks_file_sink_dev0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.radar_usrp_echotimer_cc_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.radar_usrp_echotimer_cc_0_0, 0))
        self.connect((self.blocks_selector_dev0, 0), (self.blocks_file_sink_dev0, 0))
        self.connect((self.blocks_selector_dev0, 1), (self.blocks_null_sink_rec_dev0, 0))
        self.connect((self.blocks_selector_dev1, 0), (self.blocks_file_sink_dev1, 0))
        self.connect((self.blocks_selector_dev1, 1), (self.blocks_null_sink_rec_dev1, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.fft_rx_dev0, 0), (self.ofdm_range_profile_dev0, 1))
        self.connect((self.fft_rx_dev1, 0), (self.ofdm_range_profile_dev1, 1))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.ofdm_range_profile_dev0, 1), (self.range_music_block_dev0, 0))
        self.connect((self.ofdm_range_profile_dev0, 0), (self.range_profile_plot_dev0, 0))
        self.connect((self.ofdm_range_profile_dev0, 1), (self.range_profile_record_limiter_dev0, 0))
        self.connect((self.ofdm_range_profile_dev1, 1), (self.range_music_block_dev1, 0))
        self.connect((self.ofdm_range_profile_dev1, 0), (self.range_profile_plot_dev1, 0))
        self.connect((self.ofdm_range_profile_dev1, 1), (self.range_profile_record_limiter_dev1, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_rx_dev1, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0_0, 0), (self.fft_rx_dev0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.qtgui_freq_sink_x_dev1, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0_0, 0), (self.qtgui_freq_sink_x_dev0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0_0, 0))
        self.connect((self.range_profile_record_limiter_dev0, 0), (self.blocks_selector_dev0, 0))
        self.connect((self.range_profile_record_limiter_dev1, 0), (self.blocks_selector_dev1, 0))
        self.connect((self.sionna_resource_grid_tx, 0), (self.fft_vxx_0, 0))
        self.connect((self.sionna_resource_grid_tx, 0), (self.ofdm_range_profile_dev0, 0))
        self.connect((self.sionna_resource_grid_tx, 0), (self.ofdm_range_profile_dev1, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "data_collection_cooperative_monostatic")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_address0(self):
        return self.address0

    def set_address0(self, address0):
        self.address0 = address0

    def get_address1(self):
        return self.address1

    def set_address1(self, address1):
        self.address1 = address1

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))
        self.set_n_carriers(self.fft_len - 2)
        self.set_burst_len_samples(self.transpose_len * (self.fft_len + self.fft_len//4))
        self.set_range_bin_step(3e8/(2*int(self.fft_len*self.subcarrier_spacing)*self.zeropadding_fac))
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.set_packet_len(self.transpose_len * self.n_carriers // 4)
        self.set_burst_len_samples(self.transpose_len * (self.fft_len + self.fft_len//4))

    def get_subcarrier_spacing(self):
        return self.subcarrier_spacing

    def set_subcarrier_spacing(self, subcarrier_spacing):
        self.subcarrier_spacing = subcarrier_spacing
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))
        self.set_range_bin_step(3e8/(2*int(self.fft_len*self.subcarrier_spacing)*self.zeropadding_fac))

    def get_n_carriers(self):
        return self.n_carriers

    def set_n_carriers(self, n_carriers):
        self.n_carriers = n_carriers
        self.set_packet_len(self.transpose_len * self.n_carriers // 4)

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.set_range_bin_step(3e8/(2*int(self.fft_len*self.subcarrier_spacing)*self.zeropadding_fac))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.qtgui_freq_sink_x_dev0.set_frequency_range(0, self.samp_rate)
        self.qtgui_freq_sink_x_dev1.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_record_enable(self):
        return self.record_enable

    def set_record_enable(self, record_enable):
        self.record_enable = record_enable
        self.set_record_output_index_dev0(0 if self.record_enable else 1)
        self.set_record_output_index_dev1(0 if self.record_enable else 1)
        self.range_profile_record_limiter_dev0.record_enable = self.record_enable
        self.range_profile_record_limiter_dev1.record_enable = self.record_enable

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_min_out_buf_val(self.packet_len*2)

    def get_wait_to_start(self):
        return self.wait_to_start

    def set_wait_to_start(self, wait_to_start):
        self.wait_to_start = wait_to_start

    def get_record_output_index_dev1(self):
        return self.record_output_index_dev1

    def set_record_output_index_dev1(self, record_output_index_dev1):
        self.record_output_index_dev1 = record_output_index_dev1
        self.blocks_selector_dev1.set_output_index(self.record_output_index_dev1)

    def get_record_output_index_dev0(self):
        return self.record_output_index_dev0

    def set_record_output_index_dev0(self, record_output_index_dev0):
        self.record_output_index_dev0 = record_output_index_dev0
        self.blocks_selector_dev0.set_output_index(self.record_output_index_dev0)

    def get_record_output_dir_dev1(self):
        return self.record_output_dir_dev1

    def set_record_output_dir_dev1(self, record_output_dir_dev1):
        self.record_output_dir_dev1 = record_output_dir_dev1

    def get_record_output_dir_dev0(self):
        return self.record_output_dir_dev0

    def set_record_output_dir_dev0(self, record_output_dir_dev0):
        self.record_output_dir_dev0 = record_output_dir_dev0

    def get_record_max_frames(self):
        return self.record_max_frames

    def set_record_max_frames(self, record_max_frames):
        self.record_max_frames = record_max_frames
        self.range_profile_record_limiter_dev0.record_max_frames = int(self.record_max_frames)
        self.range_profile_record_limiter_dev1.record_max_frames = int(self.record_max_frames)

    def get_record_file_path_dev1(self):
        return self.record_file_path_dev1

    def set_record_file_path_dev1(self, record_file_path_dev1):
        self.record_file_path_dev1 = record_file_path_dev1
        self.blocks_file_sink_dev1.open(self.record_file_path_dev1)

    def get_record_file_path_dev0(self):
        return self.record_file_path_dev0

    def set_record_file_path_dev0(self, record_file_path_dev0):
        self.record_file_path_dev0 = record_file_path_dev0
        self.blocks_file_sink_dev0.open(self.record_file_path_dev0)

    def get_range_roi(self):
        return self.range_roi

    def set_range_roi(self, range_roi):
        self.range_roi = range_roi
        self.range_profile_plot_dev0.range_roi = self.range_roi
        self.range_music_block_dev0.range_roi = self.range_roi
        self.range_profile_plot_dev1.range_roi = self.range_roi
        self.range_music_block_dev1.range_roi = self.range_roi

    def get_range_bin_step(self):
        return self.range_bin_step

    def set_range_bin_step(self, range_bin_step):
        self.range_bin_step = range_bin_step
        self.range_profile_plot_dev0.range_bin_step = self.range_bin_step
        self.range_music_block_dev0.range_bin_step = self.range_bin_step
        self.range_profile_plot_dev1.range_bin_step = self.range_bin_step
        self.range_music_block_dev1.range_bin_step = self.range_bin_step

    def get_num_delay_samp1(self):
        return self.num_delay_samp1

    def set_num_delay_samp1(self, num_delay_samp1):
        self.num_delay_samp1 = num_delay_samp1
        self.radar_usrp_echotimer_cc_0.set_num_delay_samps(int(self.num_delay_samp1))

    def get_num_delay_samp0(self):
        return self.num_delay_samp0

    def set_num_delay_samp0(self, num_delay_samp0):
        self.num_delay_samp0 = num_delay_samp0
        self.radar_usrp_echotimer_cc_0_0.set_num_delay_samps(int(self.num_delay_samp0))

    def get_music_enable(self):
        return self.music_enable

    def set_music_enable(self, music_enable):
        self.music_enable = music_enable
        self._music_enable_callback(self.music_enable)
        self.range_music_block_dev0.music_enable = self.music_enable
        self.range_music_block_dev1.music_enable = self.music_enable

    def get_min_out_buf_val(self):
        return self.min_out_buf_val

    def set_min_out_buf_val(self, min_out_buf_val):
        self.min_out_buf_val = min_out_buf_val

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key

    def get_freq1(self):
        return self.freq1

    def set_freq1(self, freq1):
        self.freq1 = freq1

    def get_freq0(self):
        return self.freq0

    def set_freq0(self, freq0):
        self.freq0 = freq0

    def get_factor(self):
        return self.factor

    def set_factor(self, factor):
        self.factor = factor
        self.blocks_multiply_const_vxx_0.set_k(self.factor)

    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = device

    def get_burst_len_samples(self):
        return self.burst_len_samples

    def set_burst_len_samples(self, burst_len_samples):
        self.burst_len_samples = burst_len_samples

    def get_TX_gain1(self):
        return self.TX_gain1

    def set_TX_gain1(self, TX_gain1):
        self.TX_gain1 = TX_gain1
        self.radar_usrp_echotimer_cc_0.set_tx_gain(self.TX_gain1)

    def get_TX_gain0(self):
        return self.TX_gain0

    def set_TX_gain0(self, TX_gain0):
        self.TX_gain0 = TX_gain0
        self.radar_usrp_echotimer_cc_0_0.set_tx_gain(self.TX_gain0)

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max

    def get_RX_gain1(self):
        return self.RX_gain1

    def set_RX_gain1(self, RX_gain1):
        self.RX_gain1 = RX_gain1
        self.radar_usrp_echotimer_cc_0.set_rx_gain(self.RX_gain1)

    def get_RX_gain0(self):
        return self.RX_gain0

    def set_RX_gain0(self, RX_gain0):
        self.RX_gain0 = RX_gain0
        self.radar_usrp_echotimer_cc_0_0.set_rx_gain(self.RX_gain0)



def argument_parser():
    description = 'Dual-USRP cooperative monostatic OFDM range profile collection; both devices TX+RX simultaneously (aligned with usrp_ofdm_echotimer_dd_seperate).'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--address0", dest="address0", type=str, default='type=x4xx,serial=33ABFDE,mgmt_addr=192.168.1.101,addr=192.168.11.2,clock_source=external,time_source=external',
        help="Set address0 (33ABFDE) [default=%(default)r]")
    parser.add_argument(
        "--address1", dest="address1", type=str, default='type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=external,time_source=external',
        help="Set address1 (349B642) [default=%(default)r]")
    return parser


def main(top_block_cls=data_collection_cooperative_monostatic, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(address0=options.address0, address1=options.address1)
    snippets_main_after_init(tb)
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
