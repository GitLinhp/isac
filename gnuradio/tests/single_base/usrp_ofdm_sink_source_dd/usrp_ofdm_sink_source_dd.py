#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Usrp Ofdm Sink Source Dd
# Description: USRP Sink/Source Style1 OFDM radar zero-Doppler range profile. Scheduled RX (NUM_SAMPS_AND_DONE per CPI); packet_len via packet_len_tagger.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
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
from gnuradio import uhd
import time
import sip
import threading
import usrp_ofdm_sink_source_dd_burst_iq_tag_tx_0 as burst_iq_tag_tx_0  # embedded python block
import usrp_ofdm_sink_source_dd_cpi_rate_probe_0 as cpi_rate_probe_0  # embedded python block
import usrp_ofdm_sink_source_dd_echotimer_rx_compensator_0 as echotimer_rx_compensator_0  # embedded python block
import usrp_ofdm_sink_source_dd_ofdm_cpi_tag_normalizer_rx_0 as ofdm_cpi_tag_normalizer_rx_0  # embedded python block
import usrp_ofdm_sink_source_dd_ofdm_range_profile_0 as ofdm_range_profile_0  # embedded python block
import usrp_ofdm_sink_source_dd_packet_len_tagger_0 as packet_len_tagger_0  # embedded python block
import usrp_ofdm_sink_source_dd_range_music_block_dev1 as range_music_block_dev1  # embedded python block
import usrp_ofdm_sink_source_dd_range_profile_plot_dev1 as range_profile_plot_dev1  # embedded python block
import usrp_ofdm_sink_source_dd_sionna_ofdm_modulator_0 as sionna_ofdm_modulator_0  # embedded python block
import usrp_ofdm_sink_source_dd_sionna_resource_grid_tx_0 as sionna_resource_grid_tx_0  # embedded python block


def snipfcn_patch_usrp_source_factory_snippet(self):
    from isac_imp.scheduled_usrp_source import patch_usrp_source_factory
    patch_usrp_source_factory()

def snipfcn_scheduled_rx_bind_snippet(self):
    import time
    from gnuradio import uhd
    now = time.time()
    self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(now), uhd.ALL_MBOARDS)
    self.uhd_usrp_source_0.set_time_now(uhd.time_spec(now), uhd.ALL_MBOARDS)
    self.burst_iq_tag_tx_0.bind_scheduled_rx(self.uhd_usrp_source_0)


def snippets_main_after_init(tb):
    snipfcn_scheduled_rx_bind_snippet(tb)

def snippets_init_before_blocks(tb):
    snipfcn_patch_usrp_source_factory_snippet(tb)

class usrp_ofdm_sink_source_dd(gr.top_block, Qt.QWidget):

    def __init__(self, address="type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=internal,time_source=internal"):
        gr.top_block.__init__(self, "Usrp Ofdm Sink Source Dd", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Usrp Ofdm Sink Source Dd")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_sink_source_dd")

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

        ##################################################
        # Variables
        ##################################################
        self.fft_len = fft_len = 2048
        self.transpose_len = transpose_len = 4
        self.subcarrier_spacing = subcarrier_spacing = 60e3
        self.n_carriers = n_carriers = fft_len - 2
        self.zeropadding_fac = zeropadding_fac = 4
        self.wait_to_start = wait_to_start = 0.5
        self.samp_rate = samp_rate = int(fft_len * subcarrier_spacing)
        self.packet_len = packet_len = transpose_len * n_carriers // 4
        self.time_lead_s = time_lead_s = wait_to_start
        self.range_roi = range_roi = (0.0, 3.5)
        self.range_bin_step = range_bin_step = 3e8/(2*int(fft_len*subcarrier_spacing)*zeropadding_fac)
        self.num_delay_samp = num_delay_samp = 161
        self.music_enable = music_enable = False
        self.min_out_buf_val = min_out_buf_val = packet_len*2
        self.length_tag_key = length_tag_key = "packet_len"
        self.idle_ms = idle_ms = 10
        self.freq = freq = 6.0e9
        self.factor = factor = 1.204e-5
        self.burst_len_samples = burst_len_samples = transpose_len * (fft_len + fft_len//4)
        self.TX_gain = TX_gain = 20
        self.R_max = R_max = 3e8/2/samp_rate*fft_len
        self.RX_gain = RX_gain = 20

        ##################################################
        # Blocks
        ##################################################
        snippets_init_before_blocks(self)
        self._num_delay_samp_range = qtgui.Range(0, packet_len, 1, 161, 200)
        self._num_delay_samp_win = qtgui.RangeWidget(self._num_delay_samp_range, self.set_num_delay_samp, "Number of delayed samples", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._num_delay_samp_win)
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
        self._factor_range = qtgui.Range(0, 1, 1e-6, 1.204e-5, 200)
        self._factor_win = qtgui.RangeWidget(self._factor_range, self.set_factor, "'factor'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._factor_win)
        self._TX_gain_range = qtgui.Range(0, 50, 1, 20, 200)
        self._TX_gain_win = qtgui.RangeWidget(self._TX_gain_range, self.set_TX_gain, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._TX_gain_win)
        self._RX_gain_range = qtgui.Range(0, 50, 1, 20, 200)
        self._RX_gain_win = qtgui.RangeWidget(self._RX_gain_range, self.set_RX_gain, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._RX_gain_win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='num_recv_frames=512,recv_buff_size=25000000',
                channels=[0],
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('internal', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("RX1", 0)
        self.uhd_usrp_source_0.set_gain(RX_gain, 0)
        self.uhd_usrp_source_0.set_min_output_buffer(min_out_buf_val)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='num_send_frames=512,send_buff_size=25000000',
                channels=[0],
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_clock_source('internal', 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(TX_gain, 0)
        self.sionna_resource_grid_tx_0 = sionna_resource_grid_tx_0.SionnaResourceGridTxBlock(fft_len=fft_len, transpose_len=transpose_len, subcarrier_spacing=subcarrier_spacing, cp_len=fft_len//4, length_tag_key=length_tag_key, num_bits_per_symbol=2, device='cpu', seed=42)
        self.sionna_resource_grid_tx_0.set_min_output_buffer((4*transpose_len))
        self.sionna_ofdm_modulator_0 = sionna_ofdm_modulator_0.SionnaOfdmModulatorBlock(fft_len=fft_len, cp_len=fft_len//4, burst_len_samples=burst_len_samples, transpose_len=transpose_len, subcarrier_spacing=subcarrier_spacing, num_bits_per_symbol=2, seed=42, target_peak=1.0, length_tag_key=length_tag_key)
        self.sionna_ofdm_modulator_0.set_min_output_buffer(burst_len_samples)
        self.range_profile_plot_dev1 = range_profile_plot_dev1.RangeProfilePlotBlock(vlen_in=fft_len*zeropadding_fac, range_roi=range_roi, range_bin_step=range_bin_step)
        self.range_music_block_dev1 = range_music_block_dev1.RangeMusicBlock(vlen_in=fft_len*zeropadding_fac, range_bin_step=range_bin_step, range_roi=range_roi, num_sources=1, music_enable=music_enable, subarray_size=16, threshold=0.1)
        self.radar_ofdm_cyclic_prefix_remover_cvc_0 = radar.ofdm_cyclic_prefix_remover_cvc(fft_len, (fft_len//4), "packet_len")
        self.radar_ofdm_cyclic_prefix_remover_cvc_0.set_min_output_buffer((2*transpose_len))
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
        self.packet_len_tagger_0 = packet_len_tagger_0.PacketLenTaggerBlock(burst_len_samples=burst_len_samples, length_tag_key=length_tag_key, use_rx_time=True)
        self.ofdm_range_profile_0 = ofdm_range_profile_0.OfdmRangeProfileBlock(fft_len=fft_len, zeropadding_fac=zeropadding_fac, transpose_len=transpose_len)
        self.ofdm_cpi_tag_normalizer_rx_0 = ofdm_cpi_tag_normalizer_rx_0.OfdmCpiTagNormalizerBlock(fft_len=fft_len, transpose_len=transpose_len, length_tag_key=length_tag_key)
        self.ofdm_cpi_tag_normalizer_rx_0.set_min_output_buffer(((2*transpose_len)))
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, True, (), True, 1)
        self.fft_vxx_0_0.set_min_output_buffer((2*transpose_len))
        self.echotimer_rx_compensator_0 = echotimer_rx_compensator_0.EchotimerRxCompensatorBlock(burst_len_samples=burst_len_samples, length_tag_key=length_tag_key, num_delay_samps=num_delay_samp)
        self.cpi_rate_probe_0 = cpi_rate_probe_0.CpiRateProbeBlock(length_tag_key=length_tag_key, log_interval_s=1.0, label='scheduled_rx')
        self.burst_iq_tag_tx_0 = burst_iq_tag_tx_0.BurstIqTagTxBlock(burst_len_samples=burst_len_samples, length_tag_key=length_tag_key, time_lead_s=time_lead_s, idle_ms=idle_ms, samp_rate=float(samp_rate), scheduled_rx=True, num_delay_samp=num_delay_samp)
        self.burst_iq_tag_tx_0.set_min_output_buffer(min_out_buf_val)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(factor)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer((int(2*transpose_len*(fft_len+fft_len/4))))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.burst_iq_tag_tx_0, 'tx_schedule'), (self.ofdm_range_profile_0, 'tx_schedule'))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.burst_iq_tag_tx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.burst_iq_tag_tx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.cpi_rate_probe_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.cpi_rate_probe_0, 0), (self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0))
        self.connect((self.echotimer_rx_compensator_0, 0), (self.cpi_rate_probe_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.ofdm_cpi_tag_normalizer_rx_0, 0))
        self.connect((self.ofdm_cpi_tag_normalizer_rx_0, 0), (self.ofdm_range_profile_0, 1))
        self.connect((self.ofdm_range_profile_0, 1), (self.range_music_block_dev1, 0))
        self.connect((self.ofdm_range_profile_0, 0), (self.range_profile_plot_dev1, 0))
        self.connect((self.packet_len_tagger_0, 0), (self.echotimer_rx_compensator_0, 0))
        self.connect((self.radar_ofdm_cyclic_prefix_remover_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.sionna_ofdm_modulator_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.sionna_resource_grid_tx_0, 0), (self.ofdm_range_profile_0, 0))
        self.connect((self.sionna_resource_grid_tx_0, 0), (self.sionna_ofdm_modulator_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.packet_len_tagger_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_sink_source_dd")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_address(self):
        return self.address

    def set_address(self, address):
        self.address = address

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_burst_len_samples(self.transpose_len * (self.fft_len + self.fft_len//4))
        self.set_n_carriers(self.fft_len - 2)
        self.set_range_bin_step(3e8/(2*int(self.fft_len*self.subcarrier_spacing)*self.zeropadding_fac))
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.set_burst_len_samples(self.transpose_len * (self.fft_len + self.fft_len//4))
        self.set_packet_len(self.transpose_len * self.n_carriers // 4)

    def get_subcarrier_spacing(self):
        return self.subcarrier_spacing

    def set_subcarrier_spacing(self, subcarrier_spacing):
        self.subcarrier_spacing = subcarrier_spacing
        self.set_range_bin_step(3e8/(2*int(self.fft_len*self.subcarrier_spacing)*self.zeropadding_fac))
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))

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

    def get_wait_to_start(self):
        return self.wait_to_start

    def set_wait_to_start(self, wait_to_start):
        self.wait_to_start = wait_to_start
        self.set_time_lead_s(self.wait_to_start)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.burst_iq_tag_tx_0.samp_rate = float(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_min_out_buf_val(self.packet_len*2)

    def get_time_lead_s(self):
        return self.time_lead_s

    def set_time_lead_s(self, time_lead_s):
        self.time_lead_s = time_lead_s
        self.burst_iq_tag_tx_0.time_lead_s = self.time_lead_s

    def get_range_roi(self):
        return self.range_roi

    def set_range_roi(self, range_roi):
        self.range_roi = range_roi
        self.range_music_block_dev1.range_roi = self.range_roi
        self.range_profile_plot_dev1.range_roi = self.range_roi

    def get_range_bin_step(self):
        return self.range_bin_step

    def set_range_bin_step(self, range_bin_step):
        self.range_bin_step = range_bin_step
        self.range_music_block_dev1.range_bin_step = self.range_bin_step
        self.range_profile_plot_dev1.range_bin_step = self.range_bin_step

    def get_num_delay_samp(self):
        return self.num_delay_samp

    def set_num_delay_samp(self, num_delay_samp):
        self.num_delay_samp = num_delay_samp
        self.burst_iq_tag_tx_0.num_delay_samp = self.num_delay_samp
        self.echotimer_rx_compensator_0.num_delay_samps = self.num_delay_samp

    def get_music_enable(self):
        return self.music_enable

    def set_music_enable(self, music_enable):
        self.music_enable = music_enable
        self._music_enable_callback(self.music_enable)
        self.range_music_block_dev1.music_enable = self.music_enable

    def get_min_out_buf_val(self):
        return self.min_out_buf_val

    def set_min_out_buf_val(self, min_out_buf_val):
        self.min_out_buf_val = min_out_buf_val

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key

    def get_idle_ms(self):
        return self.idle_ms

    def set_idle_ms(self, idle_ms):
        self.idle_ms = idle_ms
        self.burst_iq_tag_tx_0.idle_ms = self.idle_ms

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_factor(self):
        return self.factor

    def set_factor(self, factor):
        self.factor = factor
        self.blocks_multiply_const_vxx_0.set_k(self.factor)

    def get_burst_len_samples(self):
        return self.burst_len_samples

    def set_burst_len_samples(self, burst_len_samples):
        self.burst_len_samples = burst_len_samples
        self.burst_iq_tag_tx_0.burst_len_samples = self.burst_len_samples
        self.echotimer_rx_compensator_0.burst_len_samples = self.burst_len_samples
        self.packet_len_tagger_0.burst_len_samples = self.burst_len_samples
        self.sionna_ofdm_modulator_0.burst_len_samples = self.burst_len_samples

    def get_TX_gain(self):
        return self.TX_gain

    def set_TX_gain(self, TX_gain):
        self.TX_gain = TX_gain
        self.uhd_usrp_sink_0.set_gain(self.TX_gain, 0)

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max

    def get_RX_gain(self):
        return self.RX_gain

    def set_RX_gain(self, RX_gain):
        self.RX_gain = RX_gain
        self.uhd_usrp_source_0.set_gain(self.RX_gain, 0)



def argument_parser():
    description = 'USRP Sink/Source Style1 OFDM radar zero-Doppler range profile. Scheduled RX (NUM_SAMPS_AND_DONE per CPI); packet_len via packet_len_tagger.'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--address", dest="address", type=str, default="type=x4xx,serial=349B642,mgmt_addr=192.168.1.100,addr=192.168.10.2,clock_source=internal,time_source=internal",
        help="Set address (349B642) [default=%(default)r]")
    return parser


def main(top_block_cls=usrp_ofdm_sink_source_dd, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(address=options.address)
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
