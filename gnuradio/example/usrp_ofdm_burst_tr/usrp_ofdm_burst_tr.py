#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: USRP TX/RX (OFDM Burst)
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import digital, analog; import pmt; from isac_imp.gr_setup import (resolve_dd_output_vlen, resolve_ofdm_samp_rate, resolve_ofdm_burst_len, resolve_ofdm_fft_cp, resolve_ofdm_num_symbols, resolve_gr_carrier_tuples)
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
from isac_imp.blocks.dd_spectrum_plot import DDSpectrogramPlot
import threading
import usrp_ofdm_burst_tr_ofdm_burst_sensing_rx as ofdm_burst_sensing_rx  # embedded python block
import usrp_ofdm_burst_tr_ofdm_burst_source as ofdm_burst_source  # embedded python block



class usrp_ofdm_burst_tr(gr.top_block, Qt.QWidget):

    def __init__(self, address="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2", freq=6.0e9):
        gr.top_block.__init__(self, "USRP TX/RX (OFDM Burst)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("USRP TX/RX (OFDM Burst)")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_burst_tr")

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
        self.config_file = config_file = "implementaion/ofdm_burst_source_large_sacle.toml"
        self.fft_len = fft_len = resolve_ofdm_fft_cp(config_file)[0]
        self.n_carriers = n_carriers = fft_len - 2
        self.pilot_carriers = pilot_carriers = (tuple(range(-n_carriers//2-2, -n_carriers//2)) + tuple(range(n_carriers//2+1, n_carriers//2+3)),)
        self.occupied_carriers = occupied_carriers = (tuple(range(-n_carriers//2, 0)) + tuple(range(1, n_carriers//2+1)),)
        self.idle_ms = idle_ms = 1000
        self.header_len = header_len = 2
        self.cp_len = cp_len = resolve_ofdm_fft_cp(config_file)[1]
        self.tx_amp = tx_amp = 30
        self.time_mag_trig_level = time_mag_trig_level = 5e-3
        self.time_lead_s = time_lead_s = 0.5
        self.sync_word2 = sync_word2 = list(digital.ofdm_txrx._make_sync_word2(fft_len, occupied_carriers, pilot_carriers))
        self.sync_word1 = sync_word1 = list(digital.ofdm_txrx._make_sync_word1(fft_len, occupied_carriers, pilot_carriers))
        self.startup_delay_s = startup_delay_s = 1.0
        self.samp_rate = samp_rate = resolve_ofdm_samp_rate(config_file)
        self.rx_sync_min_buf = rx_sync_min_buf = max(65536, int((header_len + 8) * (fft_len + cp_len)))
        self.payload_syms = payload_syms = resolve_ofdm_num_symbols(config_file)
        self.ofdm_burst_samples = ofdm_burst_samples = resolve_ofdm_burst_len(config_file)
        self.hdr_ack_msg = hdr_ack_msg = pmt.dict_add(pmt.make_dict(), pmt.intern("frame_len"), pmt.from_long(resolve_ofdm_num_symbols(config_file)))
        self.gui_update_time_ms = gui_update_time_ms = 10
        self.freq_trig_level = freq_trig_level = -90
        self.frame_len_tag = frame_len_tag = "frame_len"
        self.device = device = "cuda:0"
        self.dd_vlen = dd_vlen = resolve_dd_output_vlen(config_file)
        self.corr_threshold = corr_threshold = 0.6
        self.burst_period_ms = burst_period_ms = idle_ms
        self.TX_gain = TX_gain = 20
        self.RX_gain = RX_gain = 20

        ##################################################
        # Blocks
        ##################################################

        self._tx_amp_range = qtgui.Range(0, 100, 0.01, 30, 200)
        self._tx_amp_win = qtgui.RangeWidget(self._tx_amp_range, self.set_tx_amp, "tx_amp", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tx_amp_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._TX_gain_range = qtgui.Range(0, 50, 1, 20, 200)
        self._TX_gain_win = qtgui.RangeWidget(self._TX_gain_range, self.set_TX_gain, "tx_gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._TX_gain_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._RX_gain_range = qtgui.Range(0, 50, 1, 20, 200)
        self._RX_gain_win = qtgui.RangeWidget(self._RX_gain_range, self.set_RX_gain, "rx_gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._RX_gain_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0_0 = uhd.usrp_source(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='num_recv_frames=512,recv_buff_size=25000000',
                channels=[2],
            ),
        )
        self.uhd_usrp_source_0_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0_0.set_antenna("RX1", 0)
        self.uhd_usrp_source_0_0.set_gain(RX_gain, 0)
        self.uhd_usrp_source_0_0.set_min_output_buffer(262144)
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join((address, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='num_send_frames=512,send_buff_size=25000000',
                channels=[0],
            ),
            '',
        )
        self.uhd_usrp_sink_0_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0_0.set_gain(TX_gain, 0)
        self._time_mag_trig_level_range = qtgui.Range(0, 0.5, 0.005, 5e-3, 200)
        self._time_mag_trig_level_win = qtgui.RangeWidget(self._time_mag_trig_level_range, self.set_time_mag_trig_level, "time_mag_trig_level", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._time_mag_trig_level_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.ofdm_burst_source = ofdm_burst_source.blk(config_file=config_file, idle_ms=idle_ms, tx_amp=tx_amp, time_lead_s=time_lead_s, startup_delay_s=startup_delay_s)
        self.ofdm_burst_sensing_rx = ofdm_burst_sensing_rx.blk(config_file=config_file, device=device, seed=42)
        self._freq_trig_level_range = qtgui.Range(-120, 0, 5, -90, 200)
        self._freq_trig_level_win = qtgui.RangeWidget(self._freq_trig_level_range, self.set_freq_trig_level, "freq_trig_level", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._freq_trig_level_win, 1, 2, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.digital_ofdm_sync_sc_cfb_0 = digital.ofdm_sync_sc_cfb(fft_len, cp_len, False, corr_threshold)
        self.digital_ofdm_sync_sc_cfb_0.set_min_output_buffer(rx_sync_min_buf)
        self.digital_header_payload_demux_0 = digital.header_payload_demux(
            header_len,
            fft_len,
            cp_len,
            frame_len_tag,
            "",
            True,
            gr.sizeof_gr_complex,
            "",
            samp_rate,
            (),
            0)
        self.digital_header_payload_demux_0.set_min_output_buffer(rx_sync_min_buf)
        self.dd_spectrum_plot_0 = DDSpectrogramPlot(vlen=dd_vlen, xlabel="target_range", ylabel="target_velocity", label="DD Spectrogram", axis_x=[0, 50], axis_y=[-10, 10], axis_z=[-15, -12], autoscale_z=True, len_key="packet_len")
        self.blocks_null_sink_hdr = blocks.null_sink(gr.sizeof_gr_complex*fft_len)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0.set_min_output_buffer(rx_sync_min_buf)
        self.blocks_message_strobe_0 = blocks.message_strobe(hdr_ack_msg, 1000)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, ((fft_len+cp_len)))
        self.blocks_delay_0.set_min_output_buffer(rx_sync_min_buf)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(((-2.0/fft_len)))
        self.analog_frequency_modulator_fc_0.set_min_output_buffer(rx_sync_min_buf)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.digital_header_payload_demux_0, 'header_data'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.digital_header_payload_demux_0, 0))
        self.connect((self.digital_header_payload_demux_0, 0), (self.blocks_null_sink_hdr, 0))
        self.connect((self.digital_header_payload_demux_0, 1), (self.ofdm_burst_sensing_rx, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.digital_ofdm_sync_sc_cfb_0, 1), (self.digital_header_payload_demux_0, 1))
        self.connect((self.ofdm_burst_sensing_rx, 0), (self.dd_spectrum_plot_0, 0))
        self.connect((self.ofdm_burst_source, 0), (self.uhd_usrp_sink_0_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.digital_ofdm_sync_sc_cfb_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_ofdm_burst_tr")
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
        self.uhd_usrp_sink_0_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0_0.set_center_freq(self.freq, 0)

    def get_config_file(self):
        return self.config_file

    def set_config_file(self, config_file):
        self.config_file = config_file
        self.set_cp_len(resolve_ofdm_fft_cp(self.config_file)[1])
        self.set_dd_vlen(resolve_dd_output_vlen(self.config_file))
        self.set_fft_len(resolve_ofdm_fft_cp(self.config_file)[0])
        self.set_hdr_ack_msg(pmt.dict_add(pmt.make_dict(), pmt.intern("frame_len"), pmt.from_long(resolve_ofdm_num_symbols(self.config_file))))
        self.set_ofdm_burst_samples(resolve_ofdm_burst_len(self.config_file))
        self.set_payload_syms(resolve_ofdm_num_symbols(self.config_file))
        self.set_samp_rate(resolve_ofdm_samp_rate(self.config_file))
        self.ofdm_burst_sensing_rx.config_file = self.config_file
        self.ofdm_burst_source.config_file = self.config_file

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_n_carriers(self.fft_len - 2)
        self.set_rx_sync_min_buf(max(65536, int((self.header_len + 8) * (self.fft_len + self.cp_len))))
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.analog_frequency_modulator_fc_0.set_sensitivity(((-2.0/self.fft_len)))
        self.blocks_delay_0.set_dly(int(((self.fft_len+self.cp_len))))

    def get_n_carriers(self):
        return self.n_carriers

    def set_n_carriers(self, n_carriers):
        self.n_carriers = n_carriers
        self.set_occupied_carriers((tuple(range(-self.n_carriers//2, 0)) + tuple(range(1, self.n_carriers//2+1)),))
        self.set_pilot_carriers((tuple(range(-self.n_carriers//2-2, -self.n_carriers//2)) + tuple(range(self.n_carriers//2+1, self.n_carriers//2+3)),))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_sync_word1(list(digital.ofdm_txrx._make_sync_word1(self.fft_len, self.occupied_carriers, self.pilot_carriers)))
        self.set_sync_word2(list(digital.ofdm_txrx._make_sync_word2(self.fft_len, self.occupied_carriers, self.pilot_carriers)))

    def get_idle_ms(self):
        return self.idle_ms

    def set_idle_ms(self, idle_ms):
        self.idle_ms = idle_ms
        self.set_burst_period_ms(self.idle_ms)
        self.ofdm_burst_source.idle_ms = self.idle_ms

    def get_header_len(self):
        return self.header_len

    def set_header_len(self, header_len):
        self.header_len = header_len
        self.set_rx_sync_min_buf(max(65536, int((self.header_len + 8) * (self.fft_len + self.cp_len))))

    def get_cp_len(self):
        return self.cp_len

    def set_cp_len(self, cp_len):
        self.cp_len = cp_len
        self.set_rx_sync_min_buf(max(65536, int((self.header_len + 8) * (self.fft_len + self.cp_len))))
        self.blocks_delay_0.set_dly(int(((self.fft_len+self.cp_len))))

    def get_tx_amp(self):
        return self.tx_amp

    def set_tx_amp(self, tx_amp):
        self.tx_amp = tx_amp
        self.ofdm_burst_source.tx_amp = self.tx_amp

    def get_time_mag_trig_level(self):
        return self.time_mag_trig_level

    def set_time_mag_trig_level(self, time_mag_trig_level):
        self.time_mag_trig_level = time_mag_trig_level

    def get_time_lead_s(self):
        return self.time_lead_s

    def set_time_lead_s(self, time_lead_s):
        self.time_lead_s = time_lead_s
        self.ofdm_burst_source.time_lead_s = self.time_lead_s

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_startup_delay_s(self):
        return self.startup_delay_s

    def set_startup_delay_s(self, startup_delay_s):
        self.startup_delay_s = startup_delay_s
        self.ofdm_burst_source.startup_delay_s = self.startup_delay_s

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0_0.set_samp_rate(self.samp_rate)

    def get_rx_sync_min_buf(self):
        return self.rx_sync_min_buf

    def set_rx_sync_min_buf(self, rx_sync_min_buf):
        self.rx_sync_min_buf = rx_sync_min_buf

    def get_payload_syms(self):
        return self.payload_syms

    def set_payload_syms(self, payload_syms):
        self.payload_syms = payload_syms

    def get_ofdm_burst_samples(self):
        return self.ofdm_burst_samples

    def set_ofdm_burst_samples(self, ofdm_burst_samples):
        self.ofdm_burst_samples = ofdm_burst_samples

    def get_hdr_ack_msg(self):
        return self.hdr_ack_msg

    def set_hdr_ack_msg(self, hdr_ack_msg):
        self.hdr_ack_msg = hdr_ack_msg
        self.blocks_message_strobe_0.set_msg(self.hdr_ack_msg)

    def get_gui_update_time_ms(self):
        return self.gui_update_time_ms

    def set_gui_update_time_ms(self, gui_update_time_ms):
        self.gui_update_time_ms = gui_update_time_ms

    def get_freq_trig_level(self):
        return self.freq_trig_level

    def set_freq_trig_level(self, freq_trig_level):
        self.freq_trig_level = freq_trig_level

    def get_frame_len_tag(self):
        return self.frame_len_tag

    def set_frame_len_tag(self, frame_len_tag):
        self.frame_len_tag = frame_len_tag

    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = device
        self.ofdm_burst_sensing_rx.device = self.device

    def get_dd_vlen(self):
        return self.dd_vlen

    def set_dd_vlen(self, dd_vlen):
        self.dd_vlen = dd_vlen

    def get_corr_threshold(self):
        return self.corr_threshold

    def set_corr_threshold(self, corr_threshold):
        self.corr_threshold = corr_threshold
        self.digital_ofdm_sync_sc_cfb_0.set_threshold(self.corr_threshold)

    def get_burst_period_ms(self):
        return self.burst_period_ms

    def set_burst_period_ms(self, burst_period_ms):
        self.burst_period_ms = burst_period_ms

    def get_TX_gain(self):
        return self.TX_gain

    def set_TX_gain(self, TX_gain):
        self.TX_gain = TX_gain
        self.uhd_usrp_sink_0_0.set_gain(self.TX_gain, 0)

    def get_RX_gain(self):
        return self.RX_gain

    def set_RX_gain(self, RX_gain):
        self.RX_gain = RX_gain
        self.uhd_usrp_source_0_0.set_gain(self.RX_gain, 0)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--address", dest="address", type=str, default="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2",
        help="Set UHD dev args [default=%(default)r]")
    parser.add_argument(
        "-f", "--freq", dest="freq", type=eng_float, default=eng_notation.num_to_str(float(6.0e9)),
        help="Set Default Frequency [default=%(default)r]")
    return parser


def main(top_block_cls=usrp_ofdm_burst_tr, options=None):
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
