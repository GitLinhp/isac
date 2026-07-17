#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Usrp Echotimer Sync Pulse
# Description: USRP Sink/Source sync pulse (no echotimer). RX packet_len tags via burst_iq_tag_rx EPY.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import radar
import sip
import threading



class usrp_echotimer_sync_pulse(gr.top_block, Qt.QWidget):

    def __init__(self, address0="type=x4xx,mgmt_addr=192.168.1.101,addr=192.168.11.2", address1="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2"):
        gr.top_block.__init__(self, "Usrp Echotimer Sync Pulse", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Usrp Echotimer Sync Pulse")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_echotimer_sync_pulse")

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
        self.wait_samp = wait_samp = 100,100,100,100
        self.send_samp = send_samp = 100,400,300
        self.packet_len = packet_len = sum(wait_samp)+sum(send_samp)
        self.wait_to_start = wait_to_start = 0.03
        self.uhd_dev_args = uhd_dev_args = "type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2"
        self.samp_rate = samp_rate = 5e6
        self.num_delay_samp = num_delay_samp = 9
        self.num_corr = num_corr = packet_len
        self.min_out_buf_val = min_out_buf_val = packet_len*2
        self.freq = freq = 6000000000
        self.factor = factor = 20
        self.TX_gain = TX_gain = 10
        self.RX_gain = RX_gain = 10

        ##################################################
        # Blocks
        ##################################################

        self._num_delay_samp_range = qtgui.Range(0, packet_len, 1, 9, 200)
        self._num_delay_samp_win = qtgui.RangeWidget(self._num_delay_samp_range, self.set_num_delay_samp, "Number of delayed samples", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._num_delay_samp_win)
        self._num_corr_range = qtgui.Range(0, packet_len, 1, packet_len, 200)
        self._num_corr_win = qtgui.RangeWidget(self._num_corr_range, self.set_num_corr, "Number of cross correlations", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._num_corr_win)
        self._factor_range = qtgui.Range(0, 20, 1, 20, 200)
        self._factor_win = qtgui.RangeWidget(self._factor_range, self.set_factor, "'factor'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._factor_win)
        self._TX_gain_range = qtgui.Range(0, 100, 1, 10, 200)
        self._TX_gain_win = qtgui.RangeWidget(self._TX_gain_range, self.set_TX_gain, "TX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._TX_gain_win)
        self._RX_gain_range = qtgui.Range(0, 100, 1, 10, 200)
        self._RX_gain_win = qtgui.RangeWidget(self._RX_gain_range, self.set_RX_gain, "RX Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._RX_gain_win)
        self.radar_usrp_echotimer_cc_0_0 = radar.usrp_echotimer_cc(int(samp_rate), freq, int(num_delay_samp), address0, 0, '', 'internal', 'internal', 'TX/RX', TX_gain, 0.1, wait_to_start, 0, address0, 0, '', 'internal', 'internal', 'RX1', RX_gain, 0.1, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0_0.set_min_output_buffer(min_out_buf_val)
        self.radar_usrp_echotimer_cc_0 = radar.usrp_echotimer_cc(int(samp_rate), freq, int(num_delay_samp), address1, 0, '', 'internal', 'internal', 'TX/RX', TX_gain, 0.1, wait_to_start, 0, address1, 0, '', 'internal', 'internal', 'RX1', RX_gain, 0.1, wait_to_start, 0, "packet_len")
        self.radar_usrp_echotimer_cc_0.set_min_output_buffer(min_out_buf_val)
        self.radar_signal_generator_sync_pulse_c_0_0 = radar.signal_generator_sync_pulse_c(packet_len, send_samp, wait_samp, 0.5, "packet_len")
        self.radar_signal_generator_sync_pulse_c_0_0.set_min_output_buffer(min_out_buf_val)
        self.radar_signal_generator_sync_pulse_c_0 = radar.signal_generator_sync_pulse_c(packet_len, send_samp, wait_samp, 0.5, "packet_len")
        self.radar_signal_generator_sync_pulse_c_0.set_min_output_buffer(min_out_buf_val)
        self.radar_print_results_0_0 = radar.print_results(False, "")
        self.radar_print_results_0 = radar.print_results(False, "")
        self.radar_estimator_sync_pulse_c_0_0 = radar.estimator_sync_pulse_c(int(num_corr), "packet_len")
        self.radar_estimator_sync_pulse_c_0 = radar.estimator_sync_pulse_c(int(num_corr), "packet_len")
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            packet_len, #size
            samp_rate, #samp_rate
            'QT GUI Plot', #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0_0.enable_grid(False)
        self.qtgui_time_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
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
                self.qtgui_time_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            packet_len, #size
            samp_rate, #samp_rate
            'QT GUI Plot', #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
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
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(factor)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(factor)
        self.blocks_complex_to_mag_1_0 = blocks.complex_to_mag(1)
        self.blocks_complex_to_mag_1 = blocks.complex_to_mag(1)
        self.blocks_complex_to_mag_0_0 = blocks.complex_to_mag(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.radar_estimator_sync_pulse_c_0, 'Msg out'), (self.radar_print_results_0, 'Msg in'))
        self.msg_connect((self.radar_estimator_sync_pulse_c_0_0, 'Msg out'), (self.radar_print_results_0_0, 'Msg in'))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_complex_to_mag_0_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.blocks_complex_to_mag_1, 0), (self.qtgui_time_sink_x_0, 1))
        self.connect((self.blocks_complex_to_mag_1_0, 0), (self.qtgui_time_sink_x_0_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_complex_to_mag_1, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.blocks_complex_to_mag_1_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0, 0), (self.radar_estimator_sync_pulse_c_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0, 0), (self.radar_usrp_echotimer_cc_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0_0, 0), (self.blocks_complex_to_mag_0_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0_0, 0), (self.radar_estimator_sync_pulse_c_0_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0_0, 0), (self.radar_usrp_echotimer_cc_0_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0, 0), (self.radar_estimator_sync_pulse_c_0, 1))
        self.connect((self.radar_usrp_echotimer_cc_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.radar_usrp_echotimer_cc_0_0, 0), (self.radar_estimator_sync_pulse_c_0_0, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_echotimer_sync_pulse")
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

    def get_wait_samp(self):
        return self.wait_samp

    def set_wait_samp(self, wait_samp):
        self.wait_samp = wait_samp
        self.set_packet_len(sum(self.wait_samp)+sum(self.send_samp))

    def get_send_samp(self):
        return self.send_samp

    def set_send_samp(self, send_samp):
        self.send_samp = send_samp
        self.set_packet_len(sum(self.wait_samp)+sum(self.send_samp))

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_min_out_buf_val(self.packet_len*2)
        self.set_num_corr(self.packet_len)

    def get_wait_to_start(self):
        return self.wait_to_start

    def set_wait_to_start(self, wait_to_start):
        self.wait_to_start = wait_to_start

    def get_uhd_dev_args(self):
        return self.uhd_dev_args

    def set_uhd_dev_args(self, uhd_dev_args):
        self.uhd_dev_args = uhd_dev_args

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)

    def get_num_delay_samp(self):
        return self.num_delay_samp

    def set_num_delay_samp(self, num_delay_samp):
        self.num_delay_samp = num_delay_samp
        self.radar_usrp_echotimer_cc_0.set_num_delay_samps(int(self.num_delay_samp))
        self.radar_usrp_echotimer_cc_0_0.set_num_delay_samps(int(self.num_delay_samp))

    def get_num_corr(self):
        return self.num_corr

    def set_num_corr(self, num_corr):
        self.num_corr = num_corr
        self.radar_estimator_sync_pulse_c_0.set_num_xcorr(int(self.num_corr))
        self.radar_estimator_sync_pulse_c_0_0.set_num_xcorr(int(self.num_corr))

    def get_min_out_buf_val(self):
        return self.min_out_buf_val

    def set_min_out_buf_val(self, min_out_buf_val):
        self.min_out_buf_val = min_out_buf_val

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_factor(self):
        return self.factor

    def set_factor(self, factor):
        self.factor = factor
        self.blocks_multiply_const_vxx_0.set_k(self.factor)
        self.blocks_multiply_const_vxx_0_0.set_k(self.factor)

    def get_TX_gain(self):
        return self.TX_gain

    def set_TX_gain(self, TX_gain):
        self.TX_gain = TX_gain
        self.radar_usrp_echotimer_cc_0.set_tx_gain(self.TX_gain)
        self.radar_usrp_echotimer_cc_0_0.set_tx_gain(self.TX_gain)

    def get_RX_gain(self):
        return self.RX_gain

    def set_RX_gain(self, RX_gain):
        self.RX_gain = RX_gain
        self.radar_usrp_echotimer_cc_0.set_rx_gain(self.RX_gain)
        self.radar_usrp_echotimer_cc_0_0.set_rx_gain(self.RX_gain)



def argument_parser():
    description = 'USRP Sink/Source sync pulse (no echotimer). RX packet_len tags via burst_iq_tag_rx EPY.'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--address0", dest="address0", type=str, default="type=x4xx,mgmt_addr=192.168.1.101,addr=192.168.11.2",
        help="Set UHD dev args [default=%(default)r]")
    parser.add_argument(
        "--address1", dest="address1", type=str, default="type=x4xx,mgmt_addr=192.168.1.100,addr=192.168.10.2",
        help="Set UHD dev args [default=%(default)r]")
    return parser


def main(top_block_cls=usrp_echotimer_sync_pulse, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(address0=options.address0, address1=options.address1)

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
