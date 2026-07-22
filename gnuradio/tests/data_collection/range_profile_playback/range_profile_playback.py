#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Range Profile Playback
# Description: Replay linear-power range profiles recorded by range_profile_collection (file_meta .dat/.meta) and display as dB range spectrum.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import sip
import threading



class range_profile_playback(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Range Profile Playback", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Range Profile Playback")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "range_profile_playback")

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
        self.subcarrier_spacing = subcarrier_spacing = 60e3
        self.fft_len = fft_len = 2048
        self.samp_rate = samp_rate = int(fft_len * subcarrier_spacing)
        self.zeropadding_fac = zeropadding_fac = 2
        self.transpose_len = transpose_len = 4
        self.R_max = R_max = 3e8/2/samp_rate*fft_len
        self.record_file_path = record_file_path = "/home/caict/Desktop/isac/gnuradio/tests/data_collection/range_profile_collection/dataset/run_001/range_profiles"
        self.range_bin_step = range_bin_step = R_max/(fft_len*zeropadding_fac)
        self.frame_rate_hz = frame_rate_hz = samp_rate / (transpose_len * (fft_len + fft_len // 4))

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            (fft_len*zeropadding_fac),
            0,
            range_bin_step,
            "Range",
            "Power (dB)",
            "Range Profile Playback",
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
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*(fft_len*zeropadding_fac), frame_rate_hz,True)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, (fft_len*zeropadding_fac), 0)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*(fft_len*zeropadding_fac), record_file_path, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared((fft_len*zeropadding_fac))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_complex_to_mag_squared_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "range_profile_playback")
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
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))
        self.set_samp_rate(int(self.fft_len * self.subcarrier_spacing))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_R_max(3e8/2/self.samp_rate*self.fft_len)
        self.set_frame_rate_hz(self.samp_rate / (self.transpose_len * (self.fft_len + self.fft_len // 4)))

    def get_zeropadding_fac(self):
        return self.zeropadding_fac

    def set_zeropadding_fac(self, zeropadding_fac):
        self.zeropadding_fac = zeropadding_fac
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))

    def get_transpose_len(self):
        return self.transpose_len

    def set_transpose_len(self, transpose_len):
        self.transpose_len = transpose_len
        self.set_frame_rate_hz(self.samp_rate / (self.transpose_len * (self.fft_len + self.fft_len // 4)))

    def get_R_max(self):
        return self.R_max

    def set_R_max(self, R_max):
        self.R_max = R_max
        self.set_range_bin_step(self.R_max/(self.fft_len*self.zeropadding_fac))

    def get_record_file_path(self):
        return self.record_file_path

    def set_record_file_path(self, record_file_path):
        self.record_file_path = record_file_path
        self.blocks_file_source_0.open(self.record_file_path, True)

    def get_range_bin_step(self):
        return self.range_bin_step

    def set_range_bin_step(self, range_bin_step):
        self.range_bin_step = range_bin_step
        self.qtgui_vector_sink_f_0.set_x_axis(0, self.range_bin_step)

    def get_frame_rate_hz(self):
        return self.frame_rate_hz

    def set_frame_rate_hz(self, frame_rate_hz):
        self.frame_rate_hz = frame_rate_hz
        self.blocks_throttle_0.set_sample_rate(self.frame_rate_hz)




def main(top_block_cls=range_profile_playback, options=None):

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
