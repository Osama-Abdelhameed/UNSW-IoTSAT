#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Ground_Station
# Author: sat1
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
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
from gnuradio import soapy



class Ground_Station(gr.top_block, Qt.QWidget):

    def __init__(self, CCSDS_sync_word="00011010110011111111110000011101", Packet_Length=241):
        gr.top_block.__init__(self, "Ground_Station", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Ground_Station")
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

        self.settings = Qt.QSettings("GNU Radio", "Ground_Station")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Parameters
        ##################################################
        self.CCSDS_sync_word = CCSDS_sync_word
        self.Packet_Length = Packet_Length

        ##################################################
        # Variables
        ##################################################
        self.variable_constellation_1 = variable_constellation_1 = digital.constellation_bpsk().base()
        self.variable_constellation_1.set_npwr(1.0)
        self.sps = sps = 4
        self.samp_rate = samp_rate = 50e3
        self.nfilts = nfilts = 32
        self.Rx_freq_2 = Rx_freq_2 = 917e6
        self.Rx_freq_1 = Rx_freq_1 = 915e6

        ##################################################
        # Blocks
        ##################################################

        self.soapy_bladerf_source_0_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_source_0_0 = soapy.source(dev, "fc32", 1, 'driver=bladerf,buffer_count=32, buffer_size=16384',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_source_0_0.set_sample_rate(0, samp_rate*4)
        self.soapy_bladerf_source_0_0.set_bandwidth(0, samp_rate*4)
        self.soapy_bladerf_source_0_0.set_frequency(0, Rx_freq_2)
        self.soapy_bladerf_source_0_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_source_0_0.set_gain(0, min(max(45.0, -1.0), 60.0))
        self.soapy_bladerf_source_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_source_0 = soapy.source(dev, "fc32", 1, 'driver=bladerf,buffer_count=32, buffer_size=16384',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_source_0.set_sample_rate(0, samp_rate*4)
        self.soapy_bladerf_source_0.set_bandwidth(0, samp_rate*4)
        self.soapy_bladerf_source_0.set_frequency(0, Rx_freq_1)
        self.soapy_bladerf_source_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_source_0.set_gain(0, min(max(45.0, -1.0), 60.0))
        self.digital_pfb_clock_sync_xxx_0_0 = digital.pfb_clock_sync_ccf(4, 0.0628, firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 11*sps*nfilts), 32, 16, 1.5, 1)
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(4, 0.0628, firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 11*sps*nfilts), 32, 16, 1.5, 1)
        self.digital_diff_decoder_bb_0_0 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.digital_diff_decoder_bb_0 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.digital_crc32_bb_0_0 = digital.crc32_bb(True, "packet_len", True)
        self.digital_crc32_bb_0 = digital.crc32_bb(True, "packet_len", True)
        self.digital_costas_loop_cc_2_0_0 = digital.costas_loop_cc(0.00628, 2, False)
        self.digital_costas_loop_cc_2_0 = digital.costas_loop_cc(0.00628, 2, False)
        self.digital_correlate_access_code_xx_ts_1_0_0 = digital.correlate_access_code_bb_ts(CCSDS_sync_word,
          0, "packet_len")
        self.digital_correlate_access_code_xx_ts_1_0 = digital.correlate_access_code_bb_ts(CCSDS_sync_word,
          0, "packet_len")
        self.digital_constellation_decoder_cb_1_0_0 = digital.constellation_decoder_cb(variable_constellation_1)
        self.digital_constellation_decoder_cb_1_0 = digital.constellation_decoder_cb(variable_constellation_1)
        self.blocks_unpack_k_bits_bb_0_0_0_0 = blocks.unpack_k_bits_bb(1)
        self.blocks_unpack_k_bits_bb_0_0_0 = blocks.unpack_k_bits_bb(1)
        self.blocks_repack_bits_bb_0_0_0_0_0_0 = blocks.repack_bits_bb(1, 8, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_0_0_0_0_0 = blocks.repack_bits_bb(1, 8, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_file_sink_0_0_0_0 = blocks.file_sink(gr.sizeof_char*1, '/home/sat1/satellite_1/satellite1/sat2.txt', False)
        self.blocks_file_sink_0_0_0_0.set_unbuffered(False)
        self.blocks_file_sink_0_0_0 = blocks.file_sink(gr.sizeof_char*1, '/home/user/satellite_2/satellite2/sat1.txt', False)
        self.blocks_file_sink_0_0_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_repack_bits_bb_0_0_0_0_0, 0), (self.digital_crc32_bb_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0_0_0_0, 0), (self.digital_crc32_bb_0_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0_0_0, 0), (self.digital_correlate_access_code_xx_ts_1_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0_0_0_0, 0), (self.digital_correlate_access_code_xx_ts_1_0_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1_0, 0), (self.digital_diff_decoder_bb_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1_0_0, 0), (self.digital_diff_decoder_bb_0_0, 0))
        self.connect((self.digital_correlate_access_code_xx_ts_1_0, 0), (self.blocks_repack_bits_bb_0_0_0_0_0, 0))
        self.connect((self.digital_correlate_access_code_xx_ts_1_0_0, 0), (self.blocks_repack_bits_bb_0_0_0_0_0_0, 0))
        self.connect((self.digital_costas_loop_cc_2_0, 0), (self.digital_constellation_decoder_cb_1_0, 0))
        self.connect((self.digital_costas_loop_cc_2_0_0, 0), (self.digital_constellation_decoder_cb_1_0_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.blocks_file_sink_0_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.blocks_file_sink_0_0_0_0, 0))
        self.connect((self.digital_diff_decoder_bb_0, 0), (self.blocks_unpack_k_bits_bb_0_0_0, 0))
        self.connect((self.digital_diff_decoder_bb_0_0, 0), (self.blocks_unpack_k_bits_bb_0_0_0_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.digital_costas_loop_cc_2_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_0, 0), (self.digital_costas_loop_cc_2_0_0, 0))
        self.connect((self.soapy_bladerf_source_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.soapy_bladerf_source_0_0, 0), (self.digital_pfb_clock_sync_xxx_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Ground_Station")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_CCSDS_sync_word(self):
        return self.CCSDS_sync_word

    def set_CCSDS_sync_word(self, CCSDS_sync_word):
        self.CCSDS_sync_word = CCSDS_sync_word

    def get_Packet_Length(self):
        return self.Packet_Length

    def set_Packet_Length(self, Packet_Length):
        self.Packet_Length = Packet_Length

    def get_variable_constellation_1(self):
        return self.variable_constellation_1

    def set_variable_constellation_1(self, variable_constellation_1):
        self.variable_constellation_1 = variable_constellation_1
        self.digital_constellation_decoder_cb_1_0.set_constellation(self.variable_constellation_1)
        self.digital_constellation_decoder_cb_1_0_0.set_constellation(self.variable_constellation_1)

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.digital_pfb_clock_sync_xxx_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))
        self.digital_pfb_clock_sync_xxx_0_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.soapy_bladerf_source_0.set_sample_rate(0, self.samp_rate*4)
        self.soapy_bladerf_source_0.set_bandwidth(0, self.samp_rate*4)
        self.soapy_bladerf_source_0_0.set_sample_rate(0, self.samp_rate*4)
        self.soapy_bladerf_source_0_0.set_bandwidth(0, self.samp_rate*4)

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.digital_pfb_clock_sync_xxx_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))
        self.digital_pfb_clock_sync_xxx_0_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_Rx_freq_2(self):
        return self.Rx_freq_2

    def set_Rx_freq_2(self, Rx_freq_2):
        self.Rx_freq_2 = Rx_freq_2
        self.soapy_bladerf_source_0_0.set_frequency(0, self.Rx_freq_2)

    def get_Rx_freq_1(self):
        return self.Rx_freq_1

    def set_Rx_freq_1(self, Rx_freq_1):
        self.Rx_freq_1 = Rx_freq_1
        self.soapy_bladerf_source_0.set_frequency(0, self.Rx_freq_1)



def argument_parser():
    parser = ArgumentParser()
    return parser


def main(top_block_cls=Ground_Station, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

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
