#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import soapy




class test(gr.top_block):

    def __init__(self, CCSDS_sync_word="00011010110011111111110000011101", Packet_Length=241):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

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
        self.Tx_freq = Tx_freq = 917e6
        self.Rx_freq = Rx_freq = 915e6

        ##################################################
        # Blocks
        ##################################################

        self.soapy_bladerf_source_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_source_0 = soapy.source(dev, "fc32", 1, 'driver=bladerf,buffer_count=32, buffer_size=16384',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_source_0.set_sample_rate(0, samp_rate*4)
        self.soapy_bladerf_source_0.set_bandwidth(0, samp_rate*4)
        self.soapy_bladerf_source_0.set_frequency(0, Rx_freq)
        self.soapy_bladerf_source_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_source_0.set_gain(0, min(max(45.0, -1.0), 60.0))
        self.soapy_bladerf_sink_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_sink_0 = soapy.sink(dev, "fc32", 1, 'driver=bladerf,buffer_count=32, buffer_size=16384',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_sink_0.set_sample_rate(0, samp_rate*4)
        self.soapy_bladerf_sink_0.set_bandwidth(0, samp_rate*4)
        self.soapy_bladerf_sink_0.set_frequency(0, Tx_freq)
        self.soapy_bladerf_sink_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_sink_0.set_gain(0, min(max(55.0, 17.0), 73.0))
        self.pdu_tagged_stream_to_pdu_1_0_1_0 = pdu.tagged_stream_to_pdu(gr.types.byte_t, "packet_len")
        self.pdu_tagged_stream_to_pdu_1_0_1 = pdu.tagged_stream_to_pdu(gr.types.byte_t, "packet_len")
        self.pdu_pdu_to_tagged_stream_0_0_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, "packet_len")
        self.pdu_pdu_to_tagged_stream_0_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, "packet_len")
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, "packet_len")
        self.digital_protocol_formatter_async_0 = digital.protocol_formatter_async(digital.header_format_default(CCSDS_sync_word, 0))
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(4, 0.0628, firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 11*sps*nfilts), 32, 16, 1.5, 1)
        self.digital_diff_decoder_bb_0 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.digital_crc32_bb_0_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_crc32_bb_0 = digital.crc32_bb(True, "packet_len", True)
        self.digital_costas_loop_cc_2_0 = digital.costas_loop_cc(0.00628, 2, False)
        self.digital_correlate_access_code_xx_ts_1_0 = digital.correlate_access_code_bb_ts(CCSDS_sync_word,
          0, "packet_len")
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=variable_constellation_1,
            differential=True,
            samples_per_symbol=4,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_decoder_cb_1_0 = digital.constellation_decoder_cb(variable_constellation_1)
        self.blocks_unpack_k_bits_bb_0_0_0 = blocks.unpack_k_bits_bb(1)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_char*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_tagged_stream_mux_0_0 = blocks.tagged_stream_mux(gr.sizeof_char*1, "packet_len", 0)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, Packet_Length, "packet_len")
        self.blocks_repack_bits_bb_0_0_0_0_0 = blocks.repack_bits_bb(1, 8, "packet_len", False, gr.GR_MSB_FIRST)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(.2)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/user/satellite_2/satellite2/tx_input.txt', False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0_0_0 = blocks.file_sink(gr.sizeof_char*1, '/home/user/satellite_2/satellite2/sat1.txt', False)
        self.blocks_file_sink_0_0_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.digital_protocol_formatter_async_0, 'payload'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.digital_protocol_formatter_async_0, 'header'), (self.pdu_pdu_to_tagged_stream_0_0, 'pdus'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_1_0_1, 'pdus'), (self.pdu_pdu_to_tagged_stream_0_0_0, 'pdus'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_1_0_1_0, 'pdus'), (self.digital_protocol_formatter_async_0, 'in'))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.soapy_bladerf_sink_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0_0_0, 0), (self.digital_crc32_bb_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.pdu_tagged_stream_to_pdu_1_0_1, 0))
        self.connect((self.blocks_tagged_stream_mux_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0_0_0, 0), (self.digital_correlate_access_code_xx_ts_1_0, 0))
        self.connect((self.digital_constellation_decoder_cb_1_0, 0), (self.digital_diff_decoder_bb_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_correlate_access_code_xx_ts_1_0, 0), (self.blocks_repack_bits_bb_0_0_0_0_0, 0))
        self.connect((self.digital_costas_loop_cc_2_0, 0), (self.digital_constellation_decoder_cb_1_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.blocks_file_sink_0_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.pdu_tagged_stream_to_pdu_1_0_1_0, 0))
        self.connect((self.digital_diff_decoder_bb_0, 0), (self.blocks_unpack_k_bits_bb_0_0_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.digital_costas_loop_cc_2_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_tagged_stream_mux_0_0, 1))
        self.connect((self.pdu_pdu_to_tagged_stream_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0_0_0, 0), (self.digital_crc32_bb_0_0, 0))
        self.connect((self.soapy_bladerf_source_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))


    def get_CCSDS_sync_word(self):
        return self.CCSDS_sync_word

    def set_CCSDS_sync_word(self, CCSDS_sync_word):
        self.CCSDS_sync_word = CCSDS_sync_word

    def get_Packet_Length(self):
        return self.Packet_Length

    def set_Packet_Length(self, Packet_Length):
        self.Packet_Length = Packet_Length
        self.blocks_stream_to_tagged_stream_0.set_packet_len(self.Packet_Length)
        self.blocks_stream_to_tagged_stream_0.set_packet_len_pmt(self.Packet_Length)

    def get_variable_constellation_1(self):
        return self.variable_constellation_1

    def set_variable_constellation_1(self, variable_constellation_1):
        self.variable_constellation_1 = variable_constellation_1
        self.digital_constellation_decoder_cb_1_0.set_constellation(self.variable_constellation_1)

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.digital_pfb_clock_sync_xxx_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.soapy_bladerf_sink_0.set_sample_rate(0, self.samp_rate*4)
        self.soapy_bladerf_sink_0.set_bandwidth(0, self.samp_rate*4)
        self.soapy_bladerf_source_0.set_sample_rate(0, self.samp_rate*4)
        self.soapy_bladerf_source_0.set_bandwidth(0, self.samp_rate*4)

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.digital_pfb_clock_sync_xxx_0.update_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 11*self.sps*self.nfilts))

    def get_Tx_freq(self):
        return self.Tx_freq

    def set_Tx_freq(self, Tx_freq):
        self.Tx_freq = Tx_freq
        self.soapy_bladerf_sink_0.set_frequency(0, self.Tx_freq)

    def get_Rx_freq(self):
        return self.Rx_freq

    def set_Rx_freq(self, Rx_freq):
        self.Rx_freq = Rx_freq
        self.soapy_bladerf_source_0.set_frequency(0, self.Rx_freq)



def argument_parser():
    parser = ArgumentParser()
    return parser


def main(top_block_cls=test, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
