'''
Driver program to provide CLI and start up an FX correlator.
'''

import argparse
import asyncio
import multiprocessing
import numpy as np
import sys
import time

from rtlsdr import RtlSdr

import effex as fx


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # CONSTANTS 
    # -------------------------------------------------------------------------
    d_len = int(5e8 // (2**18 * np.dtype(np.complex128).itemsize) // 2) # sized for 4GB RAM on NVIDIA Jetson Nano
    streaming_fudge_factor = 1. # sec, allow some time for streaming subprocesses to get to starting line

    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='GPU FX correlator accelerated with NVIDIA cuSignal.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--time',       '-T', default=1, type=float, dest='run_time',
        help='(sec) Total amount of time to run correlator.')
    parser.add_argument('--bandwidth',  '-B', default=2.4e6, type=float, dest='rate',
        help='(Hz) Receiver bandwidth. For RTL-SDR dongles, this determines sample rate. Applied to both channels.')
    parser.add_argument('--frequency',  '-F', default=1.4204e9, type=float, dest='fc',
        help='(Hz) Receiver center tuning frequency. Applied to both channels.')
    parser.add_argument('--num_samp',   '-N', default=2**18, type=int, dest='num_samp',
        help='(int) Number of samples to read from SDR on each call. Large powers of 2 less than 2**18 are recommended for RTL-SDRs.')
    parser.add_argument('--resolution', '-R', default=2**12, type=int, dest='nfft',
        help='(int) Number of FFT bins to use in processing and plotting.')
    parser.add_argument('--gain',       '-G', default=49.6, type=float, dest='gain',
        help='(dB) Tuner gain in Decibels. Tuner gain has an impact on receiver sensitivity and may affect clock stability due to heat generation. Applied to both channels.')
    parser.add_argument('--mode',       '-M', default='spectrum', type=str, choices=['continuum', 'spectrum'], dest='mode', 
        help='(str) Choose one: continuum mode estimates visibility amplitude over time, throwing away phase and frequency information. Spectrum mode keeps complex visibilities. Affects data memory usage, visualization speed, and output file size.')
    parser.add_argument('--omit_plot',  '-P', default=False, type=bool, dest='omit_plot', 
        help='If True, skip post-processing step using matplotlib to visualize recorded data. This may help avoid memory usage problems on low-memory systems. Raw data will still be recorded to a file for further post-processing.')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # CORRELATOR INIT
    # -------------------------------------------------------------------------
    #cor = fx.Correlator()

    # -------------------------------------------------------------------------
    # SDR INIT
    # -------------------------------------------------------------------------
    # Dithering depends on evanmayer's fork of roger-'s pyrtlsdr and keenerd's
    # experimental fork of librtlsdr
    sdr_0 = RtlSdr(device_index=0, dithering_enabled=False)
    sdr_1 = RtlSdr(device_index=1, dithering_enabled=False)
    sdr_0.rs = args.rate
    sdr_1.rs = args.rate
    sdr_0.fc = args.fc
    sdr_1.fc = args.fc
    sdr_0.gain = args.gain
    sdr_1.gain = args.gain
    
    # -------------------------------------------------------------------------
    # CPU & GPU MEMORY SETUP
    # -------------------------------------------------------------------------
    iq_0 = np.array([], dtype=np.complex128)
    iq_1 = np.array([], dtype=np.complex128)

    # Store sample chunks in 2 deques
    buf_0 = multiprocessing.Queue(d_len)
    buf_1 = multiprocessing.Queue(d_len)

    # -------------------------------------------------------------------------
    # RUN 
    # -------------------------------------------------------------------------
    start_time = time.time() + streaming_fudge_factor
    print('Interferometry begins at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(start_time))))
    # IQ source processes
    # ECM: FIXME:
    # streaming() is an async function, so this will throw a warning about 
    # not awaiting it, but of course it's being run by asyncio.run, just not
    # here. There might be another way to do this, but this works for now.
    proc_0 = fx.streaming(sdr_0, buf_0, args.num_samp, start_time, args.run_time)
    producer_0 = multiprocessing.Process(target=asyncio.run, args=(proc_0,))
    proc_1 = fx.streaming(sdr_1, buf_1, args.num_samp, start_time, args.run_time)
    producer_1 = multiprocessing.Process(target=asyncio.run, args=(proc_1,))
    producer_0.start()
    producer_1.start()
    
    # IQ sink
    raw_output = fx.process_iq(buf_0, buf_1,
                              args.num_samp,
                              args.nfft,
                              args.rate,
                              args.fc,
                              start_time, args.run_time,
                              mode=args.mode)

    print('IQ processing complete, buffers drained.')

    sdr_0.close()
    sdr_1.close()
    print('SDRs closed.')

    fx.post_process(raw_output, args.rate, args.fc, args.nfft, args.num_samp, args.mode, args.omit_plot)

