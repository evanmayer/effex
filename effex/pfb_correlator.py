'''
Driver program to provide CLI and start up an FX correlator.
'''

import argparse
import effex as fx


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='GPU FX correlator accelerated with NVIDIA cuSignal.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--time',       '-T', default=1, type=float, dest='run_time',
        help='(sec) Total amount of time to run correlator.')
    parser.add_argument('--bandwidth',  '-B', default=2.4e6, type=float, dest='bandwidth',
        help='(Hz) Receiver bandwidth. For RTL-SDR dongles, this determines sample rate. Applied to both channels.')
    parser.add_argument('--frequency',  '-F', default=1.4204e9, type=float, dest='fc',
        help='(Hz) Receiver center tuning frequency. Applied to both channels.')
    parser.add_argument('--num_samp',   '-N', default=2**18, type=int, dest='num_samp',
        help='(int) Number of samples to read from SDR on each call. Large powers of 2 less than 2**18 are recommended for RTL-SDRs.')
    parser.add_argument('--resolution', '-R', default=2**12, type=int, dest='nfft',
        help='(int) Number of FFT bins to use in processing and plotting.')
    parser.add_argument('--gain',       '-G', default=49.6, type=float, dest='gain',
        help='(dB) Tuner gain in Decibels. Tuner gain has an impact on receiver sensitivity and may affect clock stability due to heat generation. Applied to both channels.')
    parser.add_argument('--mode',       '-M', default='spectrum', type=str, choices=['continuum', 'spectrum', 'test'], dest='mode', 
        help='(str) Choose one: continuum mode estimates visibility amplitude over time, throwing away phase and frequency information. Spectrum mode keeps complex visibilities. Affects data memory usage, visualization speed, and output file size.')
    parser.add_argument('--omit_plot',  '-P', default=False, type=bool, dest='omit_plot', 
        help='If True, skip post-processing step using matplotlib to visualize recorded data. This may help avoid memory usage problems on low-memory systems. Raw data will still be recorded to a file for further post-processing.')
    parser.add_argument('--loglevel', '-L', default='INFO', type=str, choices=['INFO', 'WARNING', 'DEBUG', 'ERROR', 'CRITICAL'], dest='loglevel',
        help='Python logging module loglevel.')

    args = parser.parse_args()

    cor = fx.Correlator(run_time=args.run_time,
                        bandwidth=args.bandwidth,
                        frequency=args.fc,
                        num_samp=args.num_samp,
                        nbins=args.nfft,
                        gain=args.gain,
                        mode=args.mode,
                        loglevel=args.loglevel)
    cor.run_state_machine()
    cor.close()
    cor.post_process(cor.vis_out, args.bandwidth, args.fc, args.nfft, args.num_samp, args.mode, args.omit_plot)

