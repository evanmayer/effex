import argparse
import numpy as np
import matplotlib.pyplot as plt


def visualize(visibilities, rate, fc, nfft, mode, test_delay_sweep_step=0):
    '''Helper that handles plotting 1D continuum data or 2D spectrum
    data with respect to time.'''

    print('Plotting data...')

    amp = np.sqrt(np.real(visibilities * np.conj(visibilities)))
    phase = np.angle(visibilities)
    real_part = np.real(visibilities)
    imag_part = np.imag(visibilities)

    if mode in ['continuum', 'test']:
        sharey = 'none'
    else:
        sharey = 'all'

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey=sharey)

    if mode in ['continuum', 'test']:
        # Convert x axis from SDR samples to time delay
        samples = np.arange(0, len(amp))
        if test_delay_sweep_step:
            delay_ns = samples * test_delay_sweep_step * 1e9
            x = delay_ns
            xlabel = 'Delay (ns)'
        else:
            x = samples
            xlabel = 'Sample #'

        im00 = axes[0][0].plot(x, amp)
        axes[0][0].set_xlabel(xlabel)
        axes[0][0].set_ylabel('Amplitude (uncalibrated)')
        axes[0][0].set_title('Complex Cross-Correlation Amplitude')

        im01 = axes[0][1].plot(x, real_part, label='real part')
        im01 = axes[0][1].plot(x, imag_part, alpha=0.5, label='imag_part')
        axes[0][1].set_xlabel(xlabel)
        axes[0][1].set_ylabel('Amplitude')
        axes[0][1].set_title('Complex Cross-Correlation Real & Imag')
        axes[0][1].legend(loc='best')
        
        im10 = axes[1][0].plot(x, phase)
        axes[1][0].set_xlabel(xlabel)
        axes[1][0].set_ylabel('Phase')
        axes[1][0].set_title('Complex Cross-Correlation Phase')

        im11 = axes[1][1].plot(x, imag_part, label='imag_part')
        axes[1][1].set_xlabel(xlabel)
        axes[1][1].set_ylabel('Amplitude')
        axes[1][1].set_title('Complex Cross-Correlation Imag')
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate)) + fc
        num_spectra = np.array(range(visibilities.shape[0]))
        # Limit the time resolution of plots to make displaying more snappy.
        stride = 1
        max_rows = 50
        if num_spectra.max() > max_rows:
            stride = num_spectra.max() // max_rows

        X,Y = np.meshgrid(freqs, num_spectra[::stride])

        im00 = axes[0][0].pcolormesh(X, Y, amp[::stride,:], shading='auto', cmap='viridis')
        axes[0][0].set_xlabel('Frequency (Hz)')
        axes[0][0].set_ylabel('Sample #')
        axes[0][0].set_title('Complex Cross-Correlation Amplitude')

        im01 = axes[0][1].pcolormesh(X, Y, real_part[::stride,:], shading='auto', cmap='viridis')
        axes[0][1].set_xlabel('Frequency (Hz)')
        axes[0][1].set_ylabel('Sample #')
        axes[0][1].set_title('Real part of XCorrs')

        im10 = axes[1][0].pcolormesh(X, Y, phase[::stride,:], shading='auto', cmap='viridis')
        im10.set_clim(-np.pi, np.pi)
        axes[1][0].set_xlabel('Frequency (Hz)')
        axes[1][0].set_ylabel('Sample #')
        axes[1][0].set_title('Complex Cross-Correlation Phase')

        im11 = axes[1][1].pcolormesh(X, Y, imag_part[::stride,:], shading='auto', cmap='viridis')
        axes[1][1].set_xlabel('Frequency (Hz)')
        axes[1][1].set_ylabel('Sample #')
        axes[1][1].set_title('Imag part of XCorrs')

        fig.colorbar(im00, ax=axes[0][0])
        fig.colorbar(im01, ax=axes[0][1])
        fig.colorbar(im10, ax=axes[1][0])
        fig.colorbar(im11, ax=axes[1][1])
    fig.tight_layout()

    print('Plotting complete.')

    plt.show()

    return


def post_process(raw_output, rate, fc, nfft, mode, omit_plot, test_delay_sweep_step=0):
    '''
    Handles displaying data.

    Parameters
    ----------
    raw_output : np.array
        If mode 'continuum', a 1d array of complex floats. If mode 'spectrum',
        a 2d array, each row being a complex visibility spectrum from a
        pair of SDR buffer reads
    rate : float
        SDR sample rate, samples per second.
    fc : float
        SDR center tuning frequency, Hz
    nfft : int
        Number of fft bins to use in frequency axis.
    mode : str
        Either 'continuum' for recording visibility amplitudes with time,
        or 'spectrum' for recording spectrum visibilities with time.
        Defaults to 'continuum'.
    omit_plot : bool
        If True, don't plot recorded data with matplotlib.
    test_delay_sweep_step : float (optional)
        If nonzero, aids plotting of test mode data by allowing x-axis to show
        delay in ns.

    Returns
    -------
    fname : str
        The filename to which output processed data is written.
    '''
    if not omit_plot:
        visualize(raw_output, rate, fc, nfft, mode, test_delay_sweep_step=test_delay_sweep_step)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Pull data from effex-generated .csv file and post-process it. Shows a plot.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename',
        type=str,
        help='(str) output visibilities.csv file from effex.'
    )

    args = parser.parse_args()

    metadata = {}
    for meta in np.loadtxt(args.filename, dtype=str, max_rows=1, delimiter=','):
        key, val = meta.split(':')
        metadata[key] = val

    if metadata['mode'].lower() in ['continuum', 'test']:
        skiprows = 1
    else:
        skiprows = 2

    # Magic number: (1/f) / 10 is the spacing the Correlator class uses in
    # to step through delay-space in test mode to ensure the delay pattern is
    # adequately sampled
    if 'test' == metadata['mode'].lower():
        test_delay_sweep_step = (1 / float(metadata['frequency'])) / 10.
    else:
        test_delay_sweep_step = 0

    output = np.loadtxt(args.filename, dtype=np.complex128, delimiter=',', skiprows=skiprows)

    post_process(output,
        float(metadata['bandwidth']),
        float(metadata['frequency']),
        int(metadata['resolution']),
        metadata['mode'].lower(),
        False, # never omit plot, that is the point
        test_delay_sweep_step=test_delay_sweep_step
    )

