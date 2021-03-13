"""
This FX correlator streams IQ data synchronously from 2 SDRs
to a deque circular buffer in pairs.
The GPU is kept fed by popping sample chunks off of 
the deque, performing polyphase filter-bank preprocessing.
Then the two streams are combined by cross-correlation.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import asyncio
import multiprocessing
import concurrent.futures
from scipy import stats
from scipy import optimize

import cupy as cp
import cusignal
from rtlsdr import RtlSdr


def spectrometer_poly(x, n_taps, n_branches): 
    '''Take the fft of input data using cuSignal polyphase channelizer. Returns
    the power spectral density of x.

    :param x: cupy.array, signal of interest
    :param n_taps: int, number of polyphase channelizer taps
    :param n_branches: int, number of polyphase channelizer branches
    :return: cupy.array, psd(x)
    :rtype: cupy.array
    '''
    # Create window coefficients
    w = cusignal.get_window("hamming", n_taps * n_branches)\
      * cusignal.firwin(n_taps * n_branches, cutoff=1.0/n_branches, window='rectangular')

    # Pad the signal to an even number of chunks
    x = cp.zeros(len(x)+len(x)%n_branches, dtype=np.complex128) + x

    channelized = cusignal.filtering.channelize_poly(x, w, n_branches).T
    x_psd = cp.fft.fftshift(channelized)

    # Get rid of that nasty DC spike, thanks
    x_psd[:, x_psd.shape[-1]//2] = (x_psd[:, -1 + x_psd.shape[-1]//2] + x_psd[:, 1 + x_psd.shape[-1]//2]) / 2.   

    return x_psd


def pfb_xcorr(gpu_iq_0, gpu_iq_1, total_lag, nfft=8192, mode='continuum'):
    '''Consume buffer data to compute PSDs in pairs and then cross-
    correlate them. Use mapped, pinned memory space allocated on the GPU.
    :param gpu_iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param gpu_iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :param total_lag: float, number of samples lag between channels 0 and 1.
    Calculated by sum of estimate_lag retvals.
    :param nfft: int, number of fft bins to use in psd. Defaults to 8192.
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectram visibilities with time.
    Defaults to 'continuum'.
    :return: the result of one complex cross-correlation of the input IQ data.
    :rtype: If mode == 'continuum', float. If mode =='spectrum', cupy.array.
    '''
    n_branches = nfft # Number of 'branches', also fft length
    n_taps = 4 # Number of taps in PFB
    # Constraint: input timeseries only affords us n_taps * n_int ffts
    # of length n_branches in our PFB.
    n_int = len(gpu_iq_0) // n_taps // n_branches

    if (n_int < 1):
        raise ValueError('Assertion failed: there must be at least 1 window of '
             +'length n_branches*n_taps in each input timestream.\n'
             +'timestream len: {}\n'.format(len(gpu_iq_0))
             +'n_branches: {}\n'.format(n_branches)
             +'n_taps: {}\n'.format(n_taps)
             +'n_branches*n_taps: {}'.format(n_branches*n_taps))
    if (0 != (len(gpu_iq_0) % (n_taps * n_branches))):
        raise ValueError('Assertion failed: n_taps * n_branches must '
             +'divide length of input timestream evenly.\n'
             +'timestream len: {}\n'.format(len(gpu_iq_0))
             +'n_branches: {}\n'.format(n_branches)
             +'n_taps: {}\n'.format(n_taps))
    
    # Threading to take ffts using polyphase filterbank
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as iq_processor:
        future_0 = iq_processor.submit(spectrometer_poly, *(cp.array(gpu_iq_0), n_taps, n_branches))
        future_1 = iq_processor.submit(spectrometer_poly, *(cp.array(gpu_iq_1), n_taps, n_branches))
        try:
            psd_0 = future_0.result()
            psd_1 = future_1.result()
        except Exception as exc:
            print('pfb_spectrometer call generated an exception: %s' % (exc))
            raise exc

    # PSDs S(\nu) come out
    # Apply phase gradient,
    # According to http://www.gmrt.ncra.tifr.res.in/gmrt_hpage/Users/doc/WEBLF/LFRA/node70.html,
    freqs = cp.fft.fftshift(cp.fft.fftfreq(psd_1.shape[-1], d=1/rate)) + fc

    xcorr_array = psd_0 * (cp.conj(psd_1) * cp.exp(2j * (cp.pi) * freqs * (total_lag / rate) )) 

    # Average each row of PSDs together to get the visiblity
    spec_vis = xcorr_array.mean(axis=0) # The spectral visibility
    if 'continuum' == mode: # don't save spectral information
        vis = cp.mean(spec_vis) * rate # Total power est. from PSD, the visibility amplitude
    else:
        vis = spec_vis

    # Continuous monitoring
    if (0 == int(time.time() - start_time) % 30):
        np.savetxt('live_monitoring.csv', cp.asnumpy(cp.array(vis).flatten()), delimiter=',')
    
    return vis


def estimate_lag(iq_0, iq_1):
    '''Returns integer and fractional sample lag estimates. The sum is the 
    total estimated lag in samples between signals iq_0 and iq_1.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :return: int, integer_lag, the integer lag between the argument signals in
    samples, and frac_lag, the fractional lag between the argument signals in
    fractions of a sample
    :rtype: tuple
    '''
    integer_lag = estimate_integer_lag(iq_0, iq_1)
    frac_lag = estimate_fractional_lag(iq_0, iq_1, integer_lag)

    return integer_lag, frac_lag


def estimate_integer_lag(iq_0, iq_1):
    '''Returns integer sample lag estimate in samples between signals iq_0 and
    iq_1.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :return: int, integer_lag, the integer lag between the argument signals in
    samples
    :rtype: int
    '''
    # Perform fractional sampling time correction on complex samples
    # Find the integer number of samples of the delay 
    # This is a common way to find lag in samples between timeseries, see
    # https://www.dsprelated.com/showcode/207.php
    xcorr = cp.fft.fft(cp.array(iq_0)) * cp.conj(cp.fft.fft(cp.array(iq_1)))
    #fig = plt.figure(99)
    #ax = plt.axes()
    #ax.plot(cp.asnumpy(cp.abs(cp.fft.ifft(xcorr))))
    #fig.show()
    
    integer_lag = int(cp.argmax(cp.abs(cp.fft.ifft(xcorr))))
    return integer_lag


def estimate_fractional_lag(iq_0, iq_1, integer_lag):
    '''Returns fractional sample lag estimate in samples between signals iq_0
    and iq_1. First corrects integer sample lag to make estimating the
    fractional lag tractable, then finds the slope of the phase of the cross-
    correlation by linear regression to estimate the fractional lag.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing
    SDR data from channel 1
    :return: float, frac_lag, the fractional lag between the argument signals
    in samples
    :rtype: float
    '''
    xcorr = cp.fft.fftshift(cp.fft.fft(cp.array(iq_0)) * cp.conj(cp.fft.fft(cp.array(iq_1))))
    freqs = cp.fft.fftshift(cp.fft.fftfreq(len(xcorr), d=1/rate)) + fc
    # Integer sample correction as a phase rotation in frequency space
    xcorr *= cp.exp(2j * cp.pi * freqs * (integer_lag / rate) )
    # Prepare to fit residual phase gradient:
    phases = cp.angle(xcorr)
    # Due to RTLSDR bandpass shape, edge frequencies have less power => less certain phase
    # Assign weights accordingly
    weights = cp.abs(xcorr)
    weights /= cp.max(weights)
    # Fit phase slope across band
    # https://scipython.com/book/chapter-8-scipy/examples/weighted-and-non-weighted-least-squares-fitting/ 
    def model(x, m, b):
        # From "Reliable fitting of phase data without unwrapping by wrapping the fit model," Kramer et. al. 2012
        return ((m * x + b) + np.pi) % (2. * np.pi) - np.pi
    # initial guesses
    p0 = [0, 0]
    # fitting
    popt, pcov = optimize.curve_fit(model,
        cp.asnumpy(freqs),
        cp.asnumpy(phases), 
        p0,
        sigma=1./cp.asnumpy(weights),
        absolute_sigma=False # not real sigmas, just weights
    )
    slope, intc = popt

    frac_lag = slope * rate

    if np.abs(frac_lag) > 1:
        fig = plt.figure(100)
        ax = plt.axes()
        ax.scatter(cp.asnumpy(freqs), cp.asnumpy(phases), alpha=0.1, label='Calibration data: phase')
        ax.plot(cp.asnumpy(freqs), model(cp.asnumpy(freqs), slope, intc), color='red', label='Fit: phase slope')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (rad)')
        ax.legend(loc='best', framealpha=0.4)
        fig.show()
        print('WARNING: Integer delay calibration failed: '
            + 'calculated fractional sample delay |{}| > 1 '.format(frac_lag))

    return frac_lag


async def streaming(sdr, buf, num_samp, start_time, run_time):
    '''Begins streaming sample chunks from a pyrtlsdr RtlSdr() instance to a
    multiprocess.Queue() buffer at a given time and stops at a given later time.
    :param sdr: RtlSdr() instance. Should already be initialized/tuned to the
    frequency of interest.
    :param num_samp: int, number of samples to read async from sdr at a time.
    2^18 works well for RTL-SDRblog v3 dongles.
    :param start_time: float, time in ms since UNIX epoch to begin streaming
    async samples. Helps multiple streaming processes start closer to the same time.
    :param run_time: float, time in ms since UNIX epoch to end streaming async
    samples.
    '''
    while time.time() < start_time:
        await asyncio.sleep(1e-9)
    try: 
        async for samples in sdr.stream(format='samples', num_samples_or_bytes=num_samp):
            buf.put(samples)
            if (time.time() - start_time > run_time):
                break
    except Exception as exc:
        print('streaming call generated an exception: %s' % (exc))
        raise exc
    await sdr.stop()
                                                                                                               
    print('Buffering ended at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(time.time()))))

    return


def process_iq(buf_0, buf_1, num_samp, nfft, start_time, run_time, mode):
    '''This is the main function of the correlator. It holds on to two
    multiprocessing.Queue() instances, one for each SDR channel, and handles
    taking chunks of num_samp IQ samples off of each buffer and sending them to
    mapped, pinned GPU memory to keep the polyphase filterbank-driven cross-
    correlation function fed. It also handles calculating and calibrating out
    the initial delay caused by cables and USB sampling between the SDR
    channels to "phase-up" the array. Finally, it can optionally sweep through
    a series of delays to generate the interferometer response in delay-space.
    :param buf_0: multiprocessing.Queue() instance
    :param buf_1: multiprocessing.Queue() instance
    :param num_samp: int, number of samples to read async from sdr at a time.
    2^18 works well for RTL-SDRblog v3 dongles.
    :param nfft: int, number of fft bins to use in psd. Defaults to 8192.
    :param start_time: float, time in ms since UNIX epoch to begin streaming
    async samples. Helps multiple streaming processes start closer to the same time.
    :param run_time: float, time in ms since UNIX epoch to end streaming async
    samples.
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectram visibilities with time.
    Defaults to 'continuum'.
    :return: vis_out, a list of either floats or cupy.arrays, depending on mode.
    '''
    # Store off cross-correlated chunks of IQ samples
    vis_out = []
    # Create mapped, pinned memory for zero copy between CPU and GPU
    gpu_iq_0 = cusignal.get_shared_mem(num_samp, dtype=np.complex128)
    gpu_iq_1 = cusignal.get_shared_mem(num_samp, dtype=np.complex128)
    first_time = True
    while True:
        buf_0_empty = False
        buf_1_empty = False
        if time.time() < start_time:
            continue
        try: 
            data_0 = buf_0.get(block=True, timeout=1)
        except:
            buf_0_empty = True
        try: 
            data_1 = buf_1.get(block=True, timeout=1)
        except:
            buf_1_empty = True
        if (buf_0_empty and buf_1_empty):
            if time.time()-start_time < run_time:
                continue
            else:
                break
        else:
            # Complex chunks of IQ data vs. time go over to GPU
            gpu_iq_0[:] = data_0
            gpu_iq_1[:] = data_1
            # Self-calibration assumes a noise source w/flat PSD in-band is 
            # used as input on first cycle.
            # Estimate integer and fractional sample delays
            if first_time:
                integer_lag, frac_lag = estimate_lag(gpu_iq_0, gpu_iq_1)
                total_lag = integer_lag + frac_lag
                print('Estimated lag (samples): {} + {}'.format(integer_lag, frac_lag))
                total_lag -= 1
            # Provide a rudimentary method of sweeping phase in time: increase
            # lag in samples
            elif sweep_step:
                total_lag += sweep_step
                print('Estimated lag (samples): {}'.format(float(total_lag)))

            visibility = pfb_xcorr(gpu_iq_0, gpu_iq_1, total_lag, nfft=nfft, mode=mode)
            vis_out.append(visibility)
            first_time = False

    return vis_out


def post_process(raw_output, mode):
    '''Handles saving and displaying data.
    :param raw_output: python list, if mode 'continuum', a list of visibility
    amplitudes, if mode 'spectrum', a list of cupy arrays, each
    one being a complex visibility spectrum from a pair of SDR
    buffer reads.
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectram visibilities with time.
    Defaults to 'continuum'.
    :return: fname, the filename to which output processed data is written
    :rtype: str
    '''
    def record_visibilities(visibilities, mode):
        '''
        Inputs:
        - visibilites: ndim cupy array
        - mode: ['continuum', 'spectrum']
        '''
        fname = time.strftime('visibilities_%Y%m%d-%H%M%S')+'.csv'             
        
        if 'continuum' == mode: # Continuum mode, don't save spectral information
            visibilities = visibilities.flatten()
            with open(fname, 'a') as f:
                np.savetxt(f, cp.asnumpy(visibilities), delimiter=',')
        else:
            freqs = fc + np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate))
            with open(fname, 'ab') as f:
                np.savetxt(f, [freqs], delimiter=',')
                np.savetxt(f, cp.asnumpy(visibilities), delimiter=',')

        return fname


    def visualize(visibilities, mode):
        '''Handles plotting 1D continuum data or 2D spectrum data with respect to time.
        :param visibilites: ndim cupy array, output of correlator function pfb_xcorr
        :param mode: str, either 'continuum' for recording visibility amplitudes
        with time, or 'spectrum' for recording spectram visibilities with time.
        Defaults to 'continuum'.
        '''
        amp = cp.asnumpy(cp.sqrt(cp.real(visibilities * cp.conj(visibilities))))
        phase = cp.asnumpy(cp.angle(visibilities))
        real_part = cp.asnumpy(cp.real(visibilities))
        imag_part = cp.asnumpy(cp.imag(visibilities))
        
        if 'continuum' == mode:
            sharey = 'none'
        else:
            sharey = 'all'
                                                                                          
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey=sharey)
        fig.tight_layout()
                                                                                          
        if 'continuum' == mode:
            # Convert x axis from SDR samples to time delay
            samples = np.arange(0, len(amp))                                    
            if sweep_step:
                samples_to_ns = sweep_step / rate / 1e-9
                delay_ns = samples * samples_to_ns
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
            axes[0][1].set_ylabel('Re[xcorr]')
            axes[0][1].set_title('Complex Cross-Correlation Re{}')
            axes[0][1].legend(loc='best')
            
            im10 = axes[1][0].plot(x, phase)
            axes[1][0].set_xlabel(xlabel)
            axes[1][0].set_ylabel('Phase')
            axes[1][0].set_title('Complex Cross-Correlation Phase')
                                                                                
            im11 = axes[1][1].plot(x, imag_part, label='imag_part')
            axes[1][1].set_xlabel(xlabel)
            axes[1][1].set_ylabel('Amplitude')
            axes[1][1].set_title('Complex Cross-Correlation imag')
        else:
            freqs = fc + np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate))
            num_spectra = np.array(range(visibilities.shape[0]))
            X,Y = np.meshgrid(freqs, num_spectra)

            im00 = axes[0][0].pcolormesh(X, Y, amp, shading='auto', cmap='viridis')
            axes[0][0].set_xlabel('Frequency (Hz)')
            axes[0][0].set_ylabel('Sample #')
            axes[0][0].set_title('Complex Cross-Correlation Amplitude')
        
            im01 = axes[0][1].pcolormesh(X, Y, real_part, shading='auto', cmap='viridis')
            axes[0][1].set_xlabel('Frequency (Hz)')
            axes[0][1].set_ylabel('Sample #')
            axes[0][1].set_title('Real part of XCorrs')
            
            im10 = axes[1][0].pcolormesh(X, Y, phase, shading='auto', cmap='viridis')
            im10.set_clim(-np.pi, np.pi)
            axes[1][0].set_xlabel('Frequency (Hz)')
            axes[1][0].set_ylabel('Sample #')
            axes[1][0].set_title('Complex Cross-Correlation Phase')
        
            im11 = axes[1][1].pcolormesh(X, Y, imag_part, shading='auto', cmap='viridis')
            axes[1][1].set_xlabel('Frequency (Hz)')
            axes[1][1].set_ylabel('Sample #')
            axes[1][1].set_title('Imag part of XCorrs')
        
            fig.colorbar(im00, ax=axes[0][0])
            fig.colorbar(im01, ax=axes[0][1])
            fig.colorbar(im10, ax=axes[1][0])
            fig.colorbar(im11, ax=axes[1][1])
                                                                                          
        plt.show()

        return

    # Convert list to cupy array
    visibilities = cp.array(raw_output)
    print('Processed', len(visibilities) * 2 * num_samp * iq_0.itemsize / run_time, 'Bytes/sec')

    fname = record_visibilities(visibilities, mode)
    print('Data recorded to {}.'.format(fname))

    visualize(visibilities, mode)

    return



if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # SDR INIT
    # -------------------------------------------------------------------------
    num_samp = 2**18
    rate = 2.2e6
    fc = 1420.4e6 # Frequency of interest
    gain = 49.6
    
    # Dithering depends on evanmayer's fork of roger-'s pyrtlsdr and keenerd's
    # experimental fork of librtlsdr
    sdr_0 = RtlSdr(device_index=0, dithering_enabled=False)
    sdr_1 = RtlSdr(device_index=1, dithering_enabled=False)
    sdr_0.rs = rate
    sdr_1.rs = rate
    sdr_0.fc = fc
    sdr_1.fc = fc
    sdr_0.gain = gain
    sdr_1.gain = gain
    
    # -------------------------------------------------------------------------
    # CPU & GPU MEMORY SETUP
    # -------------------------------------------------------------------------
    iq_0 = np.array([], dtype=np.complex128)
    iq_1 = np.array([], dtype=np.complex128)

    # Store sample chunks in 2 deques
    d_len = int(5e8 // (num_samp * iq_0.itemsize) // 2)
    buf_0 = multiprocessing.Queue(d_len)
    buf_1 = multiprocessing.Queue(d_len)

    # FFT Frequency bin resolution
    nfft = 2**12

    # -------------------------------------------------------------------------
    # RUN 
    # -------------------------------------------------------------------------
    # ['spectrum', 'continuum']
    mode = 'spectrum'
    run_time = 1
    sweep_step = 0.0 # 2e-4 # of samples to add to signal delay on each xcorr to sweep across delay space
    start_time = time.time() + 1 # Give streaming processes a second to get to to the starting line
    print('Interferometry begins at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(start_time))))
    # IQ source processes
    proc_0 = streaming(sdr_0, buf_0, num_samp, start_time, run_time)
    producer_0 = multiprocessing.Process(target=asyncio.run, args=(proc_0,))
    proc_1 = streaming(sdr_1, buf_1, num_samp, start_time, run_time)
    producer_1 = multiprocessing.Process(target=asyncio.run, args=(proc_1,))
    producer_0.start()
    producer_1.start()
    
    # IQ sink
    raw_output = process_iq(buf_0, buf_1,
                            num_samp,
                            nfft,
                            start_time, run_time,
                            mode=mode)

    print('IQ processing complete, buffers drained.')
    sdr_0.close()
    sdr_1.close()
    print('SDRs closed.')

    post_process(raw_output, mode)

