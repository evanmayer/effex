"""
This FX correlator streams IQ data synchronously from 2 SDRs
to a deque circular buffer in pairs.
The GPU is kept fed by popping sample chunks off of 
the deque, performing polyphase filter-bank preprocessing.
Then the two streams are combined by cross-correlation.
"""

import asyncio
import concurrent.futures
import contextlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy import optimize
from scipy import stats
import sys
import time

import cupy as cp
import cusignal

from rtlsdr import RtlSdr


class Correlator(object):
    '''
    '''
    # -------------------------------------------------------------------------
    # Class constants
    # -------------------------------------------------------------------------
    _states = ('OFF', 'STARTUP', 'RUN', 'CALIBRATE', 'DRAIN')
    _modes = ('SPECTRUM', 'CONTINUUM')
    # sized for 4GB RAM on NVIDIA Jetson Nano
    BUFFER_SIZE =  int(5e8 // (2**18 * np.dtype(np.complex128).itemsize) // 2)
    # allow some time for streaming subprocesses to get to starting line
    STARTUP_DURATION = 1. # sec

    # -------------------------------------------------------------------------
    # Init and destruct
    # -------------------------------------------------------------------------
    def __init__(self,
                 run_time=1,
                 bandwidth=2.4e6,
                 frequency=1.4204e9,
                 num_samp=2**18,
                 nbins=2**12,
                 gain=49.6,
                 mode='SPECTRUM'):

        self.sdr0 = RtlSdr(device_index=0, dithering_enabled=False)        
        self.sdr1 = RtlSdr(device_index=1, dithering_enabled=False)

        self._state = 'OFF'
        self.run_time = run_time
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.num_samp = num_samp
        self.nbins = nbins
        self.gain = gain
        self.mode = mode

        assert(self._state in self._states), f'State {self._state} not in allowed states {self._states}.'
        assert(self._mode in self._modes), f'Mode {self._mode} not in allowed modes {self._modes}.'

    def close(self):
        self.sdr0.close()
        self.sdr1.close()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    class StateTransitionError(Exception):
        def __init__(self, prev, next):
            self.prev = prev
            self.next = next
            self.message = f'Transition from {self.prev} to {self.next} is not permitted.'
         
        def __str__(self):
            return repr(self.message)

    @property
    def state(self):
        '''The current state in the correlator's internal state machine.'''
        return self._state

    @state.setter
    def state(self, input_state):
        '''This state setter is used post-init to handle state transitions. State always starts out OFF.'''
        if input_state in self._states:
            # State transition checking
            if 'OFF' == self.state and 'STARTUP' != input_state:
                raise self.StateTransitionError(self.state, input_state)
            if 'STARTUP' == self.state and input_state not in ('RUN', 'OFF'):
                raise self.StateTransitionError(self.state, input_state)
            if 'RUN' == self.state and input_state not in ('CALIBRATE', 'DRAIN', 'OFF'):
                raise self.StateTransitionError(self.state, input_state)
            if 'CALIBRATE' == self.state and input_state not in ('RUN', 'DRAIN', 'OFF'):
                raise self.StateTransitionError(self.state, input_state)
            if 'DRAIN' == self.state and input_state not in ('RUN', 'CALIBRATE', 'OFF'):
                raise self.StateTransitionError(self.state, input_state)
            self._state = input_state
        else:
            raise ValueError(f'State {input_state} is not in known states: {self.states_}')

    @property
    def run_time(self):
        '''The amount of real-world time after which the correlator will shut down.'''
        return self._run_time

    @run_time.setter
    def run_time(self, run_time):
        if run_time < 1:
            raise ValueError(f'run time {run_time} is not allowed; run times must be >= 1 second.')
        else:
            self._run_time = run_time

    @property
    def bandwidth(self):
        '''The width in frequency over which observation takes place. Intrinsically tied to sample rate in SDRs.'''
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        threshold = 2.8e6
        if value > threshold:
            print(f'WARNING: bandwidth value {value} is greater than {threshold}, and RtlSdrs may not be stable.')
        self._bandwidth = value
        self.sdr0.rs = self._bandwidth
        self.sdr1.rs = self._bandwidth

    @property
    def frequency(self):
        '''The center tuning frequency of the correlator.'''
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.sdr0.fc = self._frequency
        self.sdr1.fc = self._frequency

    @property
    def gain(self):
        '''The tuner gain of the RtlSdr devices.'''
        return self._gain
                                                             
    @gain.setter
    def gain(self, value):
        self._gain = value
        self.sdr0.gain = self._gain
        self.sdr1.gain = self._gain

    @property
    def mode(self):
        '''The current data processing mode.'''
        return self._mode

    @mode.setter
    def mode(self, input_mode):
        if input_mode in self._modes:
            self._mode = input_mode
        else:
            raise ValueError(f'Mode input {input_mode} is not in known modes: {self._modes}')


def spectrometer_poly(x, n_taps, n_branches): 
    '''Polyphase channelize input data using cuSignal polyphase channelizer. Returns
    input array x, channelized into n_branches coefficients

    :param x: cupy.array, signal of interest
    :param n_taps: int, number of polyphase channelizer taps
    :param n_branches: int, number of polyphase channelizer branches
    :return: cupy.array, channelized
    :rtype: cupy.array
    '''
    # Create window coefficients
    w = cusignal.get_window("hamming", n_taps * n_branches)\
      * cusignal.firwin(n_taps * n_branches, cutoff=1.0/n_branches, window='rectangular')

    # Pad the signal to an even number of chunks
    x = cp.zeros(len(x)+len(x)%n_branches, dtype=np.complex128)[:len(x)] + x

    channelized = cusignal.filtering.channelize_poly(x, w, n_branches).T

    return channelized


def pfb_xcorr(gpu_iq_0, gpu_iq_1, total_delay, nfft, rate, fc, mode):
    '''Consume buffer data to compute PSDs in pairs and then cross-
    correlate them. Use mapped, pinned memory space allocated on the GPU.
    :param gpu_iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param gpu_iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :param total_delay: float, time delay in sec between channels 0 and 1.
    Calculated by sum of estimate_lag retvals.
    :param nfft: int, number of fft bins to use in psd.
    :param rate: float, SDR sample rate.
    :param fc: float, SDR center tuning frequency
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectrum visibilities with time.
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
    
    # Threading to take ffts using polyphase filterbank
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as iq_processor:
        future_0 = iq_processor.submit(spectrometer_poly, *(cp.array(gpu_iq_0), n_taps, n_branches))
        future_1 = iq_processor.submit(spectrometer_poly, *(cp.array(gpu_iq_1), n_taps, n_branches))
        try:
            f0 = future_0.result()
            f1 = future_1.result()
        except Exception as exc:
            print('pfb_spectrometer call generated an exception: %s' % (exc))
            raise exc

    # Apply phase gradient, inspired by 
    # http://www.gmrt.ncra.tifr.res.in/doc/WEBLF/LFRA/node70.html
    # implemented according to Thompson, Moran, Swenson's Interferometry and 
    # Synthesis in Radio Astronoy, 3rd ed., p.364: Fractional Sample Delay 
    # Correction
    f0 = cp.fft.fftshift(f0)
    f1 = cp.fft.fftshift(f1)
    freqs = cp.fft.fftshift(cp.fft.fftfreq(f0.shape[-1], d=1/rate))

    # Calculate cross-power spectrum and apply FSTC by a phase gradient
    rot = cp.exp(-2j * cp.pi * freqs * (-total_delay))
    xpower_spec = f0 * cp.conj(f1 * rot)
    xpower_spec = xpower_spec.mean(axis=0) # time average

    if 'continuum' == mode: # don't save spectral information
        vis = cp.mean(xpower_spec) * rate # Total power est. from PSD, the visibility amplitude
    else:
        vis = xpower_spec

    return vis


def estimate_delay(iq_0, iq_1, rate, fc):
    '''Returns delay estimate between channels in seconds.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :param rate: float, SDR sample rate.
    :param fc: float, SDR center tuning frequency
    :return: float, the delay estimate between channels in seconds
    :rtype: tuple
    '''
    integer_delay = estimate_integer_delay(iq_0, iq_1, rate)
    frac_delay = estimate_fractional_delay(iq_0, iq_1, integer_delay, rate, fc)
    total_delay = integer_delay + frac_delay

    return total_delay


def estimate_integer_delay(iq_0, iq_1, rate):
    '''Returns delay estimate between channels to the nearest sample division.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 1
    :param rate: float, SDR sample rate.
    :return: float, the delay estimate between channels in seconds
    :rtype: int
    '''
    # TODO: lift constraint on equal-length timeseries
    assert len(iq_0) == len(iq_1), ('Algorithm assumes input complex timeseries'
        + ' are of equal length.')

    # Find the delay to the nearest integer sample dt
    # First, pad by length of signals
    n = len(iq_0)
    iq_0_padded = cp.zeros(2 * n, dtype=cp.complex128)
    iq_1_padded = cp.zeros(2 * n, dtype=cp.complex128)
    iq_0_padded[0:n] += cp.array(iq_0)
    iq_1_padded[0:n] += cp.array(iq_1)

    f0 = cp.fft.fft(cp.array(iq_0_padded))
    f1 = cp.fft.fft(cp.array(iq_1_padded))
    xcorr = cp.fft.ifft(f0 * cp.conj(f1))
    xcorr = cp.fft.fftshift(xcorr)

    integer_lag = n - int(cp.argmax(cp.abs(xcorr)))
    integer_delay = integer_lag / rate

    return integer_delay


def estimate_fractional_delay(iq_0, iq_1, integer_delay, rate, fc):
    '''Returns fractional sampling time correction between channels in seconds.
    First corrects integer sample lag to make estimating the fractional lag
    tractable, then finds the slope of the phase of the cross-correlation by
    linear regression to estimate the fractional sample delay.
    :param iq_0: cusignal mapped, pinned array of GPU memory containing SDR
    data from channel 0
    :param iq_1: cusignal mapped, pinned array of GPU memory containing
    SDR data from channel 1
    :param integer_delay: int, estimated value from estimate_integer_delay()
    :param rate: float, SDR sample rate.
    :param fc: float, SDR center tuning frequency
    :return: float, frac_delay, the fractional lag between the argument signals
    in seconds
    :rtype: float
    '''
    N = 8192
    f0 = cp.fft.fftshift(cp.fft.fft(cp.array(iq_0), n=N))
    f1 = cp.fft.fftshift(cp.fft.fft(cp.array(iq_1), n=N))
    freqs = cp.fft.fftshift(cp.fft.fftfreq(N, d=1/rate)) + fc

    # Yes, there is a double negative, for mathematical clarity. Most eqns
    # have -i, and we are "undoing" the delay, so -delay.
    rot = cp.exp(-2j * cp.pi * freqs * -integer_delay)
    # Integer sample correction as a phase rotation in frequency space
    xcorr = f0 * cp.conj(f1 * rot)

    # Prepare to fit residual phase gradient:
    phases = cp.angle(xcorr)
    # Due to receiver bandpass shape, edge frequencies have less power => less certain phase
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

    frac_delay = slope

    if np.abs(frac_delay) > 1/rate:
        fig = plt.figure(100)
        ax = plt.axes()
        ax.scatter(cp.asnumpy(freqs), cp.asnumpy(phases), alpha=0.1, label='Calibration data: phase')
        ax.plot(cp.asnumpy(freqs), model(cp.asnumpy(freqs), slope, intc), color='red', label='Fit: phase slope')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (rad)')
        ax.legend(loc='best', framealpha=0.4)
        fig.show()
        print('WARNING: 1st-pass delay calibration failed: '
            + 'fractional sample time correction, |{}| > 1/sample rate, {} '.format(frac_delay, 1/rate))

    return frac_delay


async def streaming(sdr, buf, num_samp, start_time, run_time):
    '''Begins streaming sample chunks from a pyrtlsdr RtlSdr() instance to a
    multiprocess.Queue() buffer at a given time and stops at a given later time.
    :param sdr: RtlSdr() instance. Should already be initialized/tuned to the
    frequency of interest.
    :param buf: multiprocessing.Queue() instance, buffer to put sample np.arrays in
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
    finally:
        await sdr.stop()
                                                                                                               
    print('Buffering ended at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(time.time()))))


def process_iq(buf_0, buf_1, num_samp, nfft, rate, fc, start_time, run_time, mode):
    '''This is the main function of the correlator. It holds on to two
    multiprocessing.Queue() instances, one for each SDR channel, and handles
    taking chunks of num_samp IQ samples off of each buffer and sending them to
    mapped, pinned GPU memory to keep the polyphase filterbank-driven cross-
    correlation function fed. It also handles calculating and calibrating out
    the initial delay caused by cables and USB sampling between the SDR
    channels to "phase-up" the array. 
    :param buf_0: multiprocessing.Queue() instance
    :param buf_1: multiprocessing.Queue() instance
    :param num_samp: int, number of samples to read async from sdr at a time.
    2^18 works well for RTL-SDRblog v3 dongles.
    :param nfft: int, number of fft bins to use in psd.
    :param rate: float, SDR sample rate.
    :param fc: float, SDR center tuning frequency
    :param start_time: float, time in ms since UNIX epoch to begin streaming
    async samples. Helps multiple streaming processes start closer to the same time.
    :param run_time: float, time in ms since UNIX epoch to end streaming async
    samples.
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectrum visibilities with time.
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
                total_delay = estimate_delay(gpu_iq_0, gpu_iq_1, rate, fc)
                print('Estimated delay (us): {}'.format(1e6 * total_delay))
                first_time = False

            visibility = pfb_xcorr(gpu_iq_0, gpu_iq_1, total_delay, nfft, rate, fc, mode)
            vis_out.append(visibility)

    return vis_out


def post_process(raw_output, rate, fc, nfft, num_samp, mode):
    '''Handles saving and displaying data.
    :param raw_output: python list, if mode 'continuum', a list of visibility
    amplitudes, if mode 'spectrum', a list of cupy arrays, each
    one being a complex visibility spectrum from a pair of SDR
    buffer reads
    :param rate: float, SDR sample rate.
    :param fc: float, SDR center tuning frequency
    :param nfft: int, number of fft bins to use in psd.
    :param num_samp: int, number of samples to read async from sdr at a time.
    :param mode: str, either 'continuum' for recording visibility amplitudes
    with time, or 'spectrum' for recording spectrum visibilities with time.
    Defaults to 'continuum'.
    :return: fname, the filename to which output processed data is written
    :rtype: str
    '''
    def record_visibilities(visibilities, fc, mode):
        '''
        :param visibilites: ndim cupy array
        :param mode: str, either 'continuum' for recording visibility amplitudes
        with time, or 'spectrum' for recording spectrum visibilities with time.
        Defaults to 'continuum'.
        :return: str, fname, filename that was written to
        :rtype: str
        '''
        fname = time.strftime('visibilities_%Y%m%d-%H%M%S')+'.csv'             
        
        if 'continuum' == mode: # Continuum mode, don't save spectral information
            visibilities = visibilities.flatten()
            with open(fname, 'a') as f:
                np.savetxt(f, cp.asnumpy(visibilities), delimiter=',')
        else:
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/rate)) + fc
            with open(fname, 'ab') as f:
                np.savetxt(f, [freqs], delimiter=',')
                np.savetxt(f, cp.asnumpy(visibilities), delimiter=',')

        return fname


    def visualize(visibilities, rate, fc, nfft, num_samp, mode):
        '''Handles plotting 1D continuum data or 2D spectrum data with respect to time.
        :param visibilites: ndim cupy array, output of correlator function pfb_xcorr
        :param mode: str, either 'continuum' for recording visibility amplitudes
        with time, or 'spectrum' for recording spectrum visibilities with time.
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
            #ECM: TODO: delay-space sweep not implemented in production for now
            sweep_step = False
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

    fname = record_visibilities(visibilities, fc, mode)
    print('Data recorded to {}.'.format(fname))

    visualize(visibilities, rate, fc, nfft, num_samp, mode)

