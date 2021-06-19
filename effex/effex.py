import argparse
import asyncio
import concurrent.futures
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy import optimize
from scipy import stats
import sys
import threading
import time

import cupy as cp
import cusignal

from rtlsdr import RtlSdr


LINESEP = '-' * 80


class Correlator(object):
    '''
    This FX correlator streams IQ data synchronously from 2 SDRs to 2 deque
    circular buffers.
    The GPU is kept fed by popping sample chunks off of the deques, performing
    polyphase filter-bank preprocessing.
    Then the two streams are combined by cross-correlation.
    '''
    # -------------------------------------------------------------------------
    # Class constants
    # -------------------------------------------------------------------------
    _states = ('OFF', 'STARTUP', 'RUN', 'CALIBRATE', 'SHUTDOWN')
    _modes = ('SPECTRUM', 'CONTINUUM', 'TEST')

    _BUFFER_SIZE = int(5e8 // (2**18 * np.dtype(np.complex128).itemsize) // 2)
    '''sized to easily fit several large of np.complex128 arrays in the 4GB RAM on an NVIDIA Jetson Nano'''
    _STARTUP_DURATION = 1. # sec
    '''allow some time for _streaming subprocesses to get to starting line'''

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
                 mode='SPECTRUM',
                 loglevel='INFO'):

        # ---------------------------------------------------------------------
        # LOGGING
        # ---------------------------------------------------------------------
        _level = getattr(logging, loglevel)
        # Set up our logger:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(_level)
        _fh = logging.FileHandler('log_effex.log')
        _fh.setLevel(_level)
        _ch = logging.StreamHandler()
        _ch.setLevel(_level)
        # create formatter and add it to the handlers
        _formatter = logging.Formatter('{asctime} - {name} - {levelname:<8} - {message}', style='{')
        _fh.setFormatter(_formatter)
        _ch.setFormatter(_formatter)
        # add the handlers to the logger
        self.logger.addHandler(_fh)
        self.logger.addHandler(_ch)

        # ---------------------------------------------------------------------
        # SDR INIT
        # ---------------------------------------------------------------------
        # Dithering depends on evanmayer's fork of roger-'s pyrtlsdr and
        # keenerd's experimental fork of librtlsdr
        self.sdr0 = RtlSdr(device_index=0, dithering_enabled=False)
        self.sdr1 = RtlSdr(device_index=1, dithering_enabled=False)

        self.run_time = run_time
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.num_samp = num_samp
        self.nbins = nbins
        self.gain = gain

        # ----------------------------------------------------------------------
        # STATE MACHINE INIT
        # ----------------------------------------------------------------------
        self._state = 'OFF'
        self.mode = mode

        assert(self._state in self._states), f'State {self._state} not in allowed states {self._states}.'

        self.start_time = -1

        # ----------------------------------------------------------------------
        # CPU & GPU MEMORY SETUP
        # ----------------------------------------------------------------------
        # Store sample chunks in 2 queues
        self.buf0 = multiprocessing.Queue(Correlator._BUFFER_SIZE)
        self.buf1 = multiprocessing.Queue(Correlator._BUFFER_SIZE)

        # Create mapped, pinned memory for zero copy between CPU and GPU
        self.gpu_iq_0 = cusignal.get_shared_mem(self.num_samp, dtype=np.complex128)
        self.gpu_iq_1 = cusignal.get_shared_mem(self.num_samp, dtype=np.complex128)

        # ----------------------------------------------------------------------
        # SPECTROMETER SETUP
        # ----------------------------------------------------------------------
        self.ntaps = 4 # Number of taps in PFB
        # Constraint: input timeseries only affords us ntaps * n_int ffts
        # of length nbins in our PFB.
        n_int = len(self.gpu_iq_0) // self.ntaps // self.nbins
        assert(n_int >= 1), 'Assertion failed: there must be at least 1 window of '\
                          +'length n_branches*ntaps in each input timeseries.\n'\
                          +'timeseries len: {}\n'.format(len(self.gpu_iq_0))\
                          +'n_branches: {}\n'.format(self.nbins)\
                          +'ntaps: {}\n'.format(self.ntaps)\
                          +'n_branches*ntaps: {}'.format(self.nbins*self.ntaps)
        # Create window coefficients for spectrometer
        self.window = (cusignal.get_window("hamming", self.ntaps * nbins)
                     * cusignal.firwin(self.ntaps * self.nbins, cutoff=1.0/self.nbins, window='rectangular'))

        # ---------------------------------------------------------------------
        # SCIENCE DATA
        # ---------------------------------------------------------------------
        self.calibrated_delay = 0 # seconds
        # Store off cross-correlated chunks of IQ samples
        self.vis_out = multiprocessing.Queue()
        # A file to archive data
        self.output_file = time.strftime('visibilities_%Y%m%d-%H%M%S')+'.csv'
        # A thread to handle buffering data to file
        self.output_thread = None # Will be assigned once a file handle is available

        # ---------------------------------------------------------------------
        # USER INPUT
        # ---------------------------------------------------------------------
        # Thread for keyboard input
        self.kbd_queue = multiprocessing.Queue(1)
        self.input_thread = threading.Thread(target=self._get_kbd, args=(self.kbd_queue,), daemon=True)

        # ---------------------------------------------------------------------
        # TEST MODE PARAMS
        # ---------------------------------------------------------------------
        # In test mode, the delay between the channels is calibrated out, and
        # then an artifical sweep in delay-space begins.
        # To goal is to reproduce the fringe pattern of an interferometer,
        # a sinusoid of period 1/fc modulated by something resembling a
        # sinc function, having first nulls at ~+/-1/bandwidth.
        crit_delay = 1 / self.frequency
        # Step through delay space with balance between sampling fidelity and
        # sweep speed:
        self.test_delay_sweep_step = crit_delay / 10
        self.test_delay_offset = self.test_delay_sweep_step * 200


    def _get_kbd(self, queue):
        # Helper function to run in a separate thread and add user input chars to a buffer.
        # Ends listening at end of scheduled run time
        while self.state in ['RUN', 'CALIBRATE']:
             queue.put(sys.stdin.read(1))


    def close(self):
        '''Function run upon shutdown to release the program's lock on the SDR instances.'''
        self.sdr0.close()
        self.sdr1.close()
        self.logger.info('SDRs closed.')


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


    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def state(self):
        '''The current state in the correlator's internal state machine.'''
        return self._state

    @state.setter
    def state(self, input_state):
        '''This state setter is used post-init to handle state transitions. State always starts out OFF.'''
        self.logger.debug(f'State transition: {self._state} to {input_state}')
        if input_state in self._states:
            # State transition checking
            if 'OFF' == self.state and 'STARTUP' != input_state:
                self.close()
                raise self.StateTransitionError(self.state, input_state)
            if 'STARTUP' == self.state and input_state not in ('CALIBRATE', 'RUN', 'SHUTDOWN'):
                self.close()
                raise self.StateTransitionError(self.state, input_state)
            if 'RUN' == self.state and input_state not in ('CALIBRATE', 'SHUTDOWN'):
                self.close()
                raise self.StateTransitionError(self.state, input_state)
            if 'CALIBRATE' == self.state and input_state not in ('RUN', 'SHUTDOWN'):
                self.close()
                raise self.StateTransitionError(self.state, input_state)
            if 'SHUTDOWN' == self.state and 'OFF' != input_state:
                self.close()
                raise self.StateTransitionError(self.state, input_state)
            self._state = input_state
        else:
            self.close()
            raise ValueError(f'State {input_state} is not in known states: {self._states}')


    @property
    def run_time(self):
        '''The amount of real-world time after which the correlator will shut down.'''
        return self._run_time

    @run_time.setter
    def run_time(self, run_time):
        if run_time < 1:
            self.close()
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
            self.logger.warning(f'Bandwidth value {value} is greater than {threshold}, and RtlSdrs may not be stable.')
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
    def num_samp(self):
        '''The number of samples read for each SDR on each async call. Powers of 2 between 2^8 and 2^18 work.'''
        return self._num_samp

    @num_samp.setter
    def num_samp(self, value):
        int_val = int(round(value))
        if int_val < 2**8:
            value = 2**8
        elif int_val > 2**18:
            value = 2**18
        self._num_samp = value


    @property
    def nbins(self):
        '''The number of frequency bins for spectrometry. Use powers of 2 for optimal FFT performance.'''
        return self._nbins

    @nbins.setter
    def nbins(self, value):
        self._nbins = value


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
        input_mode = input_mode.upper()
        if input_mode in self._modes:
            self._mode = input_mode
        else:
            raise ValueError(f'Mode input {input_mode} is not in known modes: {self._modes}')


    # --------------------------------------------------------------------------
    # Class methods
    # --------------------------------------------------------------------------
    def run_state_machine(self):
        '''
        Start the state machine handling running, calibration, and shutdown.
        '''
        with open(self.output_file, 'a') as file_handle:
            # Assign the file handle to the output buffering thread
            self.output_thread = threading.Thread(target=self._write_data,
                args=(file_handle,),
                daemon=True
            )
            while True:
                # Check for user input
                if not self.kbd_queue.empty():
                    kbd_in = self.kbd_queue.get_nowait()
                    if 'c' == kbd_in:
                        self.logger.info('Calibration requested.')
                        self.state = 'CALIBRATE'

                if self.buf0.qsize() == Correlator._BUFFER_SIZE:
                    self.logger.warning('SDR buffer 0 filled up. Data may have been lost!')
                if self.buf1.qsize() == Correlator._BUFFER_SIZE:
                    self.logger.warning('SDR buffer 1 filled up. Data may have been lost!')

                if 'OFF' == self.state:
                    self.state = 'STARTUP'
                elif 'STARTUP' == self.state:
                    self._startup_task()
                    self.state = 'CALIBRATE'
                # Should we be pulling data?
                elif self.state in ['CALIBRATE', 'RUN']:
                    if time.time() < self.start_time:
                        continue

                    buf0_empty = False
                    buf1_empty = False
                    try: 
                        data_0 = self.buf0.get(block=True, timeout=1)
                    except:
                        self.logger.debug('Buffer 0 empty')
                        buf0_empty = True
                    try: 
                        data_1 = self.buf1.get(block=True, timeout=1)
                    except:
                        self.logger.debug('Buffer 1 empty')
                        buf1_empty = True
                    # Is it time to stop?
                    if (buf0_empty and buf1_empty):
                        if time.time() - self.start_time < self.run_time:
                            self.logger.debug('Both buffers empty, waiting')
                            continue
                        else:
                            # Wait for output buffer to drain
                            if self.vis_out.empty():
                                self.logger.info('IQ processing complete, buffers drained. Shutting down.')
                                self.state = 'SHUTDOWN'
                            else:
                                self.logger.debug('Time up, but waiting for output buffer to drain.')
                    else:
                        # Complex chunks of IQ data vs. time go over to GPU
                        self.gpu_iq_0[:] = data_0
                        self.gpu_iq_1[:] = data_1
    
                    if 'CALIBRATE' == self.state:
                        self._calibrate_task()
                        # For now, calibration only consumes one pair of sample chunks
                        self.state = 'RUN'
                    elif 'RUN' == self.state:
                        if 'TEST' == self.mode:
                            self.calibrated_delay += self.test_delay_sweep_step
                        visibility = self._run_task()
                        # send cross-correlated data to output buffer
                        self.vis_out.put(visibility)
                elif 'SHUTDOWN' == self.state:
                        break

                self.logger.debug('SDR buffer 0 size: {}'.format(self.buf0.qsize()))
                self.logger.debug('SDR buffer 1 size: {}'.format(self.buf1.qsize()))
                self.logger.debug('Correlation buffer size: {}'.format(self.vis_out.qsize()))


    def _startup_task(self):
        '''
        Initialize sub-processes to start async streaming from SDRs to sample chunk buffers.
        '''
        self._write_metadata()

        self.start_time = time.time() + Correlator._STARTUP_DURATION
        self.logger.info('Cross-correlation will begin at {}'.format(
            time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(self.start_time))))
        # IQ source processes
        # ECM: FIXME:
        # _streaming() is an async function, so this will throw a warning about 
        # not awaiting it, but of course it's being run by asyncio.run, just not
        # here. There might be another way to do this, but this works for now.
        self.logger.debug('Starting streaming subprocesses')
        proc_0 = self._streaming(self.sdr0,
                                 self.buf0,
                                 self.num_samp,
                                 self.start_time,
                                 self.run_time)
        producer_0 = multiprocessing.Process(target=asyncio.run, args=(proc_0,))
        proc_1 = self._streaming(self.sdr1,
                                 self.buf1,
                                 self.num_samp,
                                 self.start_time,
                                 self.run_time)
        producer_1 = multiprocessing.Process(target=asyncio.run, args=(proc_1,))
        producer_0.start()
        producer_1.start()

        self.output_thread.start()
        self.logger.debug('Starting output buffering thread.')

        self.input_thread.start()
        self.logger.debug('Starting to listen for keyboard input.')
        print(LINESEP)
        print('Listening for user input. Input a character & return:')
        print(LINESEP)
        print('c : request delay recalibration')
        print(LINESEP)


    def _calibrate_task(self):
        '''
        Use the data currently in the GPU memory to estimate and store the time delay between channels.
        '''
        # Calibration assumes a noise source w/flat PSD in-band is 
        # used as input
        # Estimate integer and fractional sample delays
        self.logger.debug('Starting calibration')
        self.calibrated_delay = self._estimate_delay(self.gpu_iq_0,
                                                    self.gpu_iq_1,
                                                    self.bandwidth,
                                                    self.frequency)
        self.logger.info('Estimated delay (us): {}'.format(1e6 * self.calibrated_delay))
       

    def _run_task(self):
        '''
        Use the data currently in the GPU memory to calculate the complex cross-correlation.
        '''
        return self._pfb_xcorr()
    
    
    def _pfb_xcorr(self):
        '''
        Consume buffer data to compute PSDs in pairs and then cross-
        correlate them. Use mapped, pinned memory space allocated on the GPU.

        Returns
        -------
        vis :  If mode == 'continuum', float. If mode =='spectrum', cupy.array.
            The result of one complex cross-correlation of the input IQ data.
        '''
        # Threading to take ffts using polyphase filterbank
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as iq_processor:
            future_0 = iq_processor.submit(self._spectrometer_poly, *(cp.array(self.gpu_iq_0), self.ntaps, self.nbins, self.window))
            future_1 = iq_processor.submit(self._spectrometer_poly, *(cp.array(self.gpu_iq_1), self.ntaps, self.nbins, self.window))
            try:
                f0 = future_0.result()
                f1 = future_1.result()
            except Exception as exc:
                print('pfb_spectrometer call generated an exception: {}'.format(exc))
                raise exc
    
        # Apply phase gradient, inspired by 
        # http://www.gmrt.ncra.tifr.res.in/doc/WEBLF/LFRA/node70.html
        # implemented according to Thompson, Moran, Swenson's Interferometry and 
        # Synthesis in Radio Astronoy, 3rd ed., p.364: Fractional Sample Delay 
        # Correction
        freqs = cp.fft.fftfreq(f0.shape[-1], d=1/self.bandwidth) + self.frequency
    
        # Calculate cross-power spectrum and apply FSTC by a phase gradient
        rot = cp.exp(-2j * cp.pi * freqs * (-self.calibrated_delay))
        xpower_spec = f0 * cp.conj(f1 * rot)
        xpower_spec = cp.fft.fftshift(xpower_spec.mean(axis=0))
    
        if self.mode in ['CONTINUUM', 'TEST']: # don't save spectral information
            vis = xpower_spec.mean(axis=0) / self.bandwidth # a visibility amplitude estimate
        else:
            vis = xpower_spec
    
        return vis


    def _spectrometer_poly(self, x, ntaps, n_branches, window): 
        '''
        Polyphase channelize input data using cuSignal polyphase channelizer. Returns
        input array x, channelized into n_branches coefficients

        Parameters
        ----------
        x : cupy.array
            signal of interest
        ntaps : int
            number of polyphase channelizer taps
        n_branches : int
            number of polyphase channelizer branches
        window : cupy.array
            window function coefficients

        Returns
        -------
        channelized : cupy.array
        '''
        # Pad the signal to an even number of chunks
        x = cp.zeros(len(x)+len(x)%n_branches, dtype=np.complex128)[:len(x)] + x
    
        channelized = cusignal.filtering.channelize_poly(x, window, n_branches).T
    
        return channelized
    
    
    def _estimate_delay(self, iq_0, iq_1, rate, fc):
        '''
        Returns delay estimate between channels in seconds.

        Parameters
        ----------
        iq_0, iq_1 : cusignal mapped, pinned array
            GPU memory containing SDR IQ data from channels
        rate : float
            SDR sample rate, samples per second
        fc : float
            SDR center tuning frequency, Hz

                                                                             
        Returns
        -------
        total_delay : float
            The delay estimate between channels in seconds
        '''


        integer_delay = self._estimate_integer_delay(iq_0, iq_1, rate)
        frac_delay = self._estimate_fractional_delay(iq_0, iq_1, integer_delay, rate, fc)
        total_delay = integer_delay + frac_delay

        if 'TEST' == self.mode:
            total_delay -= self.test_delay_offset
        return total_delay
    
    
    def _estimate_integer_delay(self, iq_0, iq_1, rate):
        '''
        Returns delay estimate between channels to the nearest sample division.

        Parameters
        ----------
        iq_0, iq_1 : cusignal mapped, pinned array
            GPU memory containing SDR IQ data from channels
        rate : float
            SDR sample rate, samples per second

        Returns
        -------
        integer_delay : float
            The delay estimate between channels in seconds due to an integer
            number of samples
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
    
    
    def _estimate_fractional_delay(self, iq_0, iq_1, integer_delay, rate, fc):
        '''
        Returns fractional sampling time correction between channels in seconds.
        First corrects integer sample lag to make estimating the fractional lag
        tractable, then finds the slope of the phase of the cross-correlation by
        linear regression to estimate the fractional sample delay.

        Parameters
        ----------
        iq_0, iq_1 : cusignal mapped, pinned array
            GPU memory containing SDR IQ data from channels
        integer_delay : int
            estimated value from _estimate_integer_delay(), sec
        rate : float
            SDR sample rate, samples per second
        fc : float
            SDR center tuning frequency, Hz

        Returns
        -------
        frac_delay, float
            The fractional lag between the argument signals in seconds
        '''
        N = 8192
        f0 = cp.fft.fft(cp.array(iq_0), n=N)
        f1 = cp.fft.fft(cp.array(iq_1), n=N)
        freqs = cp.fft.fftfreq(N, d=1/rate) + fc
    
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
            self.logger.warning('1st-pass delay calibration failed: fractional'
                + ' sample time correction, |{}| > 1/sample rate, {} '.format(frac_delay, 1/rate))
    
        return frac_delay
    
    
    async def _streaming(self, sdr, buf, num_samp, start_time, run_time):
        '''Begins streaming sample chunks from a pyrtlsdr RtlSdr() instance to a
        multiprocess.Queue() buffer at a given time and stops at a given later time.
        
        Parameters
        ----------
        sdr : RtlSdr() instance
            Should already be initialized/tuned to the frequency of interest.
        buf : multiprocessing.Queue() instance
            Buffer to put sample np.arrays in
        num_samp : int
            Number of samples to read async from sdr at a time. 2^18 works well
            for RTL-SDRblog v3 dongles.
        start_time: float
            Time in ms since UNIX epoch to begin streaming async samples. Helps
            multiple streaming processes start closer to the same time.
        run_time: float
            Time in ms since UNIX epoch to end streaming async samples.
        '''
        while time.time() < start_time:
            await asyncio.sleep(1e-9)
        try: 
            async for samples in sdr.stream(format='samples', num_samples_or_bytes=num_samp):
                buf.put(samples)
                if (time.time() - start_time > run_time):
                    break
        except Exception as exc:
            print('_streaming() call generated an exception: {}'.format(exc))
            raise exc
        finally:
            await sdr.stop()
                                                                                                                   
        self.logger.info('Buffering ended at {}'.format(
            time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(time.time()))))


    def _write_metadata(self):
        self.logger.info('Data will be saved to {}.'.format(self.output_file))

        with open(self.output_file, 'w') as file_handle:
            file_handle.write((f'run_time : {self.run_time}, '
                              +f'bandwidth : {self.bandwidth}, '
                              +f'frequency : {self.frequency}, '
                              +f'num_samp : {self.num_samp}, '
                              +f'resolution : {self.nbins}, '
                              +f'gain : {self.gain}, '
                              +f'mode : {self.mode}\n'))
            if 'SPECTRUM' == self.mode:
                # Label frequency bins
                freqs = np.fft.fftshift(np.fft.fftfreq(self.nbins, d=1/self.bandwidth)) + self.frequency
                np.savetxt(file_handle, [freqs], delimiter=',')
            else:
                np.savetxt(file_handle, [])


    def _write_data(self, file_handle):
        '''Buffer cross-correlations to file.'''
        while self.state in ['STARTUP', 'RUN', 'CALIBRATE']:
            while not self.vis_out.empty():
                data = self.vis_out.get_nowait()
                try: # handle case where file gets closed unexpectedly
                    np.savetxt(file_handle, [cp.asnumpy(data)], delimiter=',')
                except ValueError:
                    self.logger.warning('Attempted write to {} after file handle closed.'.format(file_handle))
            # We must balance this thread's need to write to file against the
            # need to perform the cross-correlation task.
            time.sleep(1.0) 


    def post_process(self, raw_output, rate, fc, nfft, num_samp, mode, omit_plot):
        '''
        Handles saving and displaying data.

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

        Returns
        -------
        fname : str
            The filename to which output processed data is written.
        '''
        def visualize(visibilities, rate, fc, nfft, mode, test_delay_sweep_step=0):
            '''Helper that handles plotting 1D continuum data or 2D spectrum
            data with respect to time.'''

            self.logger.info('Plotting data...')

            amp = np.sqrt(np.real(visibilities * np.conj(visibilities)))
            phase = np.angle(visibilities)
            real_part = np.real(visibilities)
            imag_part = np.imag(visibilities)
            
            if mode in ['continuum', 'test']:
                sharey = 'none'
            else:
                sharey = 'all'
                                                                                              
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey=sharey)
            fig.tight_layout()
                                                                                              
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
                                                                                              
            self.logger.info('Plotting complete.')

            plt.show()

            return
    
        if not omit_plot:
            if 'TEST' == self.mode:
                test_delay_sweep_step = self.test_delay_sweep_step
            else:
                test_delay_sweep_step = 0

            visualize(raw_output, rate, fc, nfft, mode, test_delay_sweep_step=test_delay_sweep_step)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='GPU FX correlator accelerated with NVIDIA cuSignal.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--time',
        '-T',
        default=1,
        type=float,
        dest='run_time',
        help='(sec) Total amount of time to run correlator.'
    )
    parser.add_argument('--bandwidth',
        '-B',
        default=2.4e6,
        type=float,
        dest='bandwidth',
        help='(Hz) Receiver bandwidth. For RTL-SDR dongles, this determines sample rate. Applied to both channels.'
    )
    parser.add_argument('--frequency',
        '-F',
        default=1.4204e9,
        type=float,
        dest='fc',
        help='(Hz) Receiver center tuning frequency. Applied to both channels.'
    )
    parser.add_argument('--num_samp',
        '-N',
        default=2**18,
        type=int,
        dest='num_samp',
        help='(int) Number of samples to read from SDR on each call. Large powers of 2 less than 2**18 are recommended for RTL-SDRs.'
    )
    parser.add_argument('--resolution',
        '-R',
        default=2**12,
        type=int,
        dest='nfft',
        help='(int) Number of FFT bins to use in processing and plotting.'
    )
    parser.add_argument('--gain',
        '-G',
        default=49.6,
        type=float,
        dest='gain',
        help='(dB) Tuner gain in Decibels. Tuner gain has an impact on receiver sensitivity and may affect clock stability due to heat generation. Applied to both channels.'
    )
    parser.add_argument('--mode',
        '-M',
        default='spectrum',
        type=str,
        choices=['continuum', 'spectrum', 'test'],
        dest='mode',
        help='(str) Choose one: continuum mode estimates visibility amplitude over time, throwing away phase and frequency information. Spectrum mode keeps complex visibilities. Affects data memory usage, visualization speed, and output file size.'
    )
    parser.add_argument('--omit_plot',
        '-P',
        default=False,
        type=bool,
        dest='omit_plot',
        help='If True, skip post-processing step using matplotlib to visualize recorded data. This may help avoid memory usage problems on low-memory systems. Raw data will still be recorded to a file for further post-processing.'
    )
    parser.add_argument('--loglevel',
        '-L',
        default='INFO',
        type=str,
        choices=['INFO', 'WARNING', 'DEBUG', 'ERROR', 'CRITICAL'],
        dest='loglevel',
        help='Python logging module loglevel.'
    )

    args = parser.parse_args()

    cor = Correlator(run_time=args.run_time,
                     bandwidth=args.bandwidth,
                     frequency=args.fc,
                     num_samp=args.num_samp,
                     nbins=args.nfft,
                     gain=args.gain,
                     mode=args.mode,
                     loglevel=args.loglevel)
    cor.run_state_machine()
    cor.close()

    # Plot results from file
    if args.mode in ['continuum', 'test']:
        skiprows = 1
    else:
        skiprows = 2
    output = np.loadtxt(cor.output_file, dtype=np.complex128, delimiter=',', skiprows=skiprows)
    cor.post_process(output, args.bandwidth, args.fc, args.nfft, args.num_samp, args.mode, args.omit_plot)

