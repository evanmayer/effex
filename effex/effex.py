import argparse
import asyncio
import concurrent.futures
import logging
import multiprocessing
import numpy as np
from queue import Empty, Full
import sys
import threading
import time
import traceback

import cupy as cp
import cusignal

from rtlsdr import RtlSdr
import post_process as pp


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

    _BUFFER_SIZE = int(1e9 // (2**18 * np.dtype(np.complex128).itemsize) // 2)
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
        # Threadsafe queue for child threads to report exceptions
        self.exc_queue = multiprocessing.Queue()

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

        # ---------------------------------------------------------------------
        # USER INPUT
        # ---------------------------------------------------------------------
        self.kbd_queue = multiprocessing.Queue(1)

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
        self.test_delay_sweep_step = crit_delay / 2
        self.test_delay_offset = self.test_delay_sweep_step * 1600


    def _get_kbd(self, queue):
        # Helper function to run in a separate thread and add user input chars to a buffer.
        # Ends listening at end of scheduled run time
        while self.state in ['STARTUP', 'RUN', 'CALIBRATE']:
            queue.put(sys.stdin.read(1))


    def _child_threw_exception(self):
        # Helper to give max possible info about child exceptions and trigger
        # a shutdown if necessary.
        child_threw = False
        if not self.exc_queue.empty():
            exc_formatted = self.exc_queue.get_nowait()
            self.logger.error('Parent process caught child exception:\n{}'.format(exc_formatted))
            child_threw = True
        return child_threw


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
        while True:
            # Check for user input
            if not self.kbd_queue.empty():
                kbd_in = self.kbd_queue.get_nowait()
                if 'c' == kbd_in:
                    self.logger.info('Calibration requested.')
                    self.state = 'CALIBRATE'

            # Warn if buffers fill up, but timeout does not occur
            if self.buf0.qsize() == Correlator._BUFFER_SIZE:
                self.logger.warning('SDR buffer 0 filled up. Data may have been lost!')
            if self.buf1.qsize() == Correlator._BUFFER_SIZE:
                self.logger.warning('SDR buffer 1 filled up. Data may have been lost!')

            # Begin state machine
            if self._child_threw_exception():
                self.logger.debug('Shutting down because child threw exception.')
                self.state = 'SHUTDOWN'

            if 'OFF' == self.state:
                self.state = 'STARTUP'
            elif 'STARTUP' == self.state:
                self._startup_task()
                self.state = 'CALIBRATE'
            # Should we be pulling data?
            elif self.state in ['CALIBRATE', 'RUN']:
                if time.time() < self.start_time:
                    continue
                # Check for data available
                buf0_empty = False
                buf1_empty = False
                get_samples_start_time = time.time()
                try:
                    data_0 = self.buf0.get(block=True, timeout=1)
                except Empty:
                    self.logger.debug('Buffer 0 empty')
                    buf0_empty = True
                try:
                    data_1 = self.buf1.get(block=True, timeout=1)
                except Empty:
                    self.logger.debug('Buffer 1 empty')
                    buf1_empty = True
                get_samples_exec_time = time.time() - get_samples_start_time
                self.logger.debug('Fetching SDR samples from buffers took {} s'.format(get_samples_exec_time))
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
                elif (buf0_empty or buf1_empty):
                    continue
                else:
                    # Complex chunks of IQ data vs. time go over to GPU
                    gpu_samples_transfer_start = time.time()
                    self.gpu_iq_0[:] = data_0
                    self.gpu_iq_1[:] = data_1
                    # Fix DC spike: subtract mean of real and imag components
                    self.gpu_iq_0 = (self.gpu_iq_0.real - self.gpu_iq_0.real.mean()) + 1j * (self.gpu_iq_0.imag - self.gpu_iq_0.imag.mean())
                    self.gpu_iq_1 = (self.gpu_iq_1.real - self.gpu_iq_1.real.mean()) + 1j * (self.gpu_iq_1.imag - self.gpu_iq_1.imag.mean())
                    gpu_samples_transfer_time = time.time() - gpu_samples_transfer_start
                    self.logger.debug('CPU -> GPU memory transfer took {} s'.format(gpu_samples_transfer_time))
    
                if 'CALIBRATE' == self.state:
                    self._calibrate_task()
                    self.state = 'RUN'
                elif 'RUN' == self.state:
                    if self.mode in ['TEST']:
                        self.calibrated_delay += self.test_delay_sweep_step
                    gpu_start_time = time.time()
                    visibility = self._run_task()
                    gpu_exec_time = time.time() - gpu_start_time
                    self.logger.debug('GPU task took {} s'.format(gpu_exec_time))
                    # send cross-correlated data to output buffer
                    self.vis_out.put(visibility)
            elif 'SHUTDOWN' == self.state:
                self.close()
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
            time.strftime('%a, %d %b %Y %H:%M:%S',
                time.localtime(self.start_time))
            )
        )
        # IQ source processes
        # ECM: FIXME:
        # _streaming() is an async function, so this will throw a warning about 
        # not awaiting it, but of course it's being run by asyncio.run, just not
        # here. There might be another way to do this, but this works for now.
        self.logger.debug('Starting streaming subprocesses')
        proc0 = self._streaming(self.sdr0,
            self.buf0,
            self.num_samp,
            self.start_time,
            self.run_time
        )
        producer0 = multiprocessing.Process(target=asyncio.run, args=(proc0,))
        proc1 = self._streaming(self.sdr1,
            self.buf1,
            self.num_samp,
            self.start_time,
            self.run_time
        )
        producer1 = multiprocessing.Process(target=asyncio.run, args=(proc1,))
        producer0.daemon = True
        producer1.daemon = True
        producer0.start()
        producer1.start()

        output_thread = threading.Thread(target=self._write_data,
            daemon=True
        )
        output_thread.start()
        self.logger.debug('Starting output buffering thread.')

        input_thread = threading.Thread(target=self._get_kbd,
            args=(self.kbd_queue,),
            daemon=True
        )
        input_thread.start()
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
        # Take ffts using polyphase filterbank
        f0 = self._spectrometer_poly(cp.array(self.gpu_iq_0), self.ntaps, self.nbins, self.window)
        f1 = self._spectrometer_poly(cp.array(self.gpu_iq_1), self.ntaps, self.nbins, self.window)

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

        integer_delay = self._estimate_delay_parabola(iq_0, iq_1, rate)
        total_delay = integer_delay

        if self.mode in ['TEST']:
            total_delay -= self.test_delay_offset
        return total_delay


    def _estimate_delay_gaussian(self, iq_0, iq_1, rate):
        '''
        Returns subsample delay estimate between channels using a gaussian estimator.

        Parameters
        ----------
        iq_0, iq_1 : cusignal mapped, pinned array
            GPU memory containing SDR IQ data from channels
        rate : float
            SDR sample rate, samples per second

        Returns
        -------
        delay : float
            The delay estimate between channels in seconds
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
    
        # Do subsample refinement with a Gaussian estimator of peak location
        # DOI: 10.1007/978-3-642-58288-2_15 
        imax = int(cp.argmax(cp.abs(xcorr)))
        # TODO: prevent out of bounds errors
        xprev = cp.abs(xcorr[imax - 1])
        xbest = cp.abs(xcorr[imax])
        xnext = cp.abs(xcorr[imax + 1])
        delta_subpixel = 0.5 * (np.log(xprev) - np.log(xnext)) / (np.log(xprev) - 2. * np.log(xbest) + np.log(xnext))
        integer_lag = n - (imax + delta_subpixel)
        integer_delay = integer_lag / rate

        return integer_delay


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
                buf.put(samples, timeout=30)
                if (time.time() - start_time > run_time):
                    break
        except Full as e:
            self.logger.exception('_streaming() call filled up a buffer, and it was not emptied before timeout occurred.', exc_info=sys.exc_info())
            self.exc_queue.put(traceback.format_exc())
            raise e
        finally:
            await sdr.stop()
                                                                                                                   
        self.logger.info('Buffering ended at {}'.format(
            time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime(time.time()))))


    def _write_metadata(self):
        '''Write file header.'''
        self.logger.info('Data will be saved to {}.'.format(self.output_file))

        with open(self.output_file, 'w') as file_handle:
            file_handle.write((f'run_time:{self.run_time},'
                              +f'bandwidth:{self.bandwidth},'
                              +f'frequency:{self.frequency},'
                              +f'num_samp:{self.num_samp},'
                              +f'resolution:{self.nbins},'
                              +f'gain:{self.gain},'
                              +f'mode:{self.mode}\n'))
            if 'SPECTRUM' == self.mode:
                # Label frequency bins
                freqs = np.fft.fftshift(np.fft.fftfreq(self.nbins, d=1/self.bandwidth)) + self.frequency
                np.savetxt(file_handle, [freqs], delimiter=',')
            else:
                np.savetxt(file_handle, [])


    def _write_data(self):
        '''Buffer cross-correlations to file.'''
        with open(self.output_file, 'a') as file_handle:
            while self.state in ['STARTUP', 'RUN', 'CALIBRATE']:
                while not self.vis_out.empty():
                    data = self.vis_out.get_nowait()
                    np.savetxt(file_handle, [cp.asnumpy(data)], delimiter=',')
                # We must balance this thread's need to write to file against the
                # need to perform the cross-correlation task.
                time.sleep(0.1)


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

    # Plot results from file
    if args.mode in ['continuum', 'test']:
        skiprows = 1
    else:
        skiprows = 2

    if 'test' == args.mode:
        test_delay_sweep_step = cor.test_delay_sweep_step
    else:
        test_delay_sweep_step = 0

    # Wait to ensure all file writes have finished
    time.sleep(1.0)

    output = np.loadtxt(cor.output_file, dtype=np.complex128, delimiter=',', skiprows=skiprows)

    pp.post_process(output,
        args.bandwidth,
        args.fc,
        args.nfft,
        args.mode,
        args.omit_plot,
        test_delay_sweep_step=test_delay_sweep_step
    )

