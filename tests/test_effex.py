from context import effex as fx

import cupy as cp
import cusignal
import matplotlib.pyplot as plt
import numpy as np
import pytest


cp.random.seed(77777)

def gen_complex_sinusoid(num_samp, rate, freq, noisy=False):
    time = num_samp / rate
    t = cp.linspace(0, time, num=num_samp)
    omega = 2. * cp.pi * freq
    phi = 0.0
    iq = cp.cos(omega * t + phi) + 1j * cp.sin(omega * t + phi)

    if noisy:
        iq += gen_complex_noise(num_samp, rate, scale=.1)

    return iq


def gen_complex_noise(num_samp, rate, scale=.1):
    time = num_samp / rate
    t = cp.linspace(0, time, num=num_samp)
    iq = cp.random.normal(size=num_samp, scale=scale) + 1j * cp.random.normal(size=num_samp, scale=scale)

    return iq


@pytest.mark.parametrize('num_samp', [3+2**12, 2**18])
@pytest.mark.parametrize('rate', [1e6, 2.4e6])
@pytest.mark.parametrize('freq', [2e4, 1e5])
@pytest.mark.parametrize('taps', [4, 32])
@pytest.mark.parametrize('branches', [2048, 4096])
def test_spectrometer_poly(num_samp, rate, freq, taps, branches, plot=False):
    # This is to test that the PFB frontend and subsequent methods of
    # generating a power spectrum are valid, by identifying a frequency
    # component of known value.
    iq = gen_complex_sinusoid(num_samp, rate, freq, noisy=False)

    spec = fx.spectrometer_poly(iq, taps, branches)

    psd = cp.real(spec * cp.conj(spec)).mean(axis=0)

    freqs = cp.fft.fftshift(cp.fft.fftfreq(len(psd), d=1/rate))
    psd = cp.fft.fftshift(psd)

    freq_err_pct = 100. * abs(freqs[cp.argmax(psd)] - freq) / freq
    assert(freq_err_pct < 1.)

    if plot:
        ax = plt.axes()
        ax.plot(cp.asnumpy(freqs), cp.asnumpy(psd))
        plt.show()


@pytest.mark.parametrize('num_samp', [3+2**12, 2**18])
@pytest.mark.parametrize('rate', [2.4e6])
@pytest.mark.parametrize('samp_offset_int', [-2000, -1001, -1, 0, 1, 999, 2000])
def test_estimate_integer_delay(num_samp, rate, samp_offset_int):
    # This is to test that integer-sample delay estimation is functioning by
    # artificially applying a known delay and estimating it like we would with
    # no a priori knowledge
    iq_0 = gen_complex_noise(num_samp, rate)
    iq_1 = cp.roll(iq_0, samp_offset_int)

    est_delay = fx.estimate_integer_delay(iq_0, iq_1, rate)
    est_delay_samples = est_delay * rate

    assert(abs(samp_offset_int - est_delay_samples) < 1e-9)


def test_correlator_init():
    # Test default init
    cor = fx.Correlator()
    assert('OFF' == cor.state)
    assert('SPECTRUM' == cor.mode)
    assert(2.4e6 == cor.bandwidth)
    assert(2**12 == cor.nbins)
    assert(1.4204e9 == cor.frequency)
    assert(49.6 == cor.gain)


def test_bad_run_time():
    with pytest.raises(ValueError):
        cor = fx.Correlator(run_time=0)


def test_bad_bandwidth():
    # Should just print a warning for now
    cor = fx.Correlator(bandwidth=3.0e6)


def test_change_bandwidth():
    cor = fx.Correlator()
    cor.state ='STARTUP'
    cor.state = 'RUN'
    cor.bandwidth = 2.3e6
    assert(2.3e6 == cor.bandwidth)


def test_change_nbins():
    cor = fx.Correlator()
    cor.state = 'STARTUP'
    cor.state = 'RUN'
    cor.nbins = 2**11
    assert(2**11 == cor.nbins)


def test_change_frequency():
    cor = fx.Correlator()
    cor.state = 'STARTUP'
    cor.state = 'RUN'
    cor.frequency = 1.419e9
    assert(1.419e9 == cor.frequency)


def test_change_gain():
    cor = fx.Correlator()
    cor.state = 'STARTUP'
    cor.state = 'RUN'
    cor.gain = 29.7
    assert(29.7 == cor.gain)


def test_correlator_alt_init(): 
    # Test alternate mode init
    cor = fx.Correlator(mode='CONTINUUM')
    assert('OFF' == cor.state)
    assert('CONTINUUM' == cor.mode)


def test_correlator_wrong_init():
    # Test wrong mode init
    with pytest.raises(ValueError):
        cor = fx.Correlator(mode='FOO')


def step_and_assert(sequence):
    # helper for testing correlator state machine
    cor = fx.Correlator()
    for state in sequence:
        cor.state = state
        assert(state == cor.state)


def test_correlator_nominal_state_transitions():
    cor = fx.Correlator()
    nom_sequence = ('STARTUP', 'RUN', 'CALIBRATE', 'RUN', 'DRAIN', 'OFF')
    step_and_assert(nom_sequence)


def test_correlator_early_aborts():
    cor = fx.Correlator()
    seq = ('STARTUP', 'OFF')
    step_and_assert(seq)
    seq = ('STARTUP', 'RUN', 'OFF')
    step_and_assert(seq)
    seq = ('STARTUP', 'RUN', 'CALIBRATE', 'OFF')
    step_and_assert(seq)
    seq = ('STARTUP', 'RUN', 'CALIBRATE', 'RUN', 'OFF')
    step_and_assert(seq)


def test_correlator_bad_state_transitions():
    cor = fx.Correlator()
    # Starting in OFF
    with pytest.raises(fx.Correlator.StateTransitionError):
        # already off
        cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        # can't run without starting up first
        cor.state = 'RUN'

    # Starting in STARTUP
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        # already starting
        cor.state = 'STARTUP'
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        # can't calibrate without running first
        cor.state = 'CALIBRATE'
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        # can't drain if not running
        cor.state = 'DRAIN'

    # Starting in RUN
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        # already running
        cor.state = 'RUN'
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        # already started
        cor.state = 'STARTUP'

    # Starting in CALIBRATE
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        cor.state = 'CALIBRATE'
        # already calibrating
        cor.state = 'CALIBRATE'
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        cor.state = 'CALIBRATE'
        # already started
        cor.state = 'STARTUP'

    # Starting in DRAIN
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        cor.state = 'CALIBRATE'
        cor.state = 'RUN'
        cor.state = 'DRAIN'
        # already draining
        cor.state = 'DRAIN'
    cor.state = 'OFF'
    with pytest.raises(fx.Correlator.StateTransitionError):
        cor.state = 'STARTUP'
        cor.state = 'RUN'
        cor.state = 'CALIBRATE'
        cor.state = 'RUN'
        cor.state = 'DRAIN'
        # already started
        cor.state = 'STARTUP'


def example_fstc(num_samp, rate, samp_offset_int):
    # This is to test the method of applying a phase shift to a signal by
    # multiplying its Fourier transform by a ifrequency-dependent complex
    # exponential.
    fc = 1e5
    iq_0 = gen_complex_sinusoid(num_samp, rate, fc, noisy=True)

    # First, pad by length of signals
    n = len(iq_0)                                         
    iq_0_padded = cp.zeros(2 * n, dtype=cp.complex128)
    iq_1_padded = cp.zeros(2 * n, dtype=cp.complex128)
    iq_0_padded[0:n] += cp.array(iq_0)
    iq_1_padded[0:n] += cp.array(iq_0)
    iq_1_shifted = cp.roll(iq_0_padded, samp_offset_int)

    est_lag = fx.estimate_integer_lag(iq_0_padded, iq_1_shifted)

    f0 = cp.fft.fft(iq_0_padded)
    f1 = cp.fft.fft(iq_1_shifted)
    freqs = cp.fft.fftfreq(len(f1), d=1/rate) + fc

    rot = cp.exp(-1j * cp.pi * freqs * (-est_lag) / rate)

    f1_shift = f1 * rot

    iq_1_unshifted = cp.fft.ifft(f1_shift)

    # At this point, we should have un-done the sample shift caused by the
    # roll() func
    ax = plt.axes()
    ax.plot(cp.asnumpy(cp.real(iq_0_padded)), label='Re[orig]')
    ax.plot(cp.asnumpy(cp.imag(iq_0_padded)), label='Im[orig]')
    ax.plot(cp.asnumpy(cp.real(iq_1_unshifted)), alpha=0.5, label='Re[realigned]')
    ax.plot(cp.asnumpy(cp.imag(iq_1_unshifted)), alpha=0.5, label='Im[realigned]')
    ax.set_xlabel('# samples')
    ax.set_ylabel('Amplitude')
    ax.set_title('Time-realigned with Freq Domain Rotation')
    ax.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    example_fstc(2**14, 2.4e6, -100)

