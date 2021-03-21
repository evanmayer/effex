from context import effex as fx

import cupy as cp
import numpy as np
import pytest


def gen_complex_sinusoid(time, rate, freq, noisy=False):
    num_elem = int(rate * time)

    t = cp.linspace(0, time, num=num_elem)
    omega = 2. * cp.pi * freq
    phi = 0.0
    iq = cp.cos(omega * t + phi) + 1j * cp.sin(omega * t + phi)

    if noisy:
        iq += cp.random.normal(size=num_elem) + 1j * cp.random.normal(size=num_elem)

    return iq


@pytest.mark.parametrize('time', [1e-5, .1])
@pytest.mark.parametrize('rate', [1.28e6, 2.4e6])
@pytest.mark.parametrize('freq', [300e6, 1.4204e9])
@pytest.mark.parametrize('taps', [1,2,3,4,5,6,7,8,9,10])
@pytest.mark.parametrize('branches', [128, 1024, 4096])
def test_spectrometer_poly(time, rate, freq, taps, branches):
    iq = gen_complex_sinusoid(time, rate, freq, noisy=True)

    spec = fx.spectrometer_poly(iq, taps, branches)

    psd = spec * cp.conj(spec)
    mean_psd = psd.mean(axis=0)**2.
    freqs = cp.fft.fftshift(cp.fft.fftfreq(mean_psd.shape[-1], d=1/rate)) + freq

    freq_err_pct = 100. * abs(freqs[cp.argmax(mean_psd)] - freq) / freq
    assert(freq_err_pct < 1.)

    return

