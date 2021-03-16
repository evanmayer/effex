from context import effex as fx

import numpy as np
import cupy as cp


def gen_complex_sinusoid(time, rate, freq, noisy=False):
    num_elem = int(rate * time)

    t = cp.linspace(0, time, num=num_elem)
    omega = 2. * cp.pi * freq
    phi = 0.0
    iq = cp.cos(omega * t + phi) + 1j * cp.sin(omega * t + phi)

    if noisy:
        iq += cp.random.normal(size=num_elem) + 1j * cp.random.normal(size=num_elem)

    return iq


def test_spectrometer_poly():
    rate = 2.4e6
    freq = 1.4e9
    time = 2**18 / rate
    iq = gen_complex_sinusoid(time, rate, freq, noisy=True)

    spec = fx.spectrometer_poly(iq, 4, 4096)
    psd = spec * cp.conj(spec)
    mean_psd = psd.mean(axis=0)
    freqs = cp.fft.fftshift(cp.fft.fftfreq(mean_psd.shape[-1], d=1/rate)) + freq

    freq_err_pct = 100. * abs(freqs[cp.argmax(mean_psd)] - freq) / freq
    assert(freq_err_pct < 1.)

    return


