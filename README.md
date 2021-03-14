# pfb_correlator

pfb_correlator computes the [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) of signals from two USB software-defined radios ([RTL-SDR](https://www.rtl-sdr.com/about-rtl-sdr/)s) with synchronized clocks. This has applications in radio astronomy and remote sensing.

It's written in Python and it has a command line interface:

```bash
python pfb_correlator.py
```

or
```bash
python pfb_correlator.py --time 60 --bandwidth 2.4e6 --frequency 91.3e6 --num_samp 262144 --resolution 4096 --gain 49.6 --mode spectrum
```

which generates output data that is automatically plotted as

<img src="https://github.com/evanmayer/pfb_correlator/blob/master/images/Figure_101.png" width="500">

Installation
------------

Install it by running:

    git clone https://github.com/evanmayer/pfb_correlator.git
    
Software Required
-----------------
- python3
    - numpy
    - matplotlib
    - scipy
- [cupy](https://cupy.dev/)
- NVIDIA RAPIDSAI [cuSignal](https://github.com/rapidsai/cusignal)
- My fork of roger-'s RTL-SDR python interface: [pyrtlsdr](https://github.com/evanmayer/pyrtlsdr)*
- Keenerd's experimental fork of the RTLSDR USB library: [librtlsdr](https://github.com/keenerd/rtl-sdr)*

\* These libraries add functions to disable RTL-SDR phase-locked-loop (PLL) dithering, which is necessary for coherent operation of two receivers that share a clock signal, at the cost of tuning frequency accuracy.

Hardware Required
-----------------
- Computer with NVIDIA CUDA-capable GPU (at least 4 logical processors recommended)
- 2x USB RTL-SDR dongles
    - CLK sharing modification a la: [Juha Vierinen](https://ptrkrysik.github.io/), [Piotr Krysik](https://ptrkrysik.github.io/), or [RTL-SDR Blog v3 Selectable Clock & Expansion Headers](https://www.rtl-sdr.com/rtl-sdr-blog-v-3-dongles-user-guide/)
- 2x SMA antennas
- Any amplifiers/filtering needed to achieve good signal-to-noise ratio of your signal of interest. For Hydrogen line radio astronomy, I use the [Nooelec SAWbird H1+ barebones](https://www.nooelec.com/store/sawbird-h1-barebones.html)
- A white noise source in your band of interest is useful for calibrating out the delay between channels caused by cable lengths and USB sampling startup delay. I use a room temperature 50 Ohm SMA calibration load amplified and filtered for my band of interest, a poor copy of this [CASPER noise source](https://casper.ssl.berkeley.edu/wiki/Noise_sources).

Software Implementation
-----------------------
It is an **FX** architecture complex correlator implemented in software. 
- **F**: a spectrometer is implemented for each channel by a [polyphase filterbank](https://arxiv.org/abs/1607.03579) followed by a [Fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform). 
- **X**: The cross-correlation is computed by vector multiplication of one channel's spectrum by the [complex conjugate](https://en.wikipedia.org/wiki/Complex_conjugate) of the other. Many computationally expensive operations are accelerated by performing them on a GPU using [NVIDIA RAPIDSAI cuSignal](https://medium.com/rapids-ai/gpu-accelerated-signal-processing-with-cusignal-689062a6af8).

Provided the data rate from the USB SDRs is not too high, this correlator runs in real time. This is accomplished by multicore and multithreaded execution to process data from both channels concurrently.

Contribute
----------

- Issue Tracker: https://github.com/evanmayer/pfb_correlator/issues
- Source Code: https://github.com/evanmayer/pfb_correlator

Support
-------

If you are having issues, please let me know.
See the Issues tab of this project.

License
-------

The project is licensed under the GNU General Public License v2.0 license. Make sure you read and understand the terms.

Conflict of Interest Statement
------------------------------
I do not have explicit financial interest in the products or services of NVIDIA, RTL-SDR Blog, Nooelec, or any other equipment manufacturer. This code was developed by Evan Mayer without financial assistance or sponsorship from anyone.

README.md Template from [writethedocs](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/)
