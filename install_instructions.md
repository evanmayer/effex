Until I sort out a complete solution for all dependencies, we'll bootstrap off of the excellent work done by the NVIDIA team in sorting out the dependencies for cuSignal, and that will get us 90% of the way there.

1. Follow the cuSignal setup instructions for your platform exactly. I used the [Jetson Nano instructions](https://github.com/rapidsai/cusignal#source-aarch64-jetson-nano-tk1-tx2-xavier-linux-os) to install and build from source.

2. Make a clone of the cusignal env and call it effex-dev, and activate it:

```
conda create --name effex-dev --clone cusignal-dev
conda activate effex-dev
```

3. Clone keenerd's fork of librtlsdr:

```
git clone https://github.com/keenerd/rtl-sdr.git
```

4. To build and install this library, follow the install instructions for the librtlsdr source from [here](https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/) (Linux is at the bottom).

For completeness, on Linux, you'd do:

```
cd rtl-sdr/
mkdir build
cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo make install
sudo cp ../rtl-sdr.rules /etc/udev/rules.d/
sudo ldconfig
```

Ensure any preinstalled DVB-T drivers don't intefere:

```
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee --append /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf
```

5. Clone the evanmayer fork of pyrtlsdr, and install it:

```
git clone https://github.com/evanmayer/pyrtlsdr.git
cd pyrtlsdr
pip install ./
```

At this point, you should have all dependencies necessary to run the application pfb_correlator.py.

If you're using the RTL-SDR.com V3 version of the USB receiver with selectable bias-t for powering devices over coax, see the bias-t util build instructions [here](https://www.rtl-sdr.com/rtl-sdr-blog-v-3-dongles-user-guide/).
