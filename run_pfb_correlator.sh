#!/bin/bash
~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 1 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 1 -b 1 &&

# python pfb_correlator.py

python effex/pfb_correlator.py --time 1 --bandwidth 2.4e6 --frequency 1.4204e9 --num_samp 262144 --resolution 4096 --gain 49.6 --mode spectrum

~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 0 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 1 -b 0;
