#!/bin/bash
~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 1 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 1 &&

python pfb_correlator.py

~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 0 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 0;
