#!/bin/bash
~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 1 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 1 -b 1 &&

python effex/effex.py --time 5 --bandwidth 2.4e6 --frequency 1.4204e9 --num_samp 262144 --resolution 4096 --gain 29.7 --mode spectrum --loglevel=INFO

~/github/rtl-sdr-blog/build/src/rtl_biast -d 0 -b 0 &&
~/github/rtl-sdr-blog/build/src/rtl_biast -d 1 -b 0;
