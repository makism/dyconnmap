# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import pac, PAC, PLV


if __name__ == "__main__":
    data = np.load("/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")

    #
    # Common configuration
    #
    fb = [1.0, 4.0]
    fs = 128
    pairs = None
    estimator = PLV(fb, fs, pairs)

    f_lo = [1.0, 4.0]
    f_hi = [20.0, 30.0]

    #
    # Functional
    #
    cfc, avg = pac(data, f_lo, f_hi, fs, estimator, pairs=None)

    print(avg)

    #
    # Object-oriented
    #
    pac = PAC(f_lo, f_hi, fs, estimator)
    phases, phases_lohi = pac.preprocess(data)
    cfc, avg = pac.estimate(phases, phases_lohi)

    print(avg)
