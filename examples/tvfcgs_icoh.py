# -*- coding: utf-8 -*-
import time

import numpy as np
np.set_printoptions(precision=3, linewidth=160, suppress=True)

from dyconnmap.fc import ICoherence
from dyconnmap import tvfcg


if __name__ == '__main__':
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")

    n_channels, n_samples = np.shape(data)

    fb_lo = [1.0, 4.0]
    cc = 3.0
    fs = 128
    step = 5
    #
    # windows, window_length = tvfcg_compute_windows(
    #     data, fb_lo, fs, cc, step)
    #
    # pairs = [(win_id, (win_id * step), window_length + (win_id * step), c1, c2)
    #          for win_id in range(windows)
    #          for c1 in xrange(0, n_channels)
    #          for c2 in xrange(c1, n_channels)
    #          if c1 != c2
    #          ]
    #
    # for pair in pairs:
    #     win_id, start, end, c1, c2 = pair

        # slice1 = pp_data1[c1, ..., start:end]
        # slice2 = pp_data2[c2, ..., start:end]
        # slice, _ = estimator(slice1, slice2)
        #
        # fcgs[win_id, c1, c2] = np.mean(slice)
    # return fcgs
