# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import pec


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:2, :]

    fb_lo = [1.0, 4.0]
    fb_hi = [25, 40.0]
    fs = 128

    v = pec(data, fb_lo, fb_hi, fs)

    print(v)
