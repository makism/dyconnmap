# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import glm


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:5, :]

    num_ts, ts_len = np.shape(data)
    pairs = [(r1, r2) for r1 in range(0, num_ts)
             for r2 in range(r1, num_ts)]

    window_size = ts_len / 2.0

    fb_lo = [4.0, 8.0]
    fb_hi = [25.0, 40.0]
    fs = 128.0

    ts, ts_avg = glm(data, fb_lo, fb_hi, fs, pairs=pairs,
                     window_size=window_size)

    print(ts_avg)
