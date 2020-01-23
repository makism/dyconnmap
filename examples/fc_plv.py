# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import plv, PLV


if __name__ == "__main__":
    data = np.load("/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:5, ]

    ts, avg = plv(data, [1.0, 4.0], 128.0)
    print(avg)

    # p = PLV([1.0, 4.0], 128.0, pairs=None)
    # a = data[0, :]
    # b = data[1, :]
    #
    # tmp, tmp2 = p.estimate_pair(a, b)
    # print tmp, tmp2
    #
    # tmp, tmp2 = p.estimate(data[0:2, :])
    # print tmp, tmp2
