# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import sl


if __name__ == "__main__":
    data = np.load("/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    x = data[0, 0:1024]
    y = data[1, 0:1024]

    siv = sl(x, y, 1, 4)

    print(siv)
