# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import pli


if __name__ == "__main__":
    data = np.load("data/eeg_32chans_10secs.npy")

    print(np.shape(data))

    ts, avg = pli(data, [1.0, 4.0], 128.0, pairs=None)

    print(avg)
