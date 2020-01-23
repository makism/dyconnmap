# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import rho_index


if __name__ == "__main__":
    data = np.load("/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:2, ]

    rho_index_mtx = rho_index(data, 5, [8.0, 13.0], 128.0)

    print(rho_index_mtx)
