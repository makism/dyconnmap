# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import mutual_information


if __name__ == "__main__":
    data = np.load("/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    x1 = data[0, 0:128]
    x2 = data[0, 0:128]

    mi = mutual_information(x1, x2)

    print(mi)
