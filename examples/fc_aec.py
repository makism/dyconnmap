# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

from dyconnmap.fc import aec


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:1, ]

    corr = aec(data, [1.0, 4.0], [20.0, 45.0], 128)

    print(corr)
