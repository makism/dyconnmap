# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

from dyfunconn.fc import pcoh


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyfunconn/examples/data/eeg_32chans_10secs.npy")
    data = data[0:2, ]

    v = pcoh(data, [1.0, 4.0], 128, n_bins=10)

    print v
