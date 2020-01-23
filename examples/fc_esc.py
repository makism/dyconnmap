# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=2, linewidth=256)

from dyconnmap.fc import esc


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:5, ]

    escv = esc(data, [4.0, 7.0], [20.0, 45.0], 128)

    print(escv)
