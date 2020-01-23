# -*- coding: utf-8 -*-
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import coherence, icoherence


if __name__ == "__main__":
    data = np.load(
        "/home/makism/Github/dyconnmap/examples/data/eeg_32chans_10secs.npy")
    data = data[0:5, :]

    csdparams = {'NFFT': 256, 'noverlap': 256 / 2.0}

    coh = coherence(data, [1.0, 4.0], 128.0, **csdparams)
    icoh = icoherence(data, [1.0, 4.0], 128.0)

    print("Coherence: \n", coh)
    print("Imagenary Coherence: \n", icoh)
