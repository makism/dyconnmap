# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3, linewidth=256)

import matplotlib.pyplot as plt

from dyconnmap import sim_models


if __name__ ==  "__main__":
    fs = 128
    t = 1
    epochs = 10
    min_fs = 1.0
    max_fs = 4.0
    sim_eeg = sim_models.maniken(fs * t, epochs, fs, min_fs, max_fs)

    plt.figure()
    plt.plot(sim_eeg)
    plt.xlim(0.0, fs * t)
    plt.show()
