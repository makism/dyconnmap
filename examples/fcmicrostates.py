# -*- coding: utf-8 -*-
""" A sample pipeline of computing Functional Connectivity Microstates

"""

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.fc import plv
from dyconnmap import tvfcg
from dyconnmap.cluster import NeuralGas


if __name__ == "__main__":
    data = np.load("data/eeg_32chans_10secs.npy")

    print(np.shape(data))
