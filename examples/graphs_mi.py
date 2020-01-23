# -*- coding: utf-8 -*-

from dyconnmap.graphs import mutual_information

import numpy as np
import scipy
from scipy import io
import bct


if __name__ == '__main__':
    rng = np.random.RandomState(0)

    a = rng.rand(30, 30)
    np.fill_diagonal(a, 0.0)

    b = rng.rand(30, 30)
    np.fill_diagonal(b, 0.0)

    # In case you get NaNs you have to modify the `gamma` paramter
    Ca, _ = bct.modularity_dir(a, gamma=1.0)
    Cb, _ = bct.modularity_dir(b, gamma=1.0)

    mi, nmi = mutual_information(Ca, Cb)

    print(mi)
    print(nmi)
