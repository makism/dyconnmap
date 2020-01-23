# -*- coding: utf-8 -*-

from dyconnmap.graphs import variation_information

import numpy as np
import bct


if __name__ == '__main__':
    rng = np.random.RandomState(0)

    a = rng.rand(30, 30)
    np.fill_diagonal(a, 0.0)

    b = rng.rand(30, 30)
    np.fill_diagonal(b, 0.0)

    # In case you get NaNs you have to modify the `gamma` paramter
    Ca, _ = bct.modularity_dir(a, gamma=4.0)
    Cb, _ = bct.modularity_dir(b, gamma=4.0)

    vi, nvi = variation_information(Ca, Cb)

    print(vi)
    print(nvi)
