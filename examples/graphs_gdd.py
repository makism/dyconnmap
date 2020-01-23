# -*- coding: utf-8 -*-

from dyconnmap.graphs import graph_diffusion_distance

import numpy as np


if __name__ == '__main__':
    rng = np.random.RandomState(0)

    a = rng.rand(5, 5)
    a_symm = (a + a.T)/2
    np.fill_diagonal(a_symm, 1.0)

    b = rng.rand(5, 5)
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm, 1.0)

    gdd, t = graph_diffusion_distance(a_symm, b_symm)

    print(gdd)
    print(t)
