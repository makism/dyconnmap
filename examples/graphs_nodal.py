# -*- coding: utf-8 -*-

from dyconnmap.graphs import nodal_global_efficiency
import numpy as np

if __name__ == '__main__':
    rng = np.random.RandomState(0)

    mtx = rng.rand(64, 64)
    mtx_symm = (mtx + mtx.T)/2
    np.fill_diagonal(mtx_symm, 1.0)

    inv_fcg = 1.0 / mtx_symm
    nodal_ge = nodal_global_efficiency(1.0 / mtx_symm)
    print(nodal_ge)
