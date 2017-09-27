# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing

# dyfunconn
from dyfunconn.graphs import (graph_diffusion_distance,
                              variation_information,
                              mutual_information)


def test_graph_diffusion_distance():
    mtx1 = np.load("data/test_graphs_gdd_mtx1_5x5.npy")
    mtx2 = np.load("data/test_graphs_gdd_mtx2_5x5.npy")

    gdd, t = graph_diffusion_distance(mtx1, mtx2)

    np.testing.assert_almost_equal(gdd, 0.281495413972)
    np.testing.assert_almost_equal(t, 0.443594234747)


def test_variation_information():
    # mtx1 = np.load("data/test_graphs_vi_mtx1_30x30.npy")
    # mtx2 = np.load("data/test_graphs_vi_mtx2_30x30.npy")

    # Ca, Qa = bct.modularity_dir(mtx1, 2.0)
    # Cb, Qb = bct.modularity_dir(mtx2, 2.0)

    Ca = np.load('data/test_graphs_vi_mtx1_30x30_comm_struct.npy')
    Cb = np.load('data/test_graphs_vi_mtx2_30x30_comm_struct.npy')

    vi, nvi = variation_information(Ca, Cb)

    np.testing.assert_almost_equal(vi, 0.735803959669)
    np.testing.assert_almost_equal(nvi, 0.216336741771)


def test_mutual_information():
    Ca = np.load('data/test_graphs_vi_mtx1_30x30_comm_struct.npy')
    Cb = np.load('data/test_graphs_vi_mtx2_30x30_comm_struct.npy')

    mi, nmi = mutual_information(Ca, Cb)

    np.testing.assert_almost_equal(mi, 0.0115297151096)
    np.testing.assert_almost_equal(nmi, 0.0303868002153)
