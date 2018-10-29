# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing

# dyfunconn
from dyfunconn.graphs import (graph_diffusion_distance,
                              variation_information,
                              mutual_information,
                              nodal_global_efficiency,
                              im_distance,
                              spectral_euclidean_distance,
                              spectral_k_distance,
                              laplacian_energy,
                              multilayer_pc_degree, multilayer_pc_strength)


def test_graph_diffusion_distance():
    mtx1 = np.load("data/test_graphs_gdd_mtx1_5x5.npy")
    mtx2 = np.load("data/test_graphs_gdd_mtx2_5x5.npy")

    gdd, t = graph_diffusion_distance(mtx1, mtx2)

    np.testing.assert_almost_equal(gdd, 0.281495413972)
    np.testing.assert_almost_equal(t, 0.443594234747)


def test_variation_information():
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


def test_nodal_global_efficiency():
    inv_mtx = np.load('data/test_graphs_inv_mtx.npy')
    result = np.load('data/test_graphs_nodal_global_efficiency.npy')

    nodal_ge = nodal_global_efficiency(inv_mtx)

    np.testing.assert_array_equal(nodal_ge, result)


def test_im_distance():
    X = np.load('data/test_graphs_spectral_mtx1_10x10.npy')
    Y = np.load('data/test_graphs_spectral_mtx2_10x10.npy')

    result = im_distance(X, Y, bandwidth=1.0)

    np.testing.assert_almost_equal(result, 0.02694184095918512)


def test_im_distance2():
    X = np.load('data/test_graphs_spectral_mtx1_10x10.npy')
    Y = np.load('data/test_graphs_spectral_mtx2_10x10.npy')

    result = im_distance(X, Y, bandwidth=0.1)
    np.testing.assert_almost_equal(result, 0.3210282386813861)


def test_spectral_k_distance():
    X = np.load('data/test_graphs_spectral_mtx1_10x10.npy')
    Y = np.load('data/test_graphs_spectral_mtx2_10x10.npy')

    result = spectral_k_distance(X, Y, k=4)
    np.testing.assert_almost_equal(result, 0.041686646880904045)


def test_spectral_euclidean_distance():
    X = np.load('data/test_graphs_spectral_mtx1_10x10.npy')
    Y = np.load('data/test_graphs_spectral_mtx2_10x10.npy')

    result = spectral_euclidean_distance(X, Y)
    np.testing.assert_almost_equal(result, 0.5364200234417833)


def test_laplacian_energy():
    X = np.load('data/test_graphs_spectral_mtx1_10x10.npy')

    result = laplacian_energy(X)
    np.testing.assert_almost_equal(result, 57.178145779690524)


def test_mpc_strength():
    mlgraph = np.load('data/test_graphs_mpc_mtx.npy')
    mpc_strength = multilayer_pc_strength(mlgraph)

    expected = np.load('/home/makism/Github/dyfunconn/tests/data/test_graphs_mpc_strength.npy')

    np.testing.assert_array_equal(mpc_strength, expected)


def test_mpc_degree():
    mlgraph = np.load('data/test_graphs_mpc_mtx.npy')
    mpc_deg = multilayer_pc_degree(mlgraph)

    expected = np.load('/home/makism/Github/dyfunconn/tests/data/test_graphs_mpc_deg.npy')

    np.testing.assert_array_equal(mpc_deg, expected)
