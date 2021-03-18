# -*- coding: utf-8 -*-
"""Tests for the graphs module."""

import numpy as np
from numpy import testing

from dyconnmap.graphs import (
    graph_diffusion_distance,
    variation_information,
    mutual_information,
    nodal_global_efficiency,
    im_distance,
    spectral_euclidean_distance,
    spectral_k_distance,
    laplacian_energy,
    multilayer_pc_strength,
    multilayer_pc_degree,
    multilayer_pc_gamma,
    edge_to_edge,
)


def test_graph_diffusion_distance():
    """Test the Graph Diffusion Distance algorithm."""

    # Data
    mtx1 = np.load("sample_data/graphs/gdd_mtx1_5x5.npy")
    mtx2 = np.load("sample_data/graphs/gdd_mtx2_5x5.npy")

    # Run
    gdd, t = graph_diffusion_distance(mtx1, mtx2)

    # Test against the groundtruth
    np.testing.assert_almost_equal(gdd, 0.281495413972)
    np.testing.assert_almost_equal(t, 0.443594234747)


def test_variation_information():
    """Test the Variation of Information algorithm."""

    # Data
    Ca = np.load("sample_data/graphs/vi_mtx1_30x30_comm_struct.npy")
    Cb = np.load("sample_data/graphs/vi_mtx2_30x30_comm_struct.npy")

    # Run
    vi, nvi = variation_information(Ca, Cb)

    # Test against the groundtruth
    np.testing.assert_almost_equal(vi, 0.735803959669)
    np.testing.assert_almost_equal(nvi, 0.216336741771)


def test_mutual_information():
    """Test the Mutual Information algorithm."""

    # Data
    Ca = np.load("sample_data/graphs/vi_mtx1_30x30_comm_struct.npy")
    Cb = np.load("sample_data/graphs/vi_mtx2_30x30_comm_struct.npy")

    # Run
    mi, nmi = mutual_information(Ca, Cb)

    # Test against the groundtruth
    np.testing.assert_almost_equal(mi, 0.0115297151096)
    np.testing.assert_almost_equal(nmi, 0.0303868002153)


def test_nodal_global_efficiency():
    """Test the Nodal Global Efficiency."""

    # Groundtruth
    result = np.load("groundtruth/graphs/nodal_global_efficiency.npy")
    result = result.reshape([1, -1])

    # Data
    inv_mtx = np.load("sample_data/graphs/inv_mtx.npy")

    # Run
    nodal_ge = nodal_global_efficiency(inv_mtx)
    nodal_ge = nodal_ge.reshape([1, -1])

    # Run
    np.testing.assert_array_equal(nodal_ge, result)


def test_im_distance():
    """Test the Ipsen-Mikhailov Distance algorithm."""

    # Data
    X = np.load("sample_data/graphs/spectral_mtx1_10x10.npy")
    Y = np.load("sample_data/graphs/spectral_mtx2_10x10.npy")

    # Run
    result = im_distance(X, Y, bandwidth=1.0)

    # Test against the groundtruth
    np.testing.assert_almost_equal(result, 0.02694184095918512)


def test_im_distance2():
    """Test the Ipsen-Mikhailov Distance algorithm (with a given bandwidth parameter)."""

    # Data
    X = np.load("sample_data/graphs/spectral_mtx1_10x10.npy")
    Y = np.load("sample_data/graphs/spectral_mtx2_10x10.npy")

    # Run
    result = im_distance(X, Y, bandwidth=0.1)

    # Test against the groundtruth
    np.testing.assert_almost_equal(result, 0.3210282386813861)


def test_spectral_k_distance():
    """Test the K-Spectral Distance algorithm."""

    # Data
    X = np.load("sample_data/graphs/spectral_mtx1_10x10.npy")
    Y = np.load("sample_data/graphs/spectral_mtx2_10x10.npy")

    # Run
    result = spectral_k_distance(X, Y, k=4)

    # Test against the groundtruth
    np.testing.assert_almost_equal(result, 0.041686646880904045)


def test_spectral_euclidean_distance():
    """Test the Spectral Euclidean Distance algorithm."""

    # Data
    X = np.load("sample_data/graphs/spectral_mtx1_10x10.npy")
    Y = np.load("sample_data/graphs/spectral_mtx2_10x10.npy")

    # Run
    result = spectral_euclidean_distance(X, Y)

    # Test against the groundtruth
    np.testing.assert_almost_equal(result, 0.5364200234417833)


def test_laplacian_energy():
    """Test the Laplacian Energy algorithm."""

    # Data
    X = np.load("sample_data/graphs/spectral_mtx1_10x10.npy")

    # Run
    result = laplacian_energy(X)

    # Test against the groundtruth
    np.testing.assert_almost_equal(result, 57.178145779690524)


def test_mpc_strength_und():
    """Test the Multiparticipation Coefficient based on the Undirected Strength."""

    # Groundtruth
    expected = np.load("groundtruth/graphs/mlgraph_pc_strength.npy")

    # Data
    X = np.load("sample_data/graphs/mlgraph.npy")

    # Run
    result = multilayer_pc_strength(X)

    # Test
    np.testing.assert_equal(result, expected)


def test_mpc_degree_und():
    """Test the Multiparticipation Coefficient based on the Degree."""

    # Groundtruth
    expected = np.load("groundtruth/graphs/mlgraph_mst_pc_degree.npy")

    # Data
    X = np.load("sample_data/graphs/mlgraph_mst.npy")

    # Run
    result = multilayer_pc_degree(X)

    # Test
    np.testing.assert_equal(result, expected)


def test_mpc_gamma():
    """Test the Multiparticipation Coefficient based on the Gamma connectivity."""

    # Groundtruth
    expected = np.load("groundtruth/graphs/mlgraph_pc_gamma.npy")

    # Data
    X = np.load("sample_data/graphs/mlgraph.npy")

    # Run
    gamma = multilayer_pc_gamma(X)

    # Test
    np.testing.assert_equal(gamma, expected)


def test_e2e():
    """Test the Edge-to-Edge multilayer network construction."""

    # Groundtruth
    expected = np.load("groundtruth/graphs/mlgraph_e2e_expected_result.npy")

    # Data
    dfcgs = np.load("sample_data/graphs/mlgraph_e2e.npy")

    # Run
    result = edge_to_edge(dfcgs)
    result = np.float32(result)

    # Test
    np.testing.assert_array_equal(expected, result)
