# -*- coding: utf-8 -*-
"""Test for the cluster module."""

import pytest
import scipy as sp
from scipy import io
import numpy as np
from numpy import testing
import sklearn
from sklearn import datasets

import dyconnmap
from dyconnmap import cluster


@pytest.fixture()
def initialize():
    """Prepare random random (with seed)."""
    rng = np.random.RandomState(seed=0)
    data, _ = sklearn.datasets.make_moons(n_samples=1024, noise=0.125, random_state=rng)

    return rng, data


def test_clustering_ng(initialize):
    """Test for the Neural Gas algorithm."""

    # Groundtruth
    result_protos = np.load("groundtruth/cluster/ng_protos.npy")
    result_symbols = np.load("groundtruth/cluster/ng_symbols.npy")

    # Input data
    rng, data = initialize

    # Run
    ng = dyconnmap.cluster.NeuralGas(rng=rng).fit(data)
    protos = ng.protos
    _, symbols = ng.encode(data)

    # Test
    np.testing.assert_array_almost_equal(protos, result_protos)
    np.testing.assert_array_almost_equal(symbols, result_symbols)


def test_clustering_rng(initialize):
    """Test for the Relational Neural Gas algorithm."""

    # Groundtruth
    result_protos = np.load("groundtruth/cluster/rng_protos.npy")
    result_symbols = np.load("groundtruth/cluster/rng_symbols.npy")

    # Data
    rng, data = initialize

    # Run
    reng = dyconnmap.cluster.RelationalNeuralGas(
        n_protos=10, iterations=100, rng=rng
    ).fit(data)
    protos = reng.protos
    _, symbols = reng.encode(data)

    # Test
    np.testing.assert_array_almost_equal(protos, result_protos)
    np.testing.assert_array_almost_equal(symbols, result_symbols)


def test_clustering_mng(initialize):
    """Test for the Merge Neural Gas algorithm."""

    # Groundtruth
    result_protos = np.load("groundtruth/cluster/mng_protos.npy")

    # Data
    rng, data = initialize

    # Run
    protos = dyconnmap.cluster.MergeNeuralGas(rng=rng).fit(data).protos

    # Test
    np.testing.assert_array_almost_equal(protos, result_protos)


def test_clustering_gng(initialize):
    """Test for the Growing Neural Gas algorithm."""

    # Groundtruth
    result_protos = np.load("groundtruth/cluster/gng_protos.npy")
    result_symbols = np.load("groundtruth/cluster/gng_symbols.npy")

    # Data
    rng, data = initialize

    # Run
    gng = dyconnmap.cluster.GrowingNeuralGas(rng=rng)
    gng.fit(data)

    protos = gng.protos
    encoding, symbols = gng.encode(data)

    # Test
    np.testing.assert_array_almost_equal(protos, result_protos)
    np.testing.assert_array_almost_equal(symbols, result_symbols)


def test_clustering_som(initialize):
    """Test for the Self-Organizing Maps algorithm."""

    # Groundtruth
    result_protos = np.load("groundtruth/cluster/som_protos.npy")

    # Data
    rng, data = initialize

    # Run
    protos = dyconnmap.cluster.SOM(grid=(8, 4), rng=rng).fit(data).weights

    # Test
    np.testing.assert_array_almost_equal(protos, result_protos)


def test_clustering_som_umatrix(initialize):
    """Test for the SOM' UMatrix."""

    # Groundtruth
    result_umatrix = np.load("groundtruth/cluster/som_umatrix.npy")

    # Data
    rng, data = initialize

    # Run
    protos = dyconnmap.cluster.SOM(grid=(8, 4), rng=rng).fit(data).weights
    umatrix = dyconnmap.cluster.umatrix(protos)

    # Test
    np.testing.assert_array_almost_equal(umatrix, result_umatrix)
