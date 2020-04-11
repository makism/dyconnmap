# -*- coding: utf-8 -*-
"""

"""
import nose
from nose import tools
import scipy as sp
from scipy import io
import numpy as np
from numpy import testing
import sklearn

# dyconnmap
import dyconnmap
from dyconnmap import cluster


def test_clustering_validity_ray_turi():
    data = np.load("data/test_cluster_validity_data.npy")
    labels = np.load("data/test_cluster_validity_symbols.npy")

    r = dyconnmap.cluster.ray_turi(data, labels)

    expected = np.load("data/test_cluster_validity_ray_turi.npy")
    np.testing.assert_equal(r, expected)


def test_clustering_validity_davies_bouldin():
    data = np.load("data/test_cluster_validity_data.npy")
    labels = np.load("data/test_cluster_validity_symbols.npy")

    r = dyconnmap.cluster.davies_bouldin(data, labels)

    expected = np.load("data/test_cluster_validity_davies_bouldin.npy")
    np.testing.assert_equal(r, expected)
