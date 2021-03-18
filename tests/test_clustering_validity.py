# -*- coding: utf-8 -*-
"""Tests for the clustering validity module."""

import scipy as sp
from scipy import io
import numpy as np
from numpy import testing
import sklearn

import dyconnmap
from dyconnmap import cluster


def test_clustering_validity_ray_turi():
    """Test the cluster validity using the Ray-Turi method."""

    # Groundtruth
    expected = np.load("groundtruth/cluster_validity/ray_turi.npy")

    # Data & Run
    data = np.load("sample_data/cluster_validity/data.npy")
    labels = np.load("sample_data/cluster_validity/symbols.npy")
    r = dyconnmap.cluster.ray_turi(data, labels)

    # Test
    np.testing.assert_equal(r, expected)


def test_clustering_validity_ray_turi_with_missing_labels():
    """Test the cluster validity using the Ray-Turi method (with missing labels)."""

    # Groundtruth
    expected = np.load("groundtruth/cluster_validity/ray_turi_with_missing_labels.npy")

    # Data & Run
    data = np.load("sample_data/cluster_validity/data.npy")
    labels = np.load("sample_data/cluster_validity/symbols.npy")
    # remove label "2"
    labels = labels[np.where(labels != 2)]
    r = dyconnmap.cluster.ray_turi(data, labels)

    # Test
    np.testing.assert_equal(r, expected)


def test_clustering_validity_davies_bouldin():
    """Test the cluster validity using the Davies Bouldin method."""

    # Groundtruth
    expected = np.load("groundtruth/cluster_validity/davies_bouldin.npy")

    # Data & Run
    data = np.load("sample_data/cluster_validity/data.npy")
    labels = np.load("sample_data/cluster_validity/symbols.npy")
    r = dyconnmap.cluster.davies_bouldin(data, labels)

    # Test
    np.testing.assert_almost_equal(r, expected)
