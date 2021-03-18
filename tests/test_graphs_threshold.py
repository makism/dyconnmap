"""Tests for the graphs threshlding module."""

import numpy as np
from numpy import testing
import scipy
from scipy import io

from dyconnmap.graphs import (
    threshold_mean_degree,
    threshold_mst_mean_degree,
    threshold_shortest_paths,
    k_core_decomposition,
    threshold_global_cost_efficiency,
    threshold_omst_global_cost_efficiency,
    threshold_eco,
)


def test_graphs_threshold_mean_degree():
    """Test for graph thresholding based on the mean degree."""

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/mean_degree.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    mean_degree_threshold = 5
    binary_mask = threshold_mean_degree(graph, mean_degree_threshold)

    # Test
    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_mst_mean_degree():
    """Test for graph thresholding based in the MST's mean degree."""

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/mst_mean_degree.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    tree = threshold_mst_mean_degree(graph, 3.6)

    # Test
    np.testing.assert_array_equal(expected, tree)


def test_graphs_k_core_decomposition():
    """Test the k-core decomposition algorithm."""

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/k_cores.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph_binary.npy")

    # Run
    kcores = k_core_decomposition(graph, 10)

    # Test
    np.testing.assert_array_equal(expected, kcores)


def test_graphs_threshold_shortest_paths():
    """Test for graph thresholding base on shortest paths."""

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/shortest_paths.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    binary_mask = threshold_shortest_paths(graph, treatment=False)

    # Test
    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_global_cost_efficiency():
    """Test for graph threshlding using global cost efficiency (GCE)."""

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/gce.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    iterations = 50
    binary_mask, _, _, _, _ = threshold_global_cost_efficiency(graph, iterations)

    # Test
    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_omst_global_cost_efficiency():
    """Test for graph threshlding using global cost efficiency (GCE) on OMSTs (extract all MSTs)."""
    # the function is optmized at the 3rd OMST.

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/omst_gce.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    _, CIJtree, _, _, _, _, _, _ = threshold_omst_global_cost_efficiency(
        graph, n_msts=None
    )

    # Test
    np.testing.assert_array_equal(expected, CIJtree)


def test_graphs_threshold_omst_global_cost_efficiency2():
    """Test for graph threshlding using global cost efficiency (GCE) on OMSTs (extract the first five MSTs)."""
    # the function is optmized at the 3rd OMST, so it is going to yeild the same results
    # as the exhaustive search

    # Groundtruth
    expected = np.load("groundtruth/graphs_threshold/omst_gce.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph.npy")

    # Run
    n_msts = 5
    _, CIJtree, _, _, _, _, _, _ = threshold_omst_global_cost_efficiency(
        graph, n_msts=n_msts
    )

    # Test
    np.testing.assert_array_equal(expected, CIJtree)


def test_graphs_threshold_eco():
    """Test for graph thresholding based on the economical method."""

    # Groundtruth
    expected_filt = np.load("groundtruth/graphs_threshold/eco_filtered.npy")
    expected_bin = np.load("groundtruth/graphs_threshold/eco_binary.npy")

    # Data
    graph = np.load("sample_data/graphs_threshold/graph2.npy")

    # Run
    filterted, binary, _ = threshold_eco(graph)

    # Test
    np.testing.assert_array_equal(expected_filt, filterted)
    np.testing.assert_array_equal(expected_bin, binary)
