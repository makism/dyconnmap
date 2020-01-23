# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing
import scipy
from scipy import io

# dyconnmap
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
    expected = np.load("data/test_graphs_threshold_mean_degree.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    mean_degree_threshold = 5
    binary_mask = threshold_mean_degree(graph, mean_degree_threshold)

    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_mst_mean_degree():
    expected = np.load("data/test_graphs_threshold_mst_mean_degree.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    tree = threshold_mst_mean_degree(graph, 3.6)

    np.testing.assert_array_equal(expected, tree)


def test_graphs_k_core_decomposition():
    expected = np.load("data/test_graphs_k_cores.npy")

    graph = np.load("data/test_graphs_threshold_graph_binary.npy")
    kcores = k_core_decomposition(graph, 10)

    np.testing.assert_array_equal(expected, kcores)


def test_graphs_threshold_shortest_paths():
    expected = np.load("data/test_graphs_threshold_shortest_paths.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    binary_mask = threshold_shortest_paths(graph, treatment=False)

    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_global_cost_efficiency():
    expected = np.load("data/test_graphs_threshold_gce.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    iterations = 50
    binary_mask, _, _, _, _ = threshold_global_cost_efficiency(graph, iterations)

    np.testing.assert_array_equal(expected, binary_mask)


def test_graphs_threshold_omst_global_cost_efficiency():
    # the function is optmized at the 3rd OMST.

    expected = np.load("data/test_graphs_threshold_omst_gce.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    _, CIJtree, _, _, _, _, _, _ = threshold_omst_global_cost_efficiency(
        graph, n_msts=None
    )

    np.testing.assert_array_equal(expected, CIJtree)


def test_graphs_threshold_omst_global_cost_efficiency2():
    # the function is optmized at the 3rd OMST, so it is going to yeild the same results
    # as the exhaustive search

    expected = np.load("data/test_graphs_threshold_omst_gce.npy")

    graph = np.load("data/test_graphs_threshold_graph.npy")
    n_msts = 5
    _, CIJtree, _, _, _, _, _, _ = threshold_omst_global_cost_efficiency(
        graph, n_msts=n_msts
    )

    np.testing.assert_array_equal(expected, CIJtree)


def test_graphs_threshold_eco():
    graph = np.load("data/test_graphs_threshold_graph2.npy")

    filterted, binary, _ = threshold_eco(graph)

    expected = np.load("data/test_graphs_threshold_eco_filtered.npy")
    np.testing.assert_array_equal(expected, filterted)

    expected = np.load("data/test_graphs_threshold_eco_binary.npy")
    np.testing.assert_array_equal(expected, binary)
