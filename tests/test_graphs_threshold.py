# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing

# dyfunconn
from dyfunconn.graphs import (threshold_mean_degree,
                              threshold_mst_mean_degree,
                              threshold_shortest_paths,
                              threshold_global_cost_efficiency,
                              threshold_omst_global_cost_efficiency)


def test_graph_threshold_mean_degree():
    pass


def test_graph_threshold_mst_mean_degree():
    pass


def test_graph_threshold_shortest_paths():
    pass


def test_graph_threshold_global_cost_efficiency():
    pass


def test_graph_threshold_omst_global_cost_efficiency():
    mtx1 = np.load("data/test_graphs_gdd_mtx1_5x5.npy")
    mtx2 = np.load("data/test_graphs_gdd_mtx2_5x5.npy")

    gdd, t = graph_diffusion_distance(mtx1, mtx2)

    np.testing.assert_almost_equal(gdd, 0.281495413972)
    np.testing.assert_almost_equal(t, 0.443594234747)
