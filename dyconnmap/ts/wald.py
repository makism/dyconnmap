# -*- coding: utf-8 -*-
""" Wald test


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Tuple, List
import numpy as np
import networkx as nx
from sklearn import metrics


def wald(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, List[List[int]], List[List[float]]]:
    """


    Parameters
    ----------
    x :

    y :


    Returns
    -------
    w : float

    r : float

    edges : array-like

    weights : array-like


    Notes
    -----
    The input time series will be padded with zeros if needed.
    """
    lx = len(x)
    ly = len(y)

    ld = np.abs(lx - ly)
    if lx > ly:
        y = np.lib.pad(y, ((0, ld), (0, 0)), "constant", constant_values=0)
    else:
        x = np.lib.pad(x, ((0, ld), (0, 0)), "constant", constant_values=0)

    [m, _] = np.shape(x)
    [n, _] = np.shape(y)

    N = m + n

    data = np.vstack((x, y))
    dmtx = metrics.pairwise_distances(data, data)

    g = nx.from_numpy_matrix(dmtx)
    mst_g = nx.minimum_spanning_tree(g)

    weighted_edges = mst_g.edges(data=True)

    edges = [we[:-1] for we in weighted_edges]
    weights = [we[::-2][0]["weight"] for we in weighted_edges]

    edges = np.array(edges) + 1

    degree_mst = np.zeros((N))
    for i in range(0, N):
        degree_mst[i] = mst_g.degree(i)

    z = np.sort(edges, axis=1)

    R = np.sum([(z[:, 0] <= m) * (z[:, 1] > m)]) + 1
    ER = (2.0 * m * n) / N + 1

    C = 0.5 * np.sum(degree_mst * (degree_mst - 1))

    varrc = ((2.0 * m * n) / (N * (N - 1.0))) * (
        (2.0 * m * n - N) / N
        + (C - N + 2.0) / ((N - 2.0) * (N - 3.0)) * (N * (N - 1.0) - 4.0 * m * n + 2.0)
    )

    W = (R - ER) / np.sqrt(varrc)

    edges = edges - 1

    return W, R, edges, weights
