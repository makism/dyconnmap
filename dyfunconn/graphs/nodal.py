""" Nodal network features


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>
# Author: Stravros Dimitriadis <stidimitriadis@gmail.com>

import numpy as np
import networkx as nx


def nodal_global_efficiency(mtx):
    """ Nodal Global Efficiency


    Parameters
    ----------
    mtx : array-like, shape(N, N)
        Symmetric, weighted and undirected connectivity matrix.


    Returns
    -------
    nodal_ge : array-like, shape(N, 1)
        The computed nodal global efficiency.
    """
    num_rows, num_cols = np.shape(mtx)

    if num_rows != num_cols:
        raise Exception("Given matrix is not square.")

    graph = nx.from_numpy_matrix(mtx)
    distances = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(graph)

    inv_distances = 1.0 / distances
    np.fill_diagonal(inv_distances, 0.0)

    nodal_ge = np.sum(inv_distances, axis=0) / (num_rows - 1.0)

    return nodal_ge
