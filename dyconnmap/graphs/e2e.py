"""Edge-to-Edge Network"""

import numpy as np
import scipy.spatial


def edge_to_edge(dfcgs: np.ndarray) -> np.ndarray:
    """Edge-To-Edge

    Parameters
    ----------
    mlgraph : array-like, shape(n_layers, n_rois, n_rois)
        A multilayer (undirected) graph. Each layer consists of a graph.

    Returns
    -------
    net : array-like
    """
    n_ts, n_rois, n_oirs = np.shape(dfcgs)

    N = n_rois * (n_rois - 1) / 2
    N = np.int32(N)
    count = 0

    ts = np.zeros((N, n_ts))
    for i in range(n_rois):
        for l in range(i + 1, n_rois):
            ts[count, :] = np.squeeze(dfcgs[:, i, l])
            count += 1

    net = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(ts, "euclidean")
    )
    net = np.float32(net)

    return net
