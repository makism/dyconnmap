# -*- coding: utf-8 -*-
""" Growing NeuralGas


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS


class GrowingNeuralGas:
    """ Growing Neural Gas


    Parameters
    ----------

    n_jobs : int
        Number of parallel jobs (will be passed to scikit-learn))

    rng : object or None
        An object of type numpy.random.RandomState


    Attributes
    ----------
    protos : array-like, shape(n_protos, n_features)
        The prototypical vectors

    Notes
    -----

    """

    def __init__(self, n_jobs=1, rng=None):
        self.n_jobs = n_jobs
        self.protos = None

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.__symbols = None
        self.__encoding = None

    def fit(self, data):
        """ Learn data, and construct a vector codebook.

        Parameters
        ----------
        data : real array-like, shape(n_samples, n_features)
            Data matrix, each row represents a sample.

        Returns
        -------
        self : object
            The instance itself
        """

        n_samples, n_features = np.shape(data)

        nodes = np.zeros((2, n_features), dtype=np.float32)
        initial_indices = rng.choice(n_samples, size=2, replace=False)
        nodes[0, :] = data[initial_indices[0], :]
        nodes[1, :] = data[initial_indices[1], :]

        conn_mtx = np.zeros((2, 2), dtype=np.int32)
        conn_mtx[0, 1] = 1

        errors = np.zeros((2))

        ew = 0.01  # adapt winnner
        en = 0.001  # adapt winner's neighbors

        for step in range(1):
            sample_indice = rng.choice(n_samples, size=1, replace=False)
            sample = data[sample_indice, :]

            D = pairwise_distances(sample, nodes, metric="euclidean", n_jobs=1)
            D = np.squeeze(D)
            I = np.argsort(D)
            I = np.squeeze(I)

            bmu1, bmu2 = I[0:2]

            errors[bmu1] += D[bmu1]
            print(errors)

            bmu_node = nodes[bmu1, :]
            new_bmu_pos = bmu_node + ((sample - bmu_node) * ew)

            bmu_neighborhood_indices = conn_mtx[bmu1, :]
            bmu_neighborhood_indices = np.where(conn_mtx[bmu1, :] == 1)[0]
            print(bmu_neighborhood_indices)

        return self


if __name__ == "__main__":

    rng = np.random.RandomState(1)

    n = 400000
    DATA = np.zeros((n, 3))
    i = 0
    for x in np.arange(-1.0, 1.0, 0.0125):
        for y in np.arange(-1.0, 1.0, 0.0125):
            z = x ** 2 + y ** 2
            DATA[i, :] = x, z, y
            i += 1
    DRAW_DATA = np.unique(DATA, axis=0)

    gng = GrowingNeuralGas(n_jobs=1, rng=rng)
    gng.fit(DRAW_DATA)
