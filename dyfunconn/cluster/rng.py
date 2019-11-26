# -*- coding: utf-8 -*-
""" Relational Neural Gas

Relational Neural Gas (RNG) [Hammer2007]_ is a variant of Neural Gas, that allows clustering and mining
data from a pairwise similarity or dissimilarity matrix.


|

-----

.. [Hammer2007] Hammer, B., & Hasenfuss, A. (2007, September). Relational neural gas. In Annual Conference on Artificial Intelligence (pp. 190-204). Springer, Berlin, Heidelberg.
.. [Hasenfuss2008] Hasenfuss, A., Hammer, B., & Rossi, F. (2008, July). Patch Relational Neural Gas–Clustering of Huge Dissimilarity Datasets. In IAPR Workshop on Artificial Neural Networks in Pattern Recognition (pp. 1-12). Springer, Berlin, Heidelberg.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .cluster import BaseCluster


class RelationalNeuralGas(BaseCluster):
    """ Relational Neural Gas


    Parameters
    ----------
    n_protos : int
        The number of prototypes

    iterations : int
        The maximum iterations

    lrate : list of length 2
        The initial and final rearning rates

    n_jobs : int
        Number of parallel jobs (will be passed to scikit-learn))

    metric : string
        One of the following valid options as defined for function http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html.

        Valid options include:

         - euclidean
         - cityblock
         - l1
         - cosine

    rng : object or None
        An object of type numpy.random.RandomState


    Attributes
    ----------
    protos : array-like, shape(n_protos, n_features)
        The prototypical vectors

    """

    def __init__(
        self,
        n_protos=10,
        iterations=100,
        # lrate=[0.3, 0.01],
        lrate=None,
        metric="euclidean",
        rng=None,
    ):
        self.n_protos = n_protos
        self.iterations = iterations
        if lrate is None:
            lrate = [0.3, 0.01]
        self.lrate_i = lrate[0] * n_protos
        self.lrate_f = lrate[1]
        self.lrate = self.lrate_i
        self.protos = None
        self.coeff = None
        self.__multipl = None

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.metric = metric

        self.__symbols = None
        self.__encoding = None

    def fit(self, data):
        """ Fit

        Parameters
        ----------

        data : real array-like, shape(n_samples, n_features)
            Data matrix, each row represents a sample.

        metric : string or None
            One of the following valid options as defined for function http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html.

            Valid options include:

             - euclidean
             - cityblock
             - l1
             - cosine

             If `None` is passed, the matric used for learning the data will be used


        """
        diss_mtx = pairwise_distances(data, metric=self.metric)
        np.fill_diagonal(diss_mtx, 0.0)

        N, _ = np.shape(diss_mtx)

        self.__multipl = np.ones((1, N))
        self.protos = (1.0 / N) * np.ones((self.n_protos, N))

        for iteration in range(1, self.iterations + 1):
            t = iteration / float(self.iterations)

            r_dist = np.float32(self.__rdist(diss_mtx, self.protos))

            I = np.argsort(r_dist, axis=0, kind="heapsort")
            II = np.argsort(I, axis=0, kind="heapsort")
            hl = np.exp(-II / self.lrate)

            self.protos = np.float32(np.ones((1, N)) * self.__multipl * hl)
            tmp = np.sum((np.ones((self.n_protos, 1)) * self.__multipl * hl).T, axis=0)
            tmp = np.reshape(tmp, (self.n_protos, 1))
            tmp = tmp * np.ones((1, N))
            tmp = np.float32(tmp)
            tmp2 = np.divide(self.protos, tmp)
            tmp2 = np.float32(tmp2)
            self.protos = tmp2

            self.lrate = np.float32(
                self.lrate_i * (self.lrate_f / float(self.lrate_i)) ** t
            )

        self.coeff = self.protos
        self.protos = np.matmul(self.coeff, data)

        return self

    @staticmethod
    def __rdist(mtx, coeff):
        """ Relational Distance


        Parameters
        ----------
        mtx : array-list, shape(num_features, num_features)
            Dissimilarity matrix.

        coeff :


        Returns
        -------

        """
        num_protos, num_features = np.shape(coeff)

        distances = np.zeros((num_protos, num_features))
        for i in range(num_protos):
            tmp = np.matmul(coeff[i, :], mtx)
            tmp2 = tmp - (0.5 * np.matmul(tmp, coeff[i, :].T))
            distances[i, :] = tmp2

        return distances
