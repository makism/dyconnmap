# -*- coding: utf-8 -*-
""" Merge NeuralGas

.. [Strickert2003] Strickert, M., & Hammer, B. (2003, September). Neural gas for sequences. In Proceedings of the Workshop on Self-Organizing Maps (WSOM’03) (pp. 53-57).
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import sklearn
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class MergeNeuralGas:
    """ Merge Neural Gas

    Parameters
    ----------

    n_protos: int
        The number of prototypes

    iterations: int
        The maximum iterations

    merge_coeffs: list of length 2
        The merging coefficients

    g: list of length 2


    epsilon: list of length 2
        The initial and final training rates

    lrate: list of length 2
        The initial and final rearning rates

    n_jobs: int
        Number of parallel jobs (will be passed to scikit-learn))

    rng: object
        An object of type numpy.random.RandomState


    Attributes
    ----------
    protos : array-like, shape(n_protos, n_features)
        The prototypical vectors

    distortion : float
        The normalized distortion error
    """

    def __init__(self, n_protos=10, iterations=1024, merge_coeffs=[0.1, 0.0], g=[0.025, 0.025], epsilon=[10, 0.001], lrate=[0.5, 0.005], n_jobs=1, rng=None):
        self.n_protos = n_protos
        self.iterations = iterations
        self.a, self.b = merge_coeffs
        self.g1, self.g2 = g
        self.epsilon_i, self.epsilon_f = epsilon
        self.lrate_i, self.lrate_f = lrate
        self.n_jobs = n_jobs
        self.protos = None
        self.context = None
        self.distortion = 0.0

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.__symbols = None
        self.__encoding = None

    def fit(self, data):
        """

        :param data:
        :return:
        """
        [n_samples, n_obs] = data.shape
        self.protos = data[self.rng.choice(n_samples, self.n_protos),] # w
        self.context = np.zeros(self.protos.shape)                     # c

        ct = np.zeros((1, n_obs))
        wr = ct
        cr = wr

        for iteration in range(self.iterations):
            input = data[self.rng.choice(n_samples, 1),]

            ct = (1 - self.a) * wr + self.b * cr

            t = iteration / float(self.iterations)
            lrate = self.lrate_i * (self.lrate_f / float(self.lrate_i)) ** t
            epsilon = self.epsilon_i * (self.lrate_f / float(self.lrate_i)) ** t

            d = (1 - self.a) * pairwise_distances(input, self.protos) + self.a * pairwise_distances(ct, self.context)
            I = np.argsort(np.argsort(d))

            min_id = np.where(I == 0)[0]

            H = np.exp(-I / epsilon).ravel()

            diff_w = input - self.protos
            diff_c = ct - self.context
            for i in range(self.n_protos):
                self.protos[i, :] += lrate * H[i] * diff_w[i, :]
                self.context[i, :] += lrate * H[i] * diff_c[i, :]

            wr = self.protos[min_id]
            cr = self.context[min_id]

        return self

    def encode(self, data, metric = 'euclidean'):
        """ Employ a nearest-neighbor rule to encode the given ``data`` using the codebook.

        Parameters
        ----------
        data : real array-like, shape(n_samples, n_features)
            Data matrix, each row represents a sample.

        metric : string
            One of the following valid options as defined for function http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html.

            Valid options include:

             - euclidean
             - cityblock
             - l1
             - cosine

        Returns
        -------
        encoded_data : real array-like, shape(n_samples, n_features)
            ``data``, as represented by the prototypes in codebook.
        ts_symbols : list, shape(n_samples, 1)
            A discrete symbolic time series
        """
        nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto', metric = metric).fit(self.protos)
        _, self.__symbols = nbrs.kneighbors(data)
        self.__encoding = self.protos[self.__symbols]

        return (self.__encoding, self.__symbols)
