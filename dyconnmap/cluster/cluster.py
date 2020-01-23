# -*- coding: utf-8 -*-
""" Base class for clustring algorithms

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS


class BaseCluster(object):
    """ Base class for clustering alorithms.


    """

    def __init__(self):
        self.rng = None
        self.protos = None

        self.metric = None

        self.__encoding = None
        self.__symbols = None

    def encode(self, data, metric="euclidean", sort=True):
        """ Employ a nearest-neighbor rule to encode the given ``data`` using the codebook.

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

             If `None` is passed, the matric used for learning the data will be used.

        sort : boolean
            Whether or not to sort the symbols using MDS first. Default `True`

        Returns
        -------
        encoded_data : real array-like, shape(n_samples, n_features)
            ``data``, as represented by the prototypes in codebook.
        ts_symbols : list, shape(n_samples, 1)
            A discrete symbolic time series
        """
        sprotos = self.protos

        if sort:
            mds = MDS(1, random_state=self.rng)
            protos_1d = mds.fit_transform(self.protos).ravel()
            sorted_protos_1d = np.argsort(protos_1d)
            sprotos = self.protos[sorted_protos_1d]

        if metric is None:
            metric = self.metric

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", metric=metric).fit(
            sprotos
        )
        _, self.__symbols = nbrs.kneighbors(data)
        self.__encoding = sprotos[self.__symbols]

        return (self.__encoding, self.__symbols)
