# -*- coding: utf-8 -*-
""" Self Organizing Map


:math:`T` is the number of reference prototypes; in :math:`X` the input patterns are stored; :math:`X^\\ast` contains
the approximated patterns as produced by the Nearest Neighbor rule.

Notes
-----
For faster convergence, we can also draw random weights from the given probability distribution :math:`P(t)`

|

-----

.. [Martinetz1991] Martinetz, T., Schulten, K., et al. A "neural-gas" network learns topologies. University of Illinois at Urbana-Champaign, 1991.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np

from .cluster import BaseCluster


class SOM(BaseCluster):
    """ Self Organizing Map

    Parameters
    ----------
    grid : list of length 2
        The X and Y sizes of the grid

    iterations : int
        The maximum iterations

    lrate : float
        The initial rearning rate

    n_jobs : int
        Number of parallel jobs (will be passed to scikit-learn))

    rng : object or None
        An object of type numpy.random.RandomState


    Attributes
    ----------
    protos : array-like, shape(n_protos, n_features)
        The prototypical vectors

    """

    def __init__(self, grid=(10, 10), iterations=1024, lrate=0.1, n_jobs=1, rng=None):
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.grid_y, self.grid_x = grid
        self.iterations = iterations

        self.weights = rng.rand(self.grid_x * self.grid_y, 2)
        self.weights = np.reshape(self.weights, (self.grid_y, self.grid_x, 2))

        self.nodes = np.arange(self.grid_y * self.grid_x)

        self.mapRadius = np.max([self.grid_x, self.grid_y]) / 2.0
        self.timeConstant = float(self.iterations) / float(np.log(self.mapRadius))

        self.lrate_0 = lrate
        self.lrate = self.lrate_0

        self.numIterations = 10000
        self.currentIteration = 0
        self.mapRadius = np.max([self.grid_x, self.grid_y]) / 2.0
        self.startLearningRate = 0.1
        self.timeConstant = float(self.numIterations) / float(np.log(self.mapRadius))
        self.learningRate = self.startLearningRate

    @classmethod
    def findBMU(self, x, y):
        distance = 0.0
        distance += (x[0] - y[0]) * (x[0] - y[0])
        distance += (x[1] - y[1]) * (x[1] - y[1])

        return distance

    def fit(self, data):
        [n_samples, _] = data.shape

        for self.currentIteration in range(self.numIterations):
            learn_sample = data[self.rng.choice(n_samples, 1),]
            learn_sample = learn_sample.squeeze()

            dist = np.inf
            I = None
            for nodes_down in range(self.grid_y):
                for nodes_left in range(self.grid_x):
                    node = self.weights[nodes_down, nodes_left, :]

                    tmp_dist = self.findBMU(learn_sample, node)

                    if tmp_dist < dist:
                        dist = tmp_dist
                        I = (nodes_down, nodes_left)

            # bmu = self.weights[I[0], I[1]]

            self.neighborhoodRadius = self.mapRadius * np.exp(
                float(-self.currentIteration) / self.timeConstant
            )
            for nodes_down in range(self.grid_y):
                for nodes_left in range(self.grid_x):
                    I2 = (nodes_down, nodes_left)
                    distToNodeSquared = self.findBMU(I, I2)

                    widthSquared = self.neighborhoodRadius * self.neighborhoodRadius

                    if distToNodeSquared < widthSquared:
                        w = self.weights[I2[0], I2[1], :]
                        infl = np.exp(-(distToNodeSquared) / (2.0 * widthSquared))
                        w += self.learningRate * infl * (learn_sample - w)

            # Should the following line read: self.learningRate ?
            # learningRate = self.startLearningRate * np.exp(
            # float(-self.currentIteration) / self.numIterations
            # )

        return self
