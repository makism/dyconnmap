# -*- coding: utf-8 -*-
""" NeuralGas

Inspired by Self Organizing Maps (SOMs), Neural Gas (*NG*), an unsupervised adaptive algorithm coined by [Martinetz1991]_.
Neural Gas does not assume a preconstructed lattice thus the adaptation cannot be based on the distances between the
neighbor neurons (like in SOMs) because be definition there are no neighbors.

The adaptation-convergence is driven by a stochastic gradient function with a soft-max adaptation rule that minimizes the
average distortion error.

First, we construct a number of neurons ( :math:`N_\\vec{w}` ) with random weights ( :math:`\\vec{w}` ).
Then we train the model by feeding it feature vectors sequentially drawn from the distribution :math:`P(t)`.
When a new feature vector is presented to the model, we sort all neurons' weights (:math:`N_\\vec{w})` based on their
Euclidean distance from :math:`\\vec{x}`. Then, the adaptation if done by:

.. math::
    \\vec{w} \\leftarrow \\vec{w} + [ N_{\\vec{w}} \\cdot  e(t)) \\cdot  h(k)) \\cdot  (\\vec{x} - \\vec{w}) ], \\forall \\vec{w} \\in N_{\\vec{w}}

where,

.. math::
    h(t)=exp{ \\frac{-k}{\\sigma^2}(t) }

    \\sigma^2 = {\\lambda_i(\\frac{\\lambda_T}{\\lambda_0})}^{(\\frac{t}{T_{max}})}

    e(t) = {e_i(\\frac{e_T}{e_0})}^{(\\frac{t}{T_{max}})}


The parameter :math:`\lambda`, governs the initial and final learning rate, while the parameter :math:`e` the training respectively.

After the presentation of a feature vector, increase the itaration counter :math:`t` and repeat
until all desired criteria are met, or :math:`t = T_{max}`.

With these prototypes, we can represent all the input feature vectors :math:`\\vec{x}` using a Nearest Neighbor rule.
The quality of this encoding can measured by the normalized distortion error:

.. math::
    \\frac{ \\sum_{t=1}^T \\left | \\left | X(t) - X^\\ast(t))  \\right | \\right |^2 }{ \\sum_{t=1}^T \\left | \\left | X(t) - \\overline{X})  \\right | \\right |^2 }

where

.. math::
    \\overline{X}` = \\frac{1}{T} \\sum_{t=1}^T{X(t)}

:math:`T` is the number of reference prototypes; in :math:`X` the input patterns are stored; :math:`X^\\ast` contains
the approximated patterns as produced by the Nearest Neighbor rule.

Notes
-----
For faster convergence, we can also draw random weights from the given probability distribution :math:`P(t)`

|

-----

.. [Martinetz1991] Martinetz, T., Schulten, K., et al. A "neural-gas" network learns topologies. University of Illinois at Urbana-Champaign, 1991.
.. [Laskaris2004] Laskaris, N. A., Fotopoulos, S., & Ioannides, A. A. (2004). Mining information from event-related recordings. Signal Processing Magazine, IEEE, 21(3), 66-77.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .cluster import BaseCluster


class NeuralGas(BaseCluster):
    """ Neural Gas


    Parameters
    ----------
    n_protos : int
        The number of prototypes

    iterations : int
        The maximum iterations

    epsilon : list of length 2
        The initial and final training rates

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

    distortion : float
        The normalized distortion error

    Notes
    -----
    Slightly based on *http://webloria.loria.fr/~rougier/downloads/ng.py*
    """

    def __init__(
        self,
        n_protos=10,
        iterations=1024,
        # epsilon=[10, 0.001],
        epsilon=None,
        # lrate=[0.5, 0.005],
        lrate=None,
        n_jobs=1,
        metric="euclidean",
        rng=None,
    ):
        self.n_protos = n_protos
        self.iterations = iterations
        if epsilon is None:
            self.epsilon_i, self.epsilon_f = [10, 0.001]
        if lrate is None:
            self.lrate_i, self.lrate_f = [0.5, 0.005]
        self.n_jobs = n_jobs
        self.protos = None
        self.distortion = 0.0

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.metric = metric

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
        [n_samples, _] = data.shape
        self.protos = data[self.rng.choice(n_samples, self.n_protos),]

        avg_p = np.mean(data, 0).reshape(1, -1)
        dist_from_avg_p = np.sum(pairwise_distances(avg_p, data))

        for iteration in range(self.iterations):
            sample = data[self.rng.choice(n_samples, 1),]

            t = iteration / float(self.iterations)
            lrate = self.lrate_i * (self.lrate_f / float(self.lrate_i)) ** t
            epsilon = self.epsilon_i * (self.epsilon_f / float(self.epsilon_i)) ** t

            D = pairwise_distances(
                sample, self.protos, metric=self.metric, n_jobs=self.n_jobs
            )
            I = np.argsort(np.argsort(D))

            H = np.exp(-I / epsilon).ravel()

            diff = sample - self.protos
            for proto_id in range(self.n_protos):
                self.protos[proto_id, :] += lrate * H[proto_id] * diff[proto_id, :]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.protos)
        distances, _ = nbrs.kneighbors(data)
        self.distortion = np.sum(distances) / dist_from_avg_p

        return self
