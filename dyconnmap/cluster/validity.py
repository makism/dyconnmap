# -*- coding: utf-8 -*-
"""


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>
import numpy as np
import sklearn
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
import math

import sys

sys.path.append("/home/makism/Github/dyconnmap-public-master/")
import dyconnmap
from dyconnmap.cluster import NeuralGas

import numpy as np
import sklearn
from sklearn import datasets
from collections import defaultdict


def ray_turi(data, labels):
    """ Ray-Turi Index





    Parameters
    ----------
    data : array-like, shape(n_ts, n_samples)
        Input time series

    labels : array-like, shape(n_ts)
        Cluster assignements (labels) per time serie.


    Returns
    -------
    index : float


    -----

    .. [RayTuri1999] Ray, S., & Turi, R. H. (1999, December). Determination of number of clusters in k-means clustering and application in colour image segmentation. In Proceedings of the 4th international conference on advances in pattern recognition and digital techniques (pp. 137-143).
    """
    index = 0.0

    num_ts, num_samples = np.shape(data)
    num_clusters = max(labels) + 1

    all_barycenters = []

    def __within_distances(label):
        vects = data[np.where(labels == label)]

        barycenter = np.mean(vects, axis=0)
        barycenter = np.reshape(barycenter, [1, -1])

        all_barycenters.append(barycenter)
        D = np.power(pairwise_distances(vects, barycenter, metric="euclidean"), 2)

        return np.sum(D)

    results = list(map(lambda label: __within_distances(label), np.unique(labels)))

    all_barycenters = np.array(all_barycenters)
    all_barycenters = np.squeeze(all_barycenters)

    index = np.sum(results)
    min_D = np.power(np.min(distance.pdist(all_barycenters)), 2)

    return np.float32(index) / (num_ts * min_D)


def davies_bouldin(data, labels):
    """ Davies-Bouldin Index



    Parameters
    ----------
    data : array-like, shape(n_ts, n_samples)
        Input time series

    labels : array-like, shape(n_ts)
        Cluster assignements (labels) per time serie.


    Returns
    -------
    index : float


    -----

    .. [Davies1979] Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. IEEE transactions on pattern analysis and machine intelligence, (2), 224-227.
    """
    num_ts, num_samples = np.shape(data)
    num_clusters = np.max(labels) + 1

    within_distances = np.zeros((num_clusters, 1))
    cluster_sizes = np.zeros((num_clusters, 1))
    barycenters = np.zeros((num_clusters, num_samples))
    for i in range(num_clusters):
        vects = data[np.where(labels == i)]
        barycenter = np.mean(vects, axis=0)
        barycenter = np.reshape(barycenter, [1, -1])
        D = pairwise_distances(vects, barycenter)
        D = np.sum(D)

        within_distances[i] = D
        barycenters[i, :] = barycenter
        cluster_sizes[i, :] = len(vects)

    pairs = [(x, y) for x in range(num_clusters) for y in range(num_clusters) if x != y]

    def __between_within_distances(pair):
        x, y = pair
        D1 = within_distances[x]
        D2 = within_distances[y]

        barycenter1 = barycenters[x, :]
        barycenter2 = barycenters[y, :]

        cl1 = cluster_sizes[x,]
        cl2 = cluster_sizes[y,]

        d = (D1 / cl1 + D2 / cl2) / distance.euclidean(barycenter1, barycenter2)

        return d.item()

    results = list(map(lambda pair: (pair[0], __between_within_distances(pair)), pairs))
    results_to_dict = defaultdict(list)
    for k, v in results:
        results_to_dict[k].append(v)

    max_distances = list(map(lambda kv: np.max(kv[1]), results_to_dict.items()))

    return np.sum(max_distances) / num_clusters


# if __name__ == "__main__":
#     rng = np.random.RandomState(seed=0)
#     data, _ = sklearn.datasets.make_moons(n_samples=1024, noise=0.125, random_state=rng)
#
#     ng = NeuralGas(n_protos=10, rng=rng).fit(data)
#     encoding, symbols = ng.encode(data)
#
#     symbols = np.squeeze(symbols)
#     encoding = np.squeeze(encoding)
#
#     np.save("test_cluster_validity_data.npy", data)
#     np.save("test_cluster_validity_symbols.npy", symbols)
#
#     r = ray_turi(data, symbols)
#     print(r)
#
#     np.save("test_cluster_validity_ray_turi.npy", r)
#
#     r = davies_bouldin(data, symbols)
#     print(r)
#
#     np.save("test_cluster_validity_dafies_bouldin.npy", r)
