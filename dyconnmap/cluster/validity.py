# -*- coding: utf-8 -*-
"""



-----

.. [RayTuri1999] Ray, S., & Turi, R. H. (1999, December). Determination of number of clusters in k-means clustering and application in colour image segmentation. In Proceedings of the 4th international conference on advances in pattern recognition and digital techniques (pp. 137-143).
.. [Davies1979] Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. IEEE transactions on pattern analysis and machine intelligence, (2), 224-227.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>
import numpy as np
import sklearn
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
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
    """
    num_ts, num_samples = np.shape(data)
    clusters = np.unique(labels)
    num_clusters = len(clusters)

    all_barycenters = []

    def __within_distances(label):
        vects = data[np.where(labels == label)]

        barycenter = np.mean(vects, axis=0)
        barycenter = np.reshape(barycenter, [1, -1])

        all_barycenters.append(barycenter)
        D = np.power(pairwise_distances(vects, barycenter, metric="euclidean"), 2)

        return np.sum(D)

    results = list(map(lambda label: __within_distances(label), clusters))

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
    """
    num_ts, num_samples = np.shape(data)

    clusters = np.unique(labels)
    num_clusters = len(clusters)

    within_distances = np.zeros((num_clusters, 1))
    cluster_sizes = np.zeros((num_clusters, 1))
    barycenters = np.zeros((num_clusters, num_samples))
    for i, cluster_id in enumerate(clusters):
        vects = data[np.where(labels == cluster_id)]
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
