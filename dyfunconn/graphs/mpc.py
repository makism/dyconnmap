""" Multilayer Participation Coefficient



|

-----

.. [Guillon2016] Guillon, J., Attal, Y., Colliot, O., La Corte, V., Dubois, B., Schwartz, D., ... & Fallani, F. D. V. (2017). Loss of brain inter-frequency hubs in Alzheimer's disease. Scientific reports, 7(1), 10879.
"""
# Author: Stavros Dimitriadis <stidimitriadis@gmail.com>
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import bct
from scipy import linalg as LA


def multilayer_pc_degree(mlgraph):
    """ Multilayer Participation Coefficient (Degree)


    Parameters
    ----------
    mlgraph : array-like, shape(n_layers, n_rois, n_rois)
        A multilayer (undirected) graph. Each layer consists of a graph.


    Returns
    -------
    mpc : array-like
        Participation coefficient based on the degree of the layers' nodes.
    """
    num_layers, num_rois, num_rois = np.shape(mlgraph)

    degrees = np.zeros((num_layers, num_rois))
    for i in range(num_layers):
        a_layer = np.squeeze(mlgraph[i, :, :])
        degrees[i] = bct.degrees_und(a_layer)

    normal_degrees = np.zeros((num_layers, num_rois))
    for i in range(num_rois):
        normal_degrees[:, i] = degrees[:, i] / np.sum(degrees[:, i])

    mpc = np.zeros((num_rois, 1))
    for i in range(num_rois):
        mpc[i] = (np.float32(num_layers) / (num_layers - 1)) * (
            1.0 - np.sum(np.power(normal_degrees[:, i], 2.0))
        )

    return mpc


def multilayer_pc_strength(mlgraph):
    """ Multilayer Participation Coefficient (Strength)


    Parameters
    ----------
    mlgraph : array-like, shape(n_layers, n_rois, n_rois)
        A multilayer (undirected) graph. Each layer consists of a graph.


    Returns
    -------
    mpc : array-like
        Participation coefficient based on the strength of the layers' nodes.
    """
    num_layers, num_rois, num_rois = np.shape(mlgraph)

    strs = np.zeros((num_layers, num_rois))
    for i in range(num_layers):
        for n in range(num_rois):
            strs[i, n] = np.sum(np.ravel(mlgraph[i, n, :]))

    normal_strs = np.zeros((num_layers, num_rois))
    for i in range(num_rois):
        normal_strs[:, i] = strs[:, i] / np.sum(strs[:, i])

    mpc = np.zeros((num_rois, 1))
    for i in range(num_rois):
        mpc[i] = (np.float32(num_layers) / (num_layers - 1)) * (
            1.0 - np.sum(np.power(normal_strs[:, i], 2.0))
        )

    return mpc


def multilayer_pc_gamma(mlgraph):
    """ Multilayer Participation Coefficient method from Guillon et al.



    Parameters
    ----------
    mlgraph : array-like, shape(n_layers, n_rois, n_rois)
        A multilayer graph.


    Returns
    -------
    gamma : array-like, shape(n_layers*n_rois, n_layers*n_rois)
        Returns the original multilayer graph flattened, with the off diagional
        containing the estimated interlayer multilayer participation coefficient.
    """
    num_layers, num_rois, _ = np.shape(mlgraph)

    flattened = LA.block_diag(*mlgraph)
    for s1 in range(num_layers - 1):
        l = list(range(0, num_layers - s1))

        if s1 == num_layers - 2:
            l = [0, num_layers - 1]

        offset = (s1 + 1) * num_rois

        tmp = mlgraph[l, :, :]
        connectivity = __interslice_coupling(tmp)
        # num_conn_layers, _ = np.shape(connectivity)
        values = connectivity.flatten()

        np.fill_diagonal(flattened[offset:], values)
        np.fill_diagonal(flattened[:, offset:], values)

    return flattened


def __interslice_coupling(mlgraph):
    """



    Parameters
    ----------
    mlgraph : array-like, shape(n_layers, n_rois, n_rois)
        A multilayer graph. Each layer consists of a graph.


    Returns
    -------
    gamma : array-like
        Description
    """
    num_layers, num_rois, _ = np.shape(mlgraph)

    gamma = np.zeros((num_layers - 1, num_rois))

    for l1 in range(num_layers - 1):
        for r1 in range(num_rois):
            sum1 = 0.0
            str1 = 0.0
            str2 = 0.0

            for r2 in range(num_rois):
                sum1 += mlgraph[l1, r1, r2] * mlgraph[l1 + 1, r1, r2]

            str1 = np.sum(mlgraph[l1, r1, :])
            str2 = np.sum(mlgraph[l1 + 1, r1, :])

            gamma[l1, r1] = sum1 / np.sqrt(str1 * str2)

    return gamma
