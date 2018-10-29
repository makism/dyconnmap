""" Multilayer Participation Coefficient


"""
# Author: Stavros Dimitriadis <stidimitriadis@gmail.com>
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import bct


# def multilayer_pc(mlgraph):
#     """ Multiparticipation Coefficient (MPC)
# 
#
#     |
#
#     ----
#     .. [Guillon2016] Guillon, J., Attal, Y., Colliot, O., La Corte, V., Dubois, B., Schwartz, D., ... & Fallani, F. D. V. (2017). Loss of brain inter-frequency hubs in Alzheimer's disease. Scientific reports, 7(1), 10879.
#
#     """
#     pass


def multilayer_pc_degree(mlgraph):
    """

    Parameters
    ----------
    mlgraph: matrix, shape(layers, rois, rois)
        A multilayer graph. Each layer consists of a graph.

    Returns
    -------
    mpc: array-like
        Participation coefficient based on the degree of the layers' nodes.
    """
    num_layers, num_rois, num_rois = np.shape(mlgraph)

    degrees = np.zeros((num_layers, num_rois))
    for i in range(num_layers):
        a_layer = np.squeeze(mlgraph[i, :, :])
        degrees[i] = bct.degrees_und(a_layer);

    normal_degrees = np.zeros((num_layers, num_rois))
    for i in range(num_rois):
        normal_degrees[:, i] = degrees[:, i] / np.sum(degrees[:, i])

    mpc = np.zeros((num_rois, 1))
    for i in range(num_rois):
        mpc[i] = (np.float32(num_layers) / (num_layers - 1)) * (1.0- np.sum(np.power(normal_degrees[:, i], 2.0)))

    return mpc


def multilayer_pc_strength(mlgraph):
    """ Multilayer Participation Coefficient (based on strength)

    Parameters
    ----------
    mlgraph: matrix, shape(layers, rois, rois)
        A multilayer graph. Each layer consists of a graph.

    Returns
    -------
    mpc: array-like
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
        mpc[i] = (np.float32(num_layers) / (num_layers - 1)) * (1.0- np.sum(np.power(normal_strs[:, i], 2.0)))

    return mpc
