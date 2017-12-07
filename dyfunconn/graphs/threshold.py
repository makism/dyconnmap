# -*- coding: utf-8 -*-
""" Thresholding schemes


Notes
-----
* This is a direct translation from `Data Driven Topological Filtering of Brain Networks via Orthogonal Minimal Spanning Trees <https://github.com/stdimitr/topological_filtering_networks>'
* Original author is Stravros Dimitriadis <>

|

-----

 """
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import networkx as nx
import scipy
import bct
from scipy import io
from numpy import testing


np.set_printoptions(precision=2, linewidth=80)

import pprint


def k_core_decomposition(mtx, threshold):
    """

    mtx :
        undirected binary

    threshold : int

    """
    imtx = mtx # input matrix ;)

    rows, cols = np.shape(mtx)

    in_degree = np.sum(mtx, 0)
    out_degree = np.sum(mtx, 1)

    degree = bct.degrees_und(mtx);

    for i in range(rows):
        if degree[i] < threshold:
            for l in range(rows):
                imtx[i, l] = 0

        # Recalculate the list of the degrees
        degree = bct.degrees_und(imtx)

    k_cores = np.zeros((rows, 1), dtype=np.int32)

    for i in range(rows):
        if degree[i] > 0:
            k_cores[i] = 1

    return k_cores


def threshold_mst_mean_degree(mtx, avg_degree):
    """

    Parameters
    ----------

    mtx :


    avg_degree :


    """
    N, _ = np.shape(mtx)

    CIJtree = np.zeros((N, N))
    CIJnotintree = mtx


    # Find the number of orthogonal msts according to the desired mean degree.
    num_edges = avg_degree * N
    num_msts = np.int32(np.round(num_edges / (N-1)) + 1)

    # Keep the N-1 connections of the num_msts MSTs.
    mst_conn = np.zeros((num_msts * (N- 1), 2), dtype=np.int32)

    nCIJtree = np.zeros((num_msts, N, N), dtype=np.int32)

    # Repeat rows-2 times
    count = 0
    for no in range(num_msts):

        graph = nx.from_numpy_matrix(1.0 / CIJnotintree)
        mst = nx.minimum_spanning_tree(graph)
        links = list(mst.edges())

        for k in range(N - 1):
            link1 = links[k][0]
            link2 = links[k][1]

            CIJtree[link1, link2] = mtx[link1, link2]
            CIJtree[link2, link1] = mtx[link1, link2]

            mst_conn[count, 0] = link1
            mst_conn[count, 1] = link2

            count += 1

        # Now add connections back, with the total number of added connections
        # determined by the desired 'threshold'
        iCIJtree = np.ones((N, N))
        iCIJtree[np.where(CIJtree != 0.0)] = 0
        CIJnotintree = CIJnotintree * iCIJtree
        nCIJtree[no, :, :] = CIJtree

    degree = np.zeros((1, num_msts * (N - 1)))
    matrix = np.zeros((N, N))

    for no in range(num_msts * (N - 1)):
        idx1 = mst_conn[no, 0]
        idx2 = mst_conn[no, 1]

        matrix[idx1, idx2] = 1
        matrix[idx2, idx1] = 1

        # matrix = double(matrix~=0)
        deg = np.sum(matrix, 0)
        degree[0, no] = np.mean(deg)

    abs_diff = np.abs(degree - avg_degree)
    cutoff = np.argmin(abs_diff)

    CIJtree = np.zeros((N, N))

    for no in range(cutoff):
        idx1 = mst_conn[no, 0]
        idx2 = mst_conn[no, 1]

        CIJtree[idx1, idx2] = mtx[idx1, idx2]
        CIJtree[idx2, idx1] = CIJtree[idx1, idx2]

    ravgdeg = avg_degree - abs_diff

    return CIJtree


def threshold_mean_degree(mtx, threshold_mean_degree):
    """
    Binarize the given weighted matrix using the mean degree.

    Parameters
    ----------

    mtx :


    threshold_mean_degree : int

    """
    binary_mtx = np.zeros_like(mtx, dtype=np.int32)
    rows, cols = np.shape(mtx)

    iter = 100
    step = 1.0 / iter
    thres = 0.0
    thresdeg = np.zeros((iter, 2))

    graph = nx.from_numpy_matrix(mtx)

    for i in range(iter):
        thres += step

        tmp_binary = np.zeros_like(binary_mtx)

        for k in range(cols):
            for l in range(k + 1, rows):
                if mtx[k, l] > thres:
                    tmp_binary[k, l] = 1
                    tmp_binary[l, k] = 1

        degree = bct.degrees_und(tmp_binary)

        thresdeg[i, 0] = np.mean(degree)
        thresdeg[i, 1] = thres

    # find the nearest mean degree to kk
    diff = np.zeros((iter, 1))

    for i in range(iter):
        diff[i] = np.abs(thresdeg[i, 0] - threshold_mean_degree)

    # find the mean degree with the min difference from kk
    r = np.argmin(diff)

    # find the threhold corresponds to the mean degree
    mdegree = 0
    mdegree = thresdeg[r, 0]
    thres = thresdeg[r, 1]

    for k in range(cols):
        for l in range(k + 1, rows):
            if mtx[k, l] > thres:
                binary_mtx[k, l] = 1
                binary_mtx[l, k] = 1

    return binary_mtx


def threshold_shortest_paths(mtx, treatment=False):
    """
    Thresholds a weighted matrix in binary via shortest path identification using
    Dijkstra's algorithm.


    .. [Dimitriadis2010] Dimitriadis, S. I., Laskaris, N. A., Tsirka, V., Vourkas, M., Micheloyannis, S., & Fotopoulos, S. (2010). Tracking brain dynamics via time-dependent network analysis. Journal of neuroscience methods, 193(1), 145-155.


    Parameters
    ----------

    mtx :


    treatment : boolean


    """
    imtx = mtx
    if treatment:
        imtx = 1.0 / mtx
        np.fill_diagonal(imtx, 0.0)

    binary_mtx = np.zeros_like(imtx, dtype=np.int32)

    graph = nx.from_numpy_matrix(imtx)
    paths = dict(nx.all_pairs_dijkstra_path(graph))

    N, _ = np.shape(mtx)

    for x in range(N):
        for y in range(N):
            r_path = paths[x][y]
            num_nodes = len(r_path)

            ind1 = -1
            ind2 = -1
            for m in range(0, num_nodes - 1):
                ind1 = ind1 + 1
                ind2 = ind1 + 1

                binary_mtx[r_path[ind1], r_path[ind2]] = 1
                binary_mtx[r_path[ind2], r_path[ind1]] = 1

    return binary_mtx


def threshold_global_cost_efficiency(mtx, iterations):
    """

    Parameters
    ----------

    mtx :


    iterations : int



    Returns
    -------

    binary_mtx :

    threshold : float
        The threshold that maximizes the global cost efficiency.

    global_cost_eff_max : float
        Global cost efficiency.

    cost_max: float
        Cost of the network at the maximum global cost efficiency

    """
    binary_mtx = np.zeros_like(mtx, dtype=np.int32)

    step = 1.0 / iterations

    thresholds = np.arange(0, 1+step, step)

    N, _ = np.shape(mtx)

    num_connections = (N * (N - 1)) / 2.0

    global_cost_eff = np.zeros((iterations, 1))

    cost = np.zeros((iterations, 1))

    for i in range(iterations):
        tmp_binary = np.zeros_like(binary_mtx)

        for k in range(N):
            for l in range(k+1, N):
                if mtx[k, l] > thresholds[i]:
                    tmp_binary[k, l] = 1
                    tmp_binary[l, k] = 1

        global_eff = bct.efficiency_bin(tmp_binary)

        degree = bct.degrees_und(tmp_binary)
        total_degree = np.sum(degree)

        cost[i] = (0.5 * total_degree) / num_connections
        global_cost_eff[i] = global_eff - cost[i]

    indx_max = np.argmax(global_cost_eff)
    threshold = thresholds[indx_max]

    for k in range(N):
        for l in range(k+1, N):
            if mtx[k, l] >= threshold:
                binary_mtx[k, l] = 1
                binary_mtx[l, k] = 1

    cost_max = cost[indx_max]
    global_cost_eff_max = global_cost_eff[indx_max]
    efficiency = bct.efficiency_bin(binary_mtx)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(cost, global_cost_eff)
    # plt.plot(cost_max, global_cost_eff_max, 'b*', label='Max Global Cost Efficiency')
    # plt.title('Economical small-world network at max Global Cost Efficiency')
    # plt.xlabel('Cost')
    # plt.ylabel('Global Cost Efficiency')
    # plt.legend()
    # plt.show()

    return binary_mtx, threshold, global_cost_eff_max, efficiency


def threshold_omst_global_cost_efficiency(mtx):
    """ Optimizing the formula GE-C via orthogonal MSTs.

    Parameters
    ----------

    mtx :

    """
    imtx = np.copy(mtx)
    imtx_uptril = np.copy(mtx)

    N, _ = np.shape(imtx)

    for k in range(N):
        for l in range(k+1, N):
            imtx_uptril[l, k] = 0.0
    np.fill_diagonal(imtx_uptril, 0.0)

    # Find the number of orthogonal msts according to the desired mean degree
    num_edges = len(np.where(imtx > 0.0)[0])
    num_msts = np.round(num_edges/(N-1)) + 1
    pos_num_msts = np.round(num_edges / (N - 1))

    if num_msts > pos_num_msts:
        num_msts = pos_num_msts

    CIJnotintree = imtx

    # Keep the N-1 connections of the num_msts MSTs
    mst_conn = np.zeros((num_msts * (N-1), 2))

    nCIJtree = np.zeros((num_msts, N, N)) #, dtype=np.int32)
    omst = np.zeros((num_msts, N, N), dtype=np.float32)

    # Repeat N-2 times
    count = 0
    CIJtree = np.zeros((N, N))

    for no in range(num_msts):
        tmp_mtx = 1.0 / CIJnotintree
        # ugly code ~_~
        # graph = nx.from_numpy_matrix(tmp_mtx)
        graph = nx.Graph()
        for x in range(N):
            for y in range(x+1, N):
                graph.add_edge(x, y, weight=tmp_mtx[x][y])
        mst = nx.minimum_spanning_tree(graph)
        links = list(mst.edges())

        new_mst = np.zeros((N, N))
        for k in range(len(links)):
            link1 = links[k][0]
            link2 = links[k][1]

            CIJtree[link1, link2] = imtx[link1, link2]
            CIJtree[link2, link1] = imtx[link1, link2]

            mst_conn[count, 0] = link1
            mst_conn[count, 1] = link2

            new_mst[link1, link2] = imtx[link1, link2]
            new_mst[link2, link1] = imtx[link1, link2]
            count += 1

        iCIJtree = np.ones((N, N))
        iCIJtree[np.where(CIJtree != 0.0)] = 0
        CIJnotintree = CIJnotintree * iCIJtree
        nCIJtree[no, :, :] = CIJtree
        omst[no, :, :] = new_mst

    global_eff_ini = bct.efficiency_wei(imtx_uptril) * 2.0
    cost_ini = np.sum(imtx_uptril[:])

    # Insert the 1st MST
    graph = np.zeros((N, N))
    global_cost_eff = np.zeros((num_msts, 1))
    degrees = np.zeros((num_msts, 1))
    cost = np.zeros((num_msts, 1))

    for k in range(num_msts):
        graph = nCIJtree[k, :, :]

        degree = bct.degrees_und(graph)
        mean_degree = np.mean(degree)
        degrees[k] = mean_degree

        cost[k] = np.sum(graph) / cost_ini

        global_eff = bct.efficiency_wei(graph)
        global_cost_eff[k] = global_eff / global_eff_ini - cost[k]

    # Get the OMST where the formula GE-C is maximized
    indx_max = np.argmax(global_cost_eff)

    # Final output
    degree = degrees[indx_max]
    CIJtree = nCIJtree[indx_max, :, :]
    cost_max = cost[indx_max]
    global_eff = bct.efficiency_wei(1.0 / CIJtree)
    global_cost_eff_max = global_cost_eff[indx_max]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(cost, global_cost_eff)
    # plt.plot(cost_max, global_cost_eff_max, 'b*', label='Max Global Cost Efficiency')
    # plt.title('Economical small-world network at max Global Cost Efficiency')
    # plt.xlabel('Cost')
    # plt.ylabel('Global Cost Efficiency')
    # plt.legend()
    # plt.show()

    return nCIJtree, CIJtree, degree, global_eff, global_cost_eff_max, cost_max

# if __name__ == '__main__':
#     expected = scipy.io.loadmat(
#         '/home/makism/Github/Other/topological_filtering_networks/threshold_schemes/threshold_schemes/gce_binary.mat')['binary']
#
#     graph = np.load('/home/makism/Github/dyfunconn-public/tests/data/test_graphs_threshold_graph.npy')
#     iterations = 50
#     binary_mask = threshold_global_cost_efficiency(graph, iterations)
