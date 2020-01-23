# -*- coding: utf-8 -*-

import dyconnmap
from dyconnmap import cluster

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets

rng = np.random.RandomState(seed = 0)
data, _ = sklearn.datasets.make_moons(n_samples = 1024, noise = 0.125, random_state = rng)

som = dyconnmap.cluster.SOM(grid=(8, 4), rng = rng).fit(data)

#
# MUST TEST WITH THE FOLLOWING CODEBOOK
# M = sp.io.loadmat("/home/makism/Development/Matlab/M.mat")['M']
#
cb = som.weights

# np.save("som_weights.npy", som.weights)

# grid_x, grid_y, dim = np.shape(cb)

U = dyconnmap.cluster.umatrix(cb)#, grid_x, grid_y, dim)

# np.save("umatrix.npy", U)

# plt.figure()
# plt.clf()
# plt.scatter(data[:, 0], data[:, 1], lw = 0.0, alpha = 0.25)
#
# for nodes_down in range(som.grid_y - 1):
#     for nodes_left in range(0, som.grid_x):
#         node1 = som.weights[nodes_down, nodes_left, :]
#         node2 = som.weights[nodes_down + 1, nodes_left, :]
#         plt.plot((node1[0], node2[0]), (node1[1], node2[1]), 'k-')
# for nodes_down in range(0, som.grid_y):
#     for nodes_left in range(0, som.grid_x - 1):
#         node1 = som.weights[nodes_down, nodes_left, :]
#         node2 = som.weights[nodes_down, nodes_left + 1, :]
#         plt.plot((node1[0], node2[0]), (node1[1], node2[1]), 'k-')
#
# for nodes_down in range(som.grid_y):
#     for nodes_left in range(som.grid_x):
#         node = som.weights[nodes_down, nodes_left, :]
#         plt.plot(node[0], node[1], marker = 'o', color = 'w', markerfacecolor = 'g', markeredgecolor = 'w',
#                  markersize = 7, antialiased = True)
# plt.axis('equal')
#
# plt.show()

#
# rng = np.random.RandomState(seed=0)
# data, labels = sklearn.datasets.make_moons(n_samples=1024, noise=0.125, random_state=rng)
#
#
# protos1, mng_symbols = dyconnmap.cluster.MergeNeuralGas(rng=rng).fit(data).encode(data)
# mng_symbols = mng_symbols.ravel()
# protos2, ng_symbols = dyconnmap.cluster.NeuralGas(rng=rng).fit(data).encode(data)
# ng_symbols = ng_symbols.ravel()
#
#
# protos1 = protos1.squeeze()
# protos2 = protos2.squeeze()
#
# plt.figure(figsize=(10, 10))
# plt.scatter(data[:, 0], data[:, 1], c=labels, s=10)
# plt.scatter(protos1[:, 0], protos1[:, 1], s=100, marker='o', edgecolors='b')
# plt.scatter(protos2[:, 0], protos2[:, 1], s=100, marker='x', edgecolors='r')
# plt.show()
