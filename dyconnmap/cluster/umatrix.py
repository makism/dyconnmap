# -*- coding: utf-8 -*-
""" UMatrix Visualization


"""
import numpy as np


def umatrix(M):
    """

    """
    grid_x, grid_y, dim = np.shape(M)

    ux = 2 * grid_x - 1
    uy = 2 * grid_y - 1
    U = np.zeros((ux, uy))

    for x in range(0, grid_y - 1):
        for y in range(0, grid_x):
            dx = M[y, x] - M[y, x + 1]
            dx = np.sqrt(np.sum(np.power(dx, 2.0)))
            U[2 * y, 2 * x + 1] = dx

    for x in range(0, grid_y):
        for y in range(0, grid_x - 1):
            dy = M[y, x] - M[y + 1, x]
            dy = np.sqrt(np.sum(np.power(dy, 2.0)))
            U[(2 * y) + 1, 2 * x] = dy

    for x in range(0, grid_y - 1):
        for y in range(0, grid_x - 1):
            dz1 = M[y, x] - M[y + 1, x + 1]
            dz1 = np.sqrt(np.sum(np.power(dz1, 2.0)))
            dz2 = M[y + 1, x] - M[y, x + 1]
            dz2 = np.sqrt(np.sum(np.power(dz2, 2.0)))
            offX = (2 * x) + 1
            offY = (2 * y) + 1
            U[offY, offX] = (dz1 + dz2) / (2 * np.sqrt(2.0))

    for x in range(0, grid_y):
        for y in range(0, grid_x):
            offX = 2 * x
            offY = 2 * y
            U[offY, offX] = np.inf

    indices = np.where(U == np.inf)
    for x, y in zip(indices[0], indices[1]):
        center = (x, y)
        __wrap_kernel(center, U)

    return U


def __wrap_kernel(center, mtx):
    """


    """
    x, y = center

    left = None
    right = None
    top = None
    bottom = None

    if x - 1 >= 0:
        left = mtx[x - 1, y]
    if x + 1 < np.shape(mtx)[0]:
        right = mtx[x + 1, y]
    if y - 1 >= 0:
        top = mtx[x, y - 1]
    if y + 1 < np.shape(mtx)[1]:
        bottom = mtx[x, y + 1]

    neighbors = np.array([left, right, top, bottom])
    neighbors = neighbors[neighbors != np.array(None)]

    mtx[center] = np.median(neighbors)
