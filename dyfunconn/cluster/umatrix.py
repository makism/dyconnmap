import numpy as np
import sys
import scipy as sp
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.path import Path
import matplotlib.patches as patches
import sklearn
from sklearn import preprocessing
import matplotlib.cm as cm

np.set_printoptions(precision=3, linewidth=256)


def make_hexagon():
    verts = [
        (0.5, 0.), (0.5, 1.),
        (1., 1.5),
        (1.5, 1), (1.5, 0),
        (1., -0.5),
        (0., -0.5),
    ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
    path = Path(verts, codes)

    return path

def wrap_kernel(center, mtx):
    x, y = center

    left = None
    right = None
    top = None
    bottom = None

    if x-1 >= 0:
        left = mtx[x-1, y]
    if x+1 < np.shape(mtx)[0]:
        right= mtx[x+1, y]
    if y-1 >= 0:
        top  = mtx[x, y-1]
    if y+1 < np.shape(mtx)[1]:
        bottom=mtx[x, y+1]

    neighbors = np.array([left, right, top, bottom])
    neighbors = neighbors[neighbors != np.array(None)]

    mtx[center] = np.median(neighbors)

def umatrix(M):
    # S = sp.io.loadmat("/home/makism/Development/Matlab/S.mat")['S']
    # cb= S['codebook']
    #
    # M = sp.io.loadmat("/home/makism/Development/Matlab/M.mat")['M']

    grid_x, grid_y, dim = np.shape(M) #(grid_x, grid_y) #(16, 4)
    # dim = 4

    ux = 2 * grid_x - 1
    uy = 2 * grid_y - 1
    U = np.zeros((ux, uy))

    for x in range(0, grid_y - 1):
        for y in range(0, grid_x):
            dx = M[y, x] - M[y, x + 1]
            dx = np.sqrt(np.sum(np.power(dx, 2.0)))
            U[2*y, 2*x+1] = dx

    for x in range(0, grid_y):
        for y in range(0, grid_x - 1):
            dy = M[y, x] - M[y+1, x]
            dy = np.sqrt(np.sum(np.power(dy, 2.0)))
            U[(2*y)+1, 2*x] = dy

    for x in range(0, grid_y - 1):
        for y in range(0, grid_x - 1):
            dz1 = M[y, x] - M[y + 1, x + 1]
            dz1 = np.sqrt(np.sum(np.power(dz1, 2.0)))
            dz2 = M[y + 1, x] - M[y, x + 1]
            dz2 = np.sqrt(np.sum(np.power(dz2, 2.0)))
            offX = (2*x) + 1
            offY = (2*y) + 1
            U[offY, offX] = (dz1 + dz2) / (2 * np.sqrt(2.0))

    for x in range(0, grid_y):
        for y in range(0, grid_x):
            offX = (2*x)
            offY = (2*y)
            U[offY, offX] = np.inf

    indices = np.where(U == np.inf)
    for x,y in zip(indices[0], indices[1]):
        center = (x, y)
        wrap_kernel(center, U)

    # np.save("U.npy", U)

    # min = np.min(np.min(U))
    # max = np.max(np.max(U))
    #
    # normU = (U - min) / max
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for row in range(ux):
    #     for col in range(uy):
    #         xtra_col = 0
    #         if row % 2 == 0:
    #             xtra_col = 0.5
    #         x, y = (col + (col * 0.0) + xtra_col, row + (row * 0.5))
    #
    #         path = make_hexagon()
    #         trans = transforms.Affine2D().translate(x, y) + ax.transData
    #         color = cm.jet(normU[row, col])
    #         patch = patches.PathPatch(path, facecolor=color, lw=1, edgecolor='white', transform=trans)
    #         ax.add_patch(patch)
    # plt.axis('auto')
    # plt.show()

    return U
