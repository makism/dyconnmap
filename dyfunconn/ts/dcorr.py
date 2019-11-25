""" Distance Correlation

Notes
-----
Snippet adapted from: https://gist.github.com/Satra/aa3d19a12b74e9ab7941


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
from scipy.spatial.distance import pdist, squareform


def dcorr(x, y):
    """ Distance Correlation


    Parameters
    ----------
    x : array-like, shape(n_samples)
        Input time series.

    y : array-like, shape(N)
        Input time series.


    Returns
    -------
    val : float
        The computed distance correlation.
    """
    lx = len(x)
    ly = len(y)
    if lx != ly:
        raise Exception("")

    X = np.atleast_1d(x)
    Y = np.atleast_1d(y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]

    a = squareform(pdist(X))
    b = squareform(pdist(Y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor
