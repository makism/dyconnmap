""" Spectral-K Distance

Given two graphs :math:`G` and :math:`H`, we can use their :math:`k` largest positive
eigenvalues of their Laplacian counterparts to compute their distance.

.. math::
    d(G, H) = \\left\{\\begin{matrix} \\sqrt{\\frac{ \\sum_{i=1}^k{(g_i - h_i)^2} }{ \\sum_{i=1}^l{g_i^2} }} & ,\\sum_{i=1}^l{g_i^2} \\leq \\sum_{j=1}^l{h_j^2}
    \\\\
    \\sqrt{\\frac{ \\sum_{i=1}^k{(g_i - h_i)^2} }{ \\sum_{j=1}^l{g_i^2} }} & , \\sum_{i=1}^l{g_i^2} > \\sum_{j=1}^l{h_j^2}
    \\end{matrix}\\right.

Where :math:`g` and :math:`h` denote the spectrums of the Laplacian matrices.

This measure is non-negative, separated, symmetric and it satisfies the triangle
inequality.


|


----

.. [Jakobson2000] Jakobson, D., & Rivin, I. (2000). Extremal metrics on graphs I. arXiv preprint math/0001169.
.. [Pincombe2007] Pincombe, B. (2007). Detecting changes in time series of network graphs using minimum mean squared error and cumulative summation. ANZIAM Journal, 48, 450-473.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import scipy


def spectral_k_distance(X, Y, k):
    """ Spectral-K Distance

    Use the largest :math:`k` eigenvalues of the given graphs to compute the
    distance between them.


    Parameters
    ----------
    X : array-like, shape(N, N)
        A weighted matrix.

    Y : array-like, shape(N, N)
        A weighted matrixY

    k : int
        Largest `k` eigenvalues to use.


    Returns
    -------
    distance : float
        Estimated distance based on selected largest eigenvalues.
    """
    l_mtx_a = scipy.sparse.csgraph.laplacian(X, normed=False)
    l_mtx_b = scipy.sparse.csgraph.laplacian(Y, normed=False)

    w_a, _ = scipy.sparse.linalg.eigs(l_mtx_a, k=k)
    w_a = np.real(w_a)
    w_a = np.sort(w_a)[::-1]

    w_b, _ = scipy.sparse.linalg.eigs(l_mtx_b, k=k)
    w_b = np.real(w_b)
    w_b = np.sort(w_b)[::-1]

    num = np.sum(np.power(w_a - w_b, 2))
    denom = np.min((np.sum(np.power(w_a, 2)), np.sum(np.power(w_b, 2))))

    distance = np.sqrt(num / denom)

    return distance
