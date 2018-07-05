""" Spectral Euclidean Distance

The spectral distance between graphs is simply the Euclidean distance between the spectra.

.. math::
    d(G, H) = \\sqrt{ \\sum_i{  (g_i  - h_j)^2   }  }


Notes
-----
* The input graphs can be a standard adjency matrix, or a variant of Laplacian.


|


----

.. [Wilson2008] Wilson, R. C., & Zhu, P. (2008). A study of graph spectra for comparing graphs and trees. Pattern Recognition, 41(9), 2833-2841.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def spectral_euclidean_distance(X, Y):
    """


    Parameters
    ----------
    X : array-like, shape(N, N)
        A weighted matrix.

    Y : array-like, shape(N, N)
        A weighted matrix:


    Returns
    -------
    distance : float
        The euclidean distance between the two spectrums.
    """
    w_a = np.linalg.eigvals(X)
    w_a = np.sort(w_a)

    w_b = np.linalg.eigvals(Y)
    w_b = np.sort(w_b)

    distance = np.linalg.norm(w_a - w_b, 2, 0)

    return distance
