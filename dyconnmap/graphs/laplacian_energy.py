# -*- coding: utf-8 -*-
""" Laplcian Energy

The Laplcian energy (LE) for a graph :math:`G` is computed as

.. math::
    LE(G) = \\sum_{i=1}^n | { \\mu_{i} - \\frac{2m}{n} } |
    ξ(A_1, A_2 ; t) = ‖exp⁡(-tL_1 ) - exp⁡(-tL_2 )‖_F^2

Where :math:`\mu_i` denote the eigenvalue associated with the node of the Laplcian
matrix of :math:`G` (Laplcian spectrum) and :math:`\\frac{2m}{n}` the average vertex degree.

For a details please go through the original work (Gutman2006_).

|

-----
.. [Gutman2006] Gutman, I., & Zhou, B. (2006). Laplacian energy of a graph. Linear Algebra and its applications, 414(1), 29-37.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>"

import numpy as np
import scipy
from scipy import sparse
import bct


def laplacian_energy(mtx: np.ndarray) -> float:
    """ Laplacian Energy


    Parameters
    ----------
    mtx : array-like, shape(N, N)
        Symmetric, weighted and undirected connectivity matrix.


    Returns
    -------
    le : float
        The Laplacian Energy.
    """
    lmtx = scipy.sparse.csgraph.laplacian(mtx, normed=False)
    w, _ = np.linalg.eig(lmtx)
    avg_degree = np.mean(bct.degrees_und(mtx))
    le = np.sum(np.abs(w - avg_degree))

    return le
