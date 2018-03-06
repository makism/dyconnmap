# -*- coding: utf-8 -*-
""" Graph Diffusion Distance


Notes
-----
Based on the code accompanied the original paper. Available at https://www.researchgate.net/publication/259621918_A_Matlab_code_for_computing_the_GDD_presented_in_the_paper

|

-----

.. [Hammond2013] Hammond, D. K., Gur, Y., & Johnson, C. R. (2013, December). Graph diffusion distance: A difference measure for weighted graphs based on the graph Laplacian exponential kernel. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 419-422). IEEE.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com"

import numpy as np
import scipy.optimize


def graph_diffusion_distance(a, b, threshold=1e-14):
    """


    Parameters
    ----------
    a :
        Input matrix.

    b :
        Input matrix.

    threshold : float
        A threshold to filter out the small eigenvalues. If the you get NaN or INFs, try lowering this threshold.


    Returns
    -------
    gdd : float
        The computed graph diffusion distance value.

    xopt : float
        The timestep in which the gdd was computed.
    """
    L1 = __graph_laplacian(a)
    L2 = __graph_laplacian(b)

    w1, v1 = np.linalg.eig(L1)
    w2, v2 = np.linalg.eig(L2)

    eigs = np.hstack((np.diag(w1), np.diag(w2)))
    eigs = eigs[np.where(eigs > threshold)]
    eigs = np.sort(eigs)

    t_upperbound = np.real(1.0 / eigs[0])

    __min_fun = lambda t: -1.0 * __gdd_xi_t(v1, w1, v2, w2, t)
    xopt, fval, _, _ = scipy.optimize.fminbound(func=__min_fun, x1=0, x2=t_upperbound, xtol=1e-4, full_output=True)
    # xopt, fval, ierr, numfunc = scipy.optimize.fminbound(func=__min_fun, x1=0, x2=t_upperbound, xtol=1e-4, full_output=True)

    gdd = np.sqrt(-fval)

    return gdd, xopt


def __gdd_xi_t(V1, D1, V2, D2, t):
    """

    """
    E = 0.0

    ed1 = np.diag(np.exp(-t * np.diag(D1)))
    ed2 = np.diag(np.exp(-t * np.diag(D2)))

    tmp1 = V1.dot((ed1 * V1.T).conj())
    tmp2 = V2.dot((ed2 * V2.T).conj())
    tmp = tmp1 - tmp2

    E = np.sum(np.power(np.real(tmp), 2.0))

    return E


def __graph_laplacian(mtx):
    """ Compute the Laplacian of the matrix.

    .. math::

    """
    L = np.diag(np.sum(mtx, 0)) - mtx

    return L
