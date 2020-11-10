# -*- coding: utf-8 -*-
""" Graph Diffusion Distance

The Graph Diffusion Distance (GDD) metric (Hammond2013_) is a measure of distance
between two (positive) weighted graphs based on the Laplacian exponential diffusion kernel.
The notion backing this metric is that two graphs are similar if they emit comparable
patterns of information transmission.

This distance is computed by searching for a diffusion time :math:`t` that maximizes the
value of the Frobenius norm between the two diffusion kernels. The Laplacian operator
is defined as :math:`L = D - A`, where :math:`A` is the positive symmetric data matrix and :math:`D` is a diagonal
degree matrix for the adjacency matrix :math:`A`. The diffusion process (per vertex) on the adjacency
matrix :math:`A` is governed by a time-varying vector :math:`u(t)∈ R^N`. Thus, between each given pair of
(vertices’) weights :math:`i` and :math:`j`, their flux is quantified by :math:`a_{ij} (u_i (t)u_j (t))`. The grand
sum of these interactions is given by :math:`\hat{u}_j(t)=\sum_i{a_{ij}(u_i(t)u_j(t))=-Lu(t)}`.
Given the initial condition :math:`u^0,t=0` this sum has the following analytic solution :math:`u(t)=exp⁡(-tL)u^0`.
The resulting matrix is known as the Laplacian exponential diffusion kernel. Letting the diffusion process
run for :math:`t` time we compute and store the diffusion patterns in each column. Finally, the actual distance
measure between two adjacency matrices :math:`A_1` and  :math:`A_2`, at diffusion time :math:`t` is given by:

.. math::
    ξ(A_1, A_2 ; t) = ‖exp⁡(-tL_1 ) - exp⁡(-tL_2 )‖_F^2

where :math:`‖∙‖_F` is the Frobenious norm.

Notes
-----
Based on the code accompanied the original paper. Available at https://www.researchgate.net/publication/259621918_A_Matlab_code_for_computing_the_GDD_presented_in_the_paper

|

-----

.. [Hammond2013] Hammond, D. K., Gur, Y., & Johnson, C. R. (2013, December). Graph diffusion distance: A difference measure for weighted graphs based on the graph Laplacian exponential kernel. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 419-422). IEEE.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Tuple, Optional

import numpy as np
import scipy.optimize


def graph_diffusion_distance(
    a: np.ndarray, b: np.ndarray, threshold: Optional[float] = 1e-14
) -> Tuple[np.float32, np.float32]:
    """ Graph Diffusion Distance


    Parameters
    ----------
    a : array-like, shape(N, N)
        Weighted matrix.

    b : array-like, shape(N, N)
        Weighted matrix.

    threshold : float
        A threshold to filter out the small eigenvalues. If the you get NaN or INFs, try lowering this threshold.

    Returns
    -------
    gdd : float
        The estimated graph diffusion distance.

    xopt : float
        Parameters (over given interval) which minimize the objective function. (see :mod:`scipy.optimize.fmindbound`)

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
    xopt, fval, _, _ = scipy.optimize.fminbound(
        func=__min_fun, x1=0, x2=t_upperbound, xtol=1e-4, full_output=True
    )
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
