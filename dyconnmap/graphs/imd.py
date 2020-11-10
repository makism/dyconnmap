""" Ipsen-Mikhailov Distance


Given two graphs, this method quantifies their difference by comparing their
spectral densities. This spectral density is computed as the sum of Lorentz distributions :math:`\\rho(\\omega)`:

.. math::
    \\rho(\\omega) = K \\sum_{i=1}^{N-1} \\frac{\\gamma}{ (\\omega - \\omega_i)^2 + \\gamma^2 }

Where :math:`\\gamma` is the bandwidth, and :math:`K` a normalization constant such that :math:`\\int_{0}^{\\infty}\\rho(\\omega)d\\omega=1`.
The spectral distance between two graphs :math:`G` and :math:`H` with densities :math:`\\rho_G(\\omega)` and :math:`\\rho_H(\\omega)` respectively, is defined as:

.. math::
    \\epsilon = \\sqrt{ \\int_{0}^{\\infty}{[\\rho_G(\\omega) - \\rho_H(\\omega) ]^2 d(\\omega)} }



|


----

.. [Ipsen2004] Ipsen, M. (2004). Evolutionary reconstruction of networks. In Function and Regulation of Cellular Systems (pp. 241-249). Birkhäuser, Basel.
.. [Donnat2018] Donnat, C., & Holmes, S. (2018). Tracking Network Dynamics: a review of distances and similarity metrics. arXiv preprint arXiv:1801.07351.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional

import numpy as np
import scipy
from scipy.integrate import quad


def im_distance(
    X: np.ndarray, Y: np.ndarray, bandwidth: Optional[float] = 1.0
) -> float:
    """

    Parameters
    ----------
    X : array-like, shape(N, N)
        A weighted matrix.

    Y : array-like, shape(N, N)
        A weighted matrix.

    bandwidth : float
        Bandwidth of the kernel. Default `1.0`.


    Returns
    -------
    distance : float
        The estimated Ipsen-Mikhailov distance.
    """
    distance = 0.0

    l_mtx_a = scipy.sparse.csgraph.laplacian(X, normed=False)
    l_mtx_b = scipy.sparse.csgraph.laplacian(Y, normed=False)

    w_a, _ = scipy.linalg.eig(l_mtx_a)
    w_a = np.sqrt(w_a)

    w_b, _ = scipy.linalg.eig(l_mtx_b)
    w_b = np.sqrt(w_b)

    func1 = lambda x: _sum_lorentz_distribution(x, w_a, bandwidth)
    fnorm1 = lambda x: func1(x) / quad(func1, a=0.0, b=np.inf)[0]

    func2 = lambda x: _sum_lorentz_distribution(x, w_b, bandwidth)
    fnorm2 = lambda x: func2(x) / quad(func2, a=0.0, b=np.inf)[0]

    integrand = lambda x: np.power((fnorm1(x) - fnorm2(x)), 2)
    distance = np.sqrt(quad(integrand, a=0.0, b=np.inf)[0])

    return distance


def _sum_lorentz_distribution(X, eigs, bandwidth=1.0):
    """ The sum of Lorentz distributions.



    Parameters
    ----------
    X :

    eigs :

    bandwidthw : float
        Bandwidth of the kernel. Default `1.0`.


    Returns
    -------
    l : float

    """
    l_sum = np.sum(bandwidth / (np.power(X - eigs, 2) + np.power(bandwidth, 2)))

    return l_sum
