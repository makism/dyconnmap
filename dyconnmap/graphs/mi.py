# -*- coding: utf-8 -*-
""" Mutual Information

Normalized Mutual Information (*NMI*) proposed by [Strehl2002]_ as an extension to
Mutual Information [cover] to enable interpretations and comparisons between two partitions.
Given the entropies :math:`H(P^a)=-\\sum_{i=1}^{k_a}{\\frac{n_i^a}{n}\\log(\\frac{n_i^a}{n})}` where :math:`n_i^a` represents the
number of patterns in group :math:`C_i^a \\in P^a` (and computed for :math:`H(P^b)` accordingly); the initial
matching of these two groups :math:`P^a` and :math:`P^b` in terms of mutual information is
[Fred2005_, Strehl2002_]:

.. math::
    I(P^a, P^b) = \\sum_{i=1}^{k_a} \\sum_{j=1}^{k_b} {\\frac{n_{ij}^{ab}}{n}} \\log \\left(\\frac{ \\frac{n_{ij}{ab}}{n} }{ \\frac{n_i^a}{n} \\frac{n_j^b}{n} } \\right)

Where :math:`n_{ij}^{ab}` denotes the number of shared patterns between the clusters :math:`C_i^a` and :math:`C_j^b`.
By exploiting the definition of mutual information, the following property holds true:
:math:`I(P^a,P^b) \\leq  \\frac{H(P^a)+H(P^b)}{2}`. This leads to the definition of NMI as:


.. math::
    NMI(A, B) = \\frac{2I(P^a, P^b)}{ H(P^a) + H(P^b)} = \\frac{ -2\\sum_{i=1}^{k_a} \\sum_{j=1}^{k_b} {n_{ij}^{ab}} \\log \\left( \\frac{ n_{ij}^{ab} n }{ n_i^a n_j^n } \\right)  }{  \\sum_{i=1}^{k_a} n_i^a \\log \\left( \\frac{n_i^a}{n} \\right) + \\sum_{j=1}^{k_b} n_j^b \\log \\left( \\frac{n_j^b}{n} \\right)   }


|

-----

.. [Fred2005] Fred, A. L., & Jain, A. K. (2005). Combining multiple clusterings using evidence accumulation. IEEE transactions on pattern analysis and machine intelligence, 27(6), 835-850.
.. [Strehl2002] Strehl, A., & Ghosh, J. (2002). Cluster ensembles---a knowledge reuse framework for combining multiple partitions. Journal of machine learning research, 3(Dec), 583-617.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>"

from typing import Tuple

import numpy as np

from dyconnmap.ts.entropy import entropy
from .vi import __unique_symbols


def mutual_information(
    indices_a: np.ndarray, indices_b: np.ndarray
) -> Tuple[float, float]:
    """ Mutual Information


    Parameters
    ----------
    indices_a : array-like, shape(n_samples)
        Symbolic time series.

    indices_b : array-like, shape(n_samples)
        Symbolic time series.


    Returns
    -------
    MI : float
        Mutual information.

    NMI : float
        Normalized mutual information.
    """
    indices_a = indices_a.flatten()
    indices_b = indices_b.flatten()

    entropy_a = -entropy(indices_a)
    entropy_b = -entropy(indices_b)

    Ua = __unique_symbols(indices_a)
    Ub = __unique_symbols(indices_b)

    N = len(indices_a)
    Sab = Ua.dot(Ub.T) / np.float32(N)
    Sa = np.diag(Ua.dot(Ua.T) / np.float32(N))
    Sb = np.diag(Ub.dot(Ub.T) / np.float32(N))

    # Add dummy dimension (needed for following computations).
    Sa = np.expand_dims(Sa, axis=1)
    Sb = np.expand_dims(Sb, axis=1)

    SS = Sab * np.log10(Sab / (Sa * Sb.T))

    MI = np.nansum(np.nansum(SS))
    NMI = 2 * MI / (entropy_a + entropy_b)

    return MI, NMI
