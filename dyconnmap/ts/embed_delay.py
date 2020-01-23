# -*- coding: utf-8 -*-
"""

When dealing with non-linear time series analysis, it is common to reconstruct
hem as time delay vectors in phase space. This new reconstruction, describes the
tmporal evolution of a system in a state space; a trajectory of interchanging states.
The need for this state space, stems from the fact that the original system may
contain latent and unobserved variables that we would like to expose. Thus, we
construct :math:`m`-dimensional phase vectors from :math:`\\tau`-time delayed samples (Takens1981_):

.. math::
    s_n = (s_(n-(m-1)τ),s_(n-(m-2)) τ, …, s_n)

This new space, is shown to preserve the dynamics properties of the original
phase space. For more on the subject, the interested readers are encouraged to
consult the work of Bradley and Kantz (Bradley2015_).


|

-----

.. [Taken1981] Takens, F. (1981). Detecting strange attractors in turbulence. Lecture notes in mathematics, 898(1), 366-381.
.. [Bradley2015] Bradley, E., & Kantz, H. (2015). Nonlinear time-series analysis revisited. Chaos: An Interdisciplinary Journal of Nonlinear Science, 25(9), 097610.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def embed_delay(ts, dim, tau):
    """ Embed delay


    Parameters
    ----------
    ts : array-like, shape(n_samples)
        One-dimensional symbolic time series.

    dim : int
        The embedding dimension.

    tau : int
        Time delay factor.

    Returns
    -------
    y : array-like
        The embedded timeseries.
    """
    ts = ts.flatten()
    new_ts = np.zeros((dim, len(ts)))
    new_ts[0, :] = ts

    l = len(ts)
    m = l - (dim - 1) * tau
    if dim == 1:
        m = l - tau

    if m < 0:
        return None

    for i in range(1, dim):
        offset = i - 1
        tmp = np.roll(new_ts[offset], l - tau)
        new_ts[i] = tmp
    new_ts = new_ts.T

    y = new_ts[0:m, 0:dim]

    return y
