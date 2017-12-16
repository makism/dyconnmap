# -*- coding: utf-8 -*-
"""




|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def embed_delay(ts, dim, tau):
    """ Embed delay


    Parameters
    ----------
    ts : array-like, shape(1, n_samples)
        Symbolic time series.

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
