""" Sample Entropy



Notes
-----
Based on https://nl.mathworks.com/matlabcentral/fileexchange/35784-sample-entropy


|

-----


.. [Stam2007] Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Human brain mapping, 28(11), 1178-1193.
.. [Hardmeier2014] Hardmeier, M., Hatz, F., Bousleiman, H., Schindler, C., Stam, C. J., & Fuhr, P. (2014). Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase lag index (wPLI) derived from high resolution EEG. PloS one, 9(10), e108648.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional
import numpy as np
from numpy import matlib


def sample_entropy(
    data: "np.ndarray[np.int32]",
    dim: Optional[int] = 2,
    tau: Optional[int] = None,
    r: Optional[float] = None,
) -> float:
    """ Sample Entropy


    Parameters
    ----------
    data : array-like, shape(n_samples)
        Symbolic time series.

    dim : int
        Embedding dimension. Default `2`.

    tau : int
        Delay time for downsampling. Is `None`, `1` is passed.

    r : float
        Tolerance factor. If `None`, `0.2 * std(data)` is passed.


    Returns
    -------
    SampEn : float
        Sample entropy.


    See also
    --------
    dyconnmap.ts.embed_ts: Embedded timeseries
    """
    if tau is None:
        tau = 1

    data = data.ravel()

    # do downsample
    if tau > 1:
        N = len(data)
        ds_data = [data[i] for i in range(0, N, tau)]
        data = ds_data

    if r is None:
        r = 0.2 * np.std(data)

    N = len(data)
    correl = np.zeros((1, 2)).squeeze()
    dataMat = np.zeros([dim + 1, N - dim])

    for i in range(0, dim + 1):
        offset = np.arange(i, N - dim + i)
        dataMat[i, :] = data[offset]

    for m in range(dim, dim + 2):
        count = np.zeros((1, N - dim)).squeeze()
        tmpMat = dataMat[0:m, :]

        for i in range(0, N - m):
            a = tmpMat[:, i + 1 : N - dim]
            b = tmpMat[:, i]
            b2 = matlib.repmat(b, N - dim - i - 1, 1)

            dist = np.abs(a.T - b2).T.max(0)

            D = dist < r
            count[i] = np.sum(D) / float(N - dim)

        correl[m - dim] = np.sum(count) / float(N - dim)

    saen = np.log(correl[0] / correl[1])

    return saen
