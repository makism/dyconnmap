# -*- coding: utf-8 -*-
""" False Nearest Neighbors


|

-----


.. [Kennel1992] Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45(6), 3403.
.. [Abarbane2012] Abarbanel, H. (2012). Analysis of observed chaotic data. Springer Science & Business Media.
"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional
import numpy as np


def fnn(
    ts: "np.ndarray[np.float32]",
    tau: int,
    max_dim: Optional[int] = 20,
    neighbors_reduction: Optional[float] = 0.10,
    rtol: Optional[float] = 15.0,
    atol: Optional[float] = 2.0,
) -> Optional[int]:
    """ False Nearest Neighbors

    Notes
    -----
    The execution stops either when the maxium number of embedding dimensions is
    reached, or the the number of neighbors is reduced to specific percentage.


    Parameters
    ----------
    ts : array-like, 1d

    tau : int
        Time-delay parameter.

    max_dim : int
        Maximum embedding dimension.

    neighbors_reduction : float
        Maximum percentage of neighbors reduction. Default '0.10' (10%).

    rtol : float
        First threshold, criterion to identify a false neighbor. (Neighborhood size)

    atol : float
        Second threshold, criterion to identify a false neighbor.


    Returns
    -------
    min_dimension : int
        Minimum embedding dimension.
    """
    ts = ts.flatten()

    neighbors_perc = neighbors_reduction * 100.0
    fnn_ini = 0.0
    min_dimension = None

    Ra = np.std(ts)
    for dim in range(1, max_dim):
        min_dimension = dim
        ed_ts, num_points = __embed_delay(ts, dim, tau)

        if ed_ts is not None and num_points > 0:
            distances = np.zeros((num_points, num_points))
            for i in range(num_points):
                distances[i, :] = __euclidean_distance(
                    ed_ts, np.ones((num_points, dim)) * ed_ts[i, :]
                )

            indices = np.argsort(distances)
            sort_distances = np.sort(distances)

            all_D = np.abs(
                ts[np.arange(num_points) + dim * tau] - ts[indices[:, 1] + dim * tau]
            )
            all_R = np.sqrt(np.power(all_D, 2) + np.power(sort_distances[:, 1], 2))

            a = all_D / sort_distances[:, 1]
            b = all_R / Ra

            fnn_a_or_b = np.where((a > rtol) | (b > atol))
            fnn_value = len(fnn_a_or_b[0])

            if dim == 1:
                fnn_ini = fnn_value

            elif fnn_value < fnn_ini / neighbors_perc:
                break

        else:
            break

    return min_dimension


def __euclidean_distance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2), 1))


def __embed_delay(ts, dim, tau):
    """ Embed delay, treated for FNN method.

    For internal use only.
    """
    ts = ts.flatten()
    new_ts = np.zeros((dim, len(ts)))
    new_ts[0, :] = ts

    l = len(ts)
    m = l - (dim) * tau
    if dim == 1:
        m = l - tau

    if m < 0:
        return None, m

    for i in range(1, dim):
        offset = i - 1
        tmp = np.roll(new_ts[offset], l - tau)
        new_ts[i] = tmp
    new_ts = new_ts.T

    y = new_ts[0:m, 0:dim]

    return y, m
