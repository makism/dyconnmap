# -*- coding: utf-8 -*-
""" Surrogate Analysis



"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional, Tuple, Callable
import numpy as np
import numbers


def surrogate_analysis(
    ts1: "np.ndarray[np.float32]",
    ts2: "np.ndarray[np.float32]",
    num_surr: Optional[int] = 1000,
    estimator_func: Optional[
        Callable[["np.ndarray[np.float32]", "np.ndarray[np.float32]"], float]
    ] = None,
    ts1_no_surr: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, "np.ndarray[np.int32]", "np.ndarray[np.float32]", float]:
    """ Surrogate Analysis


    Parameters
    ----------
    ts1 :

    ts2 :

    num_surr : int

    estimator_func : function

    ts1_no_surr : boolean

    rng : object or None
        An object of type numpy.random.RandomState


    Returns
    -------
    p_val : float

    corr_surr :

    surrogates :

    r_value : float

    """
    if rng is None:
        rng = np.random.RandomState(0)

    if estimator_func is None:

        def estimator(x, y):
            return np.abs(np.corrcoef(x, y))[0, 1]

        estimator_func = estimator

    r_value = estimator_func(ts1, ts2)

    if isinstance(r_value, numbers.Real):
        r_value = [r_value]

    num_samples = len(ts1)
    num_r_vals = len(r_value)
    surrogates = np.zeros([2, num_surr, num_samples])

    if ts1_no_surr is True:
        surrogates[0, ...] = np.tile(ts1, [num_surr, 1])

    else:
        surrogates[0, ...] = aaft(ts1, num_surr, rng)

    surrogates[1, ...] = aaft(ts2, num_surr, rng)

    surr_vals = np.zeros((num_surr, len(r_value)))
    for i in range(num_surr):
        surr_vals[i, :] = estimator_func(surrogates[0, i, ...], surrogates[1, i, ...])

    surr_vals = np.array(surr_vals)
    p_vals = np.zeros((num_r_vals))

    for i in range(num_r_vals):
        r = np.where(surr_vals[:, i] > r_value[i])[0]

        p_val = 0.0
        if len(r) == 0:
            p_val = 1.0 / float(num_surr)
        else:
            p_val = len(r) / float(num_surr)

        p_vals[i] = p_val

    p_vals = p_vals.squeeze()
    surr_vals = surr_vals.squeeze()

    return p_vals, surr_vals, surrogates, r_value


def aaft(
    ts: "np.ndarray[np.float32]",
    num_surr: Optional[int] = 1,
    rng: Optional[np.random.RandomState] = None,
) -> "np.ndarray[np.float32]":
    """ Amplitude Adjusted Fourier Transform


    Parameters
    ----------
    ts :

    num_surr :

    rng : object or None
        An object of type numpy.random.RandomState


    Returns
    -------

    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples = len(ts)
    s = np.zeros((num_surr, n_samples))

    for i in range(num_surr):
        y = ts
        normal = np.sort(rng.randn(1, n_samples)).ravel()

        y, T = np.sort(ts), np.argsort(ts)
        T, r = np.sort(T), np.argsort(T)

        normal = phase_rand(normal[r], 1, rng).ravel()

        normal, T = np.sort(normal), np.argsort(normal)
        T, r = np.sort(T), np.argsort(T)

        s[i, :] = y[r]

    return s


def fdr(
    p_values: "np.ndarray[np.float32]",
    q: Optional[float] = 0.01,
    method: Optional[str] = "pdep",
) -> Tuple[bool, float]:
    """ False Discovery Rate



    Parameters
    ----------
    p_values :

    q :

    method :



    Returns
    -------


    """
    crit_p = 0.0
    h = False

    sorted_p_values = np.sort(p_values)

    m = len(sorted_p_values)

    thresh = np.arange(1, m + 1) * (q / m)

    rej = sorted_p_values <= thresh
    max_id = np.where(rej == True)[0]

    if max_id.size == 0:
        crit_p = 0.0
        h = p_values * 0

    else:
        max_id = np.max(max_id)
        crit_p = sorted_p_values[max_id]
        crit_p = crit_p.squeeze()
        h = p_values <= crit_p

    h = h.squeeze()
    h = h.astype(np.bool)

    return h, crit_p


def phase_rand(
    data, num_surr: Optional[int] = 1, rng: Optional[np.random.RandomState] = None
) -> "np.ndarray[np.float32]":
    """ Phase-randomized suggorates



    Parameters
    ----------
    data :

    num_surr :

    rng : object or None
        An object of type numpy.random.RandomState


    Returns
    -------

    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples = np.shape(data)[0]
    surrogates = np.zeros((num_surr, n_samples))

    half = np.int32(np.floor(n_samples / 2.0))

    surrogates = np.zeros((num_surr, n_samples))

    y = np.fft.fft(data)
    m = np.abs(y)
    p = np.angle(y)

    for i in range(num_surr):
        if n_samples % 2 == 0:
            p1 = rng.randn(half - 1, 1) * 2.0 * np.pi

            a = p1.T.ravel()
            b = p[half]
            c = np.flipud(p1).T.ravel()

            p[list(range(1, n_samples))] = np.hstack((a, b, -c))

            a = m[list(range(0, half + 1))]
            b = np.flipud(m[list(range(1, half))])
            m = np.hstack((a, b))
        else:
            p1 = rng.randn(half, 1) * 2.0 * np.pi

            a = p1
            b = np.flipud(p1).ravel()

            p[list(range(1, n_samples))] = a - b

        surrogates[i, :] = np.real(np.fft.ifft(np.exp(1j * p) * m))

    return surrogates
