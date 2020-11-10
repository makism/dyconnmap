# -*- coding: utf-8 -*-
""" Symbolic Transfer Entropy


"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Tuple, Optional
import numpy as np


def entropy_reduction_rate(sym_ts: "np.ndarray[np.float32]") -> float:
    """ Entropy Reduction Rate


    Parameters
    ----------
    sym_ts : array-like, shape(n_samples)
        Input symblic time series.


    Returns
    -------
    entredrate : float
        The estimated entropy reduction rate.
    """
    r = np.unique(sym_ts)

    num_symbols = len(r)

    conprob = np.zeros((num_symbols, num_symbols))

    for i in range(len(sym_ts) - 1):
        sym1 = np.where(sym_ts[i] == r)[0].ravel()
        sym1 = sym1[0]

        sym2 = np.where(sym_ts[i + 1] == r)[0].ravel()
        sym2 = sym2[0]

        conprob[sym1, sym2] += 1

    for i in range(num_symbols):
        sum1 = np.sum(conprob[i, :])
        conprob[i, :] = conprob[i, :] / sum1

    entropy = 0.0
    prob = np.zeros((num_symbols))

    for i in range(num_symbols):
        p = np.where(sym_ts == r[i])[0]
        prob[i] = len(p) / np.float32(len(sym_ts))
        entropy += prob[i] * np.log(prob[i])

    entropy = -entropy

    # Conditional entropy
    condentropy = np.zeros((num_symbols))

    sum1 = 0.0

    for i in range(num_symbols):
        indices = np.where(conprob[i, :] > 0)[0]

        l = len(indices)
        for j in range(l):
            sum1 += conprob[i, indices[j]] * np.log(conprob[i, indices[j]])
        condentropy[i] = -sum1
        sum1 = 0

    entredrate = 0.0
    sum1 = 0.0

    for i in range(num_symbols):
        sum1 = sum1 + prob[i] * condentropy[i]

    entredrate = (entropy - sum1) / entropy

    return entredrate


def symoblic_transfer_entropy(
    x: "np.ndarray[np.int32]",
    y: "np.ndarray[np.int32]",
    s: Optional[int] = 1,
    delay: Optional[int] = 0,
    verbose: Optional[bool] = False,
) -> Tuple[float, float, float]:
    """ Symbolic Tranfer Entropy


    Parameters
    ----------
    x : array-like, shape(N)
        Symblic time series (1D).

    y : array-like, shape(N)
        Symbolic time series (1D).

    s : int
        Embedding dimension.

    delay : int
        Time delay parameter

    verbose : boolean
        Print computation messages.


    Returns
    -------
    tent_diff : float
        The difference of the tranfer entropies of the two time series.

    tentxy : float
        The estimated tranfer entropy of x -> y.

    tentyx : float
        The estimated tranfer entropy of y -> x.
    """
    if len(x) != len(y):
        raise Exception("The lengths do not match.")

    symbols = np.unique([x, y])

    num_symbols = len(symbols)

    l = len(x)
    for k in range(l):
        x[k] = np.where(x[k] == symbols)[0]
        y[k] = np.where(y[k] == symbols)[0]

    x = x.astype(np.int32)
    y = y.astype(np.int32)

    pxy = np.zeros((num_symbols, num_symbols))
    l = len(x)
    for k in range(l):
        pxy[x[k], y[k]] = pxy[x[k], y[k]] + 1
    sum1 = np.sum(np.sum(pxy))
    pxy = pxy / sum1

    tentxy = __transfer_entropy(y, x, pxy, num_symbols, s=s, delay=delay)
    tentyx = __transfer_entropy(
        x, y, pxy, num_symbols, s=s, delay=delay, switch_indices=True
    )
    tent_diff = tentxy - tentyx

    if verbose:
        if tent_diff > 0.0 and tent_diff != np.inf:
            print("System x drives y.")

        elif tent_diff < 0.0:
            print("System y drives x.")

        elif tent_diff == 0.0:
            print("Symmetric bidirectionality.")

        elif tent_diff == np.inf:
            print("No information can be extracted by the two symbolic time series.")

    return tent_diff, tentxy, tentyx


def __transfer_entropy(x, y, pxy, num_symbols, s=1, delay=0, switch_indices=False):
    """ Transfer Entropy


    Parameters
    ----------
    x : array-like, shape(N)
        Symblic time series (1D).

    y : array-like, shape(N)
        Symbolic time series (1D)

    pxy : float
        The joint entropy.

    num_symbols : int
        The number of unique symbols in the time series.

    s : int
        Embedding dimension.

    delay : int
        Time delay parameter.

    switch_indices : boolean
        Compute the transfer entropy from y -> x.


    Returns
    -------
    te : float
        Transfer entropy.
    """
    pxxy = np.zeros((num_symbols, num_symbols, num_symbols))
    for k in range(len(x) - s):
        pxxy[x[k + s], x[k], y[k]] += 1
    sum1 = np.sum(np.sum(np.sum(pxxy)))
    pxxy = pxxy / sum1

    pxx = np.zeros((num_symbols, num_symbols))
    for k in range(len(x) - s):
        pxx[x[k + s], x[k]] += 1
    sum1 = np.sum(np.sum(pxx))
    pxx = pxx / sum1

    px = np.zeros((num_symbols))
    for k in range(len(x)):
        px[x[k]] += 1
    sum1 = np.sum(px)
    px = px / sum1

    tentyx = np.zeros((num_symbols * num_symbols * num_symbols))
    count = 0
    for k in range(num_symbols):
        for l in range(num_symbols):
            for m in range(num_symbols):
                ind1, ind2 = l, m

                # This is needed when estimating the tranfer entropy of
                # y -> x
                if switch_indices:
                    ind1, ind2 = m, l

                num = pxxy[k, l, m] * px[l]
                dem = pxy[ind1, ind2] * pxx[k, l]

                tentyx[count] = pxxy[k, l, m] * np.log2(num / dem)

                count += 1

    return np.sum(tentyx)
