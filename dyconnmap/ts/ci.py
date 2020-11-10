""" Complexity Index


Complexity index (Janson2004_, Rapp2007_) computes the `l`-subword complexity (`l`-subword spectrum) of a one-dimensional,
symbolic (integer) time series, by finding the number of distinct subwords
of length `l`. The total complexity is given by the sum of all subwords of different lengths.
The unnormalized measure can be used to compare sequences of equal lengths.


Notes
-----
* This is a direct translation from the `Complexity toolbox available at` http://users.auth.gr/~stdimitr/files/software/complexitiy.rar
* Original author is Stravros Dimitriadis <stidimitriadis@gmail.com>

|

.. [Janson2004] Janson, S., Lonardi, S., & Szpankowski, W. (2004). On average sequence complexity. Theoretical Computer Science, 326(1-3), 213-227.
.. [Rapp2007] Rapp, P. E. (2007). Quantitative characterization of animal behavior following blast exposure. Cognitive neurodynamics, 1(4), 287-293.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Tuple, Optional, Union

import numpy as np


def complexity_index(
    x: "np.ndarray[np.int32]",
    sub_len: Optional[int] = -1,
    normalize: Optional[bool] = False,
    iterations: Optional[int] = 100,
) -> Union[
    Tuple[np.float32, "np.ndarray[np.int32]"],
    Tuple[np.float32, np.float32, "np.ndarray[np.int32]"],
]:
    """ Complexity Index


    Parameters
    ----------
    x : array-like, shape(n_samples)
        Input symbolic time series.

    sub_len : int
        Maximum subword length. Default is `len(x) - 1`.

    normalize : bool
        Normalize result. Default is `False`.

    iterationss : int
        Number of iterations to perform randomization. Default is `100`.


    Returns
    -------
    normal_ci : float
        The computed omplexity index after normalization against the randomization
        procedure.

    ci : float
        The computed complexity index.

    spectrum : array-like
        A list of the number of distinct subwords of length 1, up to the size
        of the input symbolic time series.
    """
    x = x.astype(np.int32)
    x = x.flatten()
    len_x = len(x)

    ci, spectrum = __compute_complexity_index(x, sub_len)

    if normalize:
        rng = np.random.RandomState(0)

        mean_ci = 0.0
        num_letters = spectrum[0]

        for _ in range(iterations):
            new_x = np.int32(np.floor(rng.rand(len_x) * num_letters))  # type: ignore
            new_ci, _ = __compute_complexity_index(new_x, sub_len)
            mean_ci += new_ci / iterations

        normal_ci = np.float32(ci / mean_ci)

        return normal_ci, ci, spectrum

    else:
        return ci, spectrum


def __compute_complexity_index(
    x, sub_len=-1
) -> Tuple[np.float32, "np.ndarray[np.int32]"]:
    """ Complexity Index


    Parameters
    ----------
    x :
        Input symbolic time series.

    sub_len : int
        Maximum subword length. Default is `len(x) - 1`.


    Returns
    -------
    ci : float
        The computed complexity index.

    spectrum : array-like
        A list of the number of distinct subwords of length 1, up to the size
        of the input symbolic time series.
    """
    ci = 0.0

    # x = x.astype(np.int32)
    # x = x.flatten()

    len_x = len(x)

    max_subword_len = len_x - 1
    if sub_len >= 2:
        max_subword_len = sub_len

    min_x = np.min(x)
    x = x - min_x

    letters = np.unique(x)
    max_len_word = np.min([max_subword_len, len_x - 1])  # type: ignore

    spectrum = np.ones((max_len_word), dtype=np.int32)
    spectrum[0] = len(letters)

    all_num_words = list()
    for word_len in range(1, max_len_word):
        real_word_len = word_len + 1
        cumulative_words = None

        for shift in range(real_word_len):
            num_words = np.int32(np.floor((len_x - shift) / real_word_len))
            all_num_words.append(num_words)

            if num_words > 0:
                idx1 = shift
                idx2 = real_word_len * num_words + shift
                sliced = x[idx1:idx2]
                words = np.reshape(sliced, (num_words, real_word_len)).T

                if cumulative_words is None:
                    cumulative_words = words
                else:
                    cumulative_words = np.hstack([cumulative_words, words])

        conv_cumulative_words = __rowsBaseConv(cumulative_words.T)  # type: ignore
        u_cumulative_words = np.unique(conv_cumulative_words)

        spectrum[word_len] = len(u_cumulative_words)
        cumulative_words = None

    ci = np.float32(np.sum(spectrum))

    all_num_words = np.array(all_num_words).flatten()

    return ci, spectrum


def __rowsBaseConv(
    x: "np.ndarray[np.int32]", base: Optional[int] = None
) -> "np.ndarray[np.float64]":
    """

    Parameters
    ----------
    x :

    base : integer


    Returns
    -------


    """
    if base is None:
        base = np.max(x) + 1

    _, p = np.shape(x)  # type: ignore

    bases = np.ones(p) * base
    indices = list(range(p - 1, -1, -1))
    base = np.power(bases, indices)  # type: ignore

    result = x.dot(base).astype(np.float64)  # type: ignore

    return result
