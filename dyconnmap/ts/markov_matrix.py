""" Markov matrix

Generation of markov matrix and some related state transition features.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

from typing import Optional, Union, Tuple
import numpy as np


def markov_matrix(
    symts: "np.ndarray[np.int32]", states_from_length: Optional[bool] = True
) -> "np.ndarray[np.float32]":
    """ Markov Matrix

    Markov matrix (also refered as "transition matrix") is a square matrix that tabulates
    the observed transition probabilities between symbols for a finite Markov Chain. It is a first-order descriptor
    by which the next symbol depends only on the current symbol (and not on the previous ones);
    a Markov Chain model.

    A transition matrix is formally depicted as:

    Given the probability :math:`Pr(j|i)` of moving between :math:`i` and :math:`j` elements,
    the transition matrix is depicted as:

    .. math::
        P = \\begin{pmatrix}
                      P_{1,1} & P_{1,2} & \\ldots & P_{1,j} & \\ldots & P_{1,S} \\\\
                      P_{2,1} & P_{2,2} & \\ldots & P_{2,j} & \\ldots & P_{2,S} \\\\
                      \\vdots & \\vdots & \\ddots & \\vdots & \\ddots & \\vdots \\\\
                      P_{i,1} & P_{i,2} & \\ldots & P_{i,j} & \\ldots & P_{i,S} \\\\
                      \\vdots & \\vdots & \\ddots & \\vdots & \\ddots & \\vdots \\\\
                      P_{S,1} & P_{S,2} & \\ldots & P_{S,j} & \\ldots & P_{S,S} \\\\
            \\end{pmatrix}

    Since the transition matrix is row-normalized, so as the total transition probability
    from state :math:`i` to all the others must be equal to :math:`1`.

    For more properties consult, among other links WolframMathWorld_ and WikipediaMarkovMatrix_.

    |

    -----

    .. [WolframMathWorld] http://mathworld.wolfram.com/StochasticMatrix.html
    .. [WikipediaMarkovMatrix] https://en.wikipedia.org/wiki/Stochastic_matrix


    Parameters
    ----------
    symts : array-like, shape(N)
        One-dimensional discrete time series.

    states_from_length: bool or int, optional
        Used to account symbolic time series in which not all the symbols are present.
        That may happen when for example the symbols are drawn from different distributions.
        Default `True`, the size of the resulting Markov Matrix is equal to the number
        of unique symbols present in the time series. If `False`, the size will be the
        `highest symbolic state + 1`.
        You may also speficy the highest (inclusive) symbolic state.

    Returns
    -------
    mtx : matrix
        The transition matrix. The size depends the parameter `states_from_length`.
    """
    symbols = np.unique(symts)

    if isinstance(states_from_length, bool):
        if states_from_length:
            l = len(symbols)
        else:
            l = np.max(symbols) + 1
    elif isinstance(states_from_length, int):
        l = states_from_length
    else:
        l = len(symbols)

    mtx = np.zeros((l, l))
    for i in range(len(symts) - 1):
        curr_sym = symts[i]
        next_sym = symts[i + 1]

        mtx[curr_sym, next_sym] += 1

    mtx /= np.float32(len(symts))
    # mtx = mtx.astype(np.float32)

    return mtx


def transition_rate(
    symts: "np.ndarray[np.int32]",
    weight: Optional[Union[np.float32, "np.ndarray[np.float32]"]] = None,
) -> float:
    """ Transition Rate

    The total sum of transition between symbols.


    Parameters
    ----------
    symts :

    weight : float


    Returns
    -------

    """
    TR = 0.0

    l = len(symts)

    if weight is None:
        weight = np.float32(l)

    for pos in range(l - 1):
        curr_sym = symts[pos]
        next_sym = symts[pos + 1]

        if curr_sym != next_sym:
            TR += 1.0

    weighted_tr = np.float64(TR / weight)

    return weighted_tr


def occupancy_time(
    symts: "np.ndarray[np.int32]",
    symbol_states: np.int32 = None,
    weight: Optional[Union[np.float32, "np.ndarray[np.float32]"]] = None,
) -> Tuple[np.float64, "np.ndarray[np.int32]"]:
    """ Occupancy Time


    Parameters
    ----------

    symts :

    symbol_states : int
        The maximum number of symbols. This is useful to define in case your
        symbolic timeseries skips some states, in which case would produce
        a matrix of different size.

    weight : float
        The weights of the reuslting transition symbols. Default `len(symts)`.

    Returns
    -------

    oc :

    symbols :

    """
    symbols = np.unique(symts)

    if symbol_states is None:
        oc = np.zeros((len(symbols)), dtype=np.float32)
    else:
        oc = np.zeros((symbol_states), dtype=np.float32)
    l = len(symts)

    if weight is None:
        weight = np.float32(l)

    for pos in range(l - 1):
        curr_sym = symts[pos]
        next_sym = symts[pos + 1]

        if curr_sym == next_sym:
            oc[curr_sym] += 1

    weighted_oc = np.float64(oc / weight)
    # oc /= weight

    return weighted_oc, symbols
