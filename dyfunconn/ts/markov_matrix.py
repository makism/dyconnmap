""" Markov matrix

Generation of markov matrix and some related state transition features.

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np


def markov_matrix(symts):
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

    Returns
    -------
    mtx : matrix
        The transition matrix.
    """
    symbols = np.unique(symts)
    l = len(symbols)

    mtx = np.zeros((l, l))

    for i in range(len(symts) - 1):
        curr_sym = symts[i]
        next_sym = symts[i + 1]

        mtx[curr_sym, next_sym] += 1

    mtx /= float(len(symts))

    return mtx


def transition_rate(symts, weight=None):
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

    return TR / weight


def occupancy_time(symts, symbol_states=None, weight=None):
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
        oc = np.zeros((len(symbols)))
    else:
        oc = np.zeros((symbol_states))
    l = len(symts)

    if weight is None:
        weight = np.float32(l)

    for pos in range(l - 1):
        curr_sym = symts[pos]
        next_sym = symts[pos + 1]

        if curr_sym == next_sym:
            oc[curr_sym] += 1

    oc /= weight

    return oc, symbols
