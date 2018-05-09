""" Markov matrix

Markov matrix (also refere as "transition matrix") is a square matrix that tabulates
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

"""
import numpy as np


def markov_matrix(symts):
    """ Markov Matrix


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
        next_sym = symts[i+1]

        mtx[curr_sym, next_sym] += 1

    mtx /= float(len(symts))

    return mtx
