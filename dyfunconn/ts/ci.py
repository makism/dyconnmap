""" Complexity Index

Notes
-----
Snippet adapted from:


"""
import numpy as np
np.set_printoptions(precision=2, linewidth=256)


def complexity_index(x, sub_len=-1, normalize=True, iter=200):
    """ Complexity Index


    Parameters
    ----------
    x :
        Input symbolic time series.

    sub_len : int
        Maximum subword length. Default is `len(x) - 1`.

    normalize : bool
        Normalize result. Default is `True`.

    iters : int
        Number of iterations to perform randomization. Default is `200`.


    Returns
    -------
    ci : float
        The computed complexity index.

    spectrum : array-like

    """
    ci = 0.0

    x = np.int32(x)
    x = x.flatten()

    len_x = len(x)

    max_subword_len = len_x - 1
    if sub_len >= 2:
        max_subword_len = sub_len

    min_x = np.min(x)
    x = x - min_x

    letters = np.unique(x)
    max_len_word = min([max_subword_len, len_x - 1])
    spectrum = np.ones((max_len_word))
    spectrum[0] = len(letters)

    print spectrum

    all_num_words = list()
    for word_len in range(1, max_len_word):
        cumulative_words = None

        for shift in range(word_len):
            num_words = np.int32(np.floor((len_x - shift) / word_len))
            all_num_words.append(num_words)

            if num_words > 0:
                idx1 = shift
                idx2 = word_len* num_words + shift
                sliced = x[idx1:idx2]
                words = np.reshape(sliced, (num_words, word_len)).T

                if cumulative_words is None:
                    cumulative_words = words
                else:
                    cumulative_words = np.hstack([cumulative_words, words])

        conv_cumulative_words = __rowsBaseConv(cumulative_words)
        u_cumulative_words = np.unique(conv_cumulative_words)
        spectrum[word_len - 1] = len(u_cumulative_words)
        cumulative_words = None

    print spectrum

    ci = np.sum(spectrum)

    all_num_words = np.array(all_num_words).flatten()

    return ci


def __rowsBaseConv(x, base=None):
    """

    """
    if base is None:
        base = np.max(x) + 1

    n, p = np.shape(x)

    bases = np.ones(p) * base
    indices = range(p-1, -1, -1)
    base = np.power(bases, indices)

    result = x.dot(base)

    return result


if __name__ == '__main__':
    import scipy
    from scipy import io

    x = scipy.io.loadmat('/home/makism/Development/Matlab/supcode/counting-forbidden-patterns/x.mat')['x']

    ci = complexity_index(x)

    print ci
