# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=3, linewidth=256)

from dyconnmap.ts import markov_matrix


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    symts = rng.randint(0, 4, 100)

    mtx =  markov_matrix(symts)


    print(mtx)
