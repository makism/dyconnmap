# -*- coding: utf-8 -*-

import dyconnmap

import numpy as np
import sklearn


if __name__ == "__main__":
    rng = np.random.RandomState(seed=0)
    data, _ = sklearn.datasets.make_moons(
        n_samples=1024, noise=0.125, random_state=rng)

    ng = dyconnmap.cluster.NeuralGas(rng=rng).fit(data)
    encoding, symbols = ng.encode(data)

    print(symbols)
