# -*- coding: utf-8 -*-
""" Relational Neural Gas

"""
# Author: Avraam Marimpis <avraam.marimpis@gmail.com>

import sys
import numpy as np
np.set_printoptions(precision=3, linewidth=256, edgeitems=10)


class RelationalNeuralGas:
    """


    """

    def __init__(self, n_protos=10, iterations=10, lrate=[0.3, 0.01]):
        self.n_protos = n_protos
        self.iterations = iterations
        self.lrate_i = lrate[0] * n_protos
        self.lrate_f = lrate[1]
        self.lrate = self.lrate_i
        self.protos = None
        self.coeff = None
        self.__multipl = None


    def fit(self, data, dma):
        """ Fit

        Parameters
        ----------

        data :
            A dissimilarity matrix.

        """
        np.fill_diagonal(data, 0.0)

        N, _ = np.shape(dma)

        self.__multipl = np.ones((1, N))
        self.protos = (1.0/N) * np.ones((self.n_protos, N))

        for iteration in range(1, self.iterations + 1):
            index = iteration # 1
            print(('Iteration {0}'.format(index)))

            t = iteration / float(self.iterations)

            r_dist = np.float32(self.__rdist(dma, self.protos))

            I = np.argsort(r_dist, axis=0, kind="heapsort")
            II = np.argsort(I, axis=0, kind="heapsort")
            hl = np.exp(-II / self.lrate)

            self.protos = np.float32(np.ones((1, N)) * self.__multipl * hl)
            tmp = np.sum((np.ones((self.n_protos, 1)) * self.__multipl * hl).T, axis=0)
            tmp = np.reshape(tmp, (self.n_protos, 1))
            tmp = tmp * np.ones((1, N))
            tmp = np.float32(tmp)

            tmp2 = np.divide(self.protos, tmp)
            tmp2 = np.float32(tmp2)

            self.protos = tmp2

            self.lrate = np.float32(self.lrate_i * (self.lrate_f / float(self.lrate_i)) ** t)

            print(("-"  * 80))

        self.coeff = self.protos
        self.protos = np.matmul(self.coeff, X)

        return self


    def encode(self, data):
        """ Encode


        """
        pass


    @classmethod
    def __rdist(mtx, coeff):
        """ Relational Distance


        Parameters
        ----------
        mtx : array-list, shape(num_features, num_features)
            Dissimilarity matrix.

        Returns
        -------

        """
        N, _ = np.shape(mtx)
        num_protos, num_features = np.shape(coeff)

        distances = np.zeros((num_protos, num_features))
        for i in range(num_protos):
            tmp = np.matmul(coeff[i, :], mtx)
            tmp2 = tmp - (0.5 * np.matmul(tmp, coeff[i, :].T))
            distances[i, :] = tmp2

        return distances

if __name__ == '__main__':
    import scipy
    from scipy import io
    from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances

    rng_workspace = scipy.io.loadmat('/home/makism/Development/Matlab/Neural Gas Networks/mfile/rng.mat')
    dma = rng_workspace['dma']
    n_prot = rng_workspace['n_prot'][0][0]
    n_iter = rng_workspace['iter'][0][0]
    X = rng_workspace['X']

    distx = euclidean_distances(X)
    np.fill_diagonal(distx, 0.0)

    rng = RelationalNeuralGas(n_prot, n_iter)
    rng.fit(X, dma)

    proj = rng.protos

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.hold(True)
    plt.scatter(proj[:, 0], proj[:, 1])
    plt.show()
