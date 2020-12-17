import numpy as np
import scipy
from joblib import Parallel, delayed

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
from dyconnmap.fc import plv_fast

from dataset import Dataset, Modality
from correlation import Correlation, correlation

# from timevar import TimeVarying
from slidingwindow import SlidingWindow
from phaselockingvalue import PhaseLockingValue, phaselockingvalue
from pipeline import Pipeline
from basicfilter import passband_filter


import json


def spectral_k_distance(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """Spectral K Distance."""
    X = np.squeeze(X)
    Y = np.squeeze(Y)

    l_mtx_a = scipy.sparse.csgraph.laplacian(X, normed=False)
    l_mtx_b = scipy.sparse.csgraph.laplacian(Y, normed=False)

    w_a, _ = scipy.sparse.linalg.eigs(l_mtx_a, k=k)
    w_a = np.real(w_a)
    w_a = np.sort(w_a)[::-1]

    w_b, _ = scipy.sparse.linalg.eigs(l_mtx_b, k=k)
    w_b = np.real(w_b)
    w_b = np.sort(w_b)[::-1]

    num = np.sum(np.power(w_a - w_b, 2))
    denom = np.min((np.sum(np.power(w_a, 2)), np.sum(np.power(w_b, 2))))

    distance = np.sqrt(num / denom)

    return distance


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 32
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)

    ds = Dataset(data, modality=Modality.Raw, fs=128.0)
    ds.labels = ["a", "b", "c", "d"]

    ds.write("/tmp/myds")

    # win = SlidingWindow(step=5, window_length=10)
    # win = TimeVarying(step=10, samples=128, rois=32, window_length=10)

    # conn = Correlation(rois=[0, 3])
    # conn = PhaseLockingValue(
    #     filter=passband_filter, filter_opts={"fs": 128.0, "fb": [1.0, 4.0]}
    # )
    #
    # # result = conn(ds, win)
    # result = conn(ds)
    # result = np.array(result)
    # print(result)

    # cb_func = spectral_k_distance
    # tmp2 = [(i, ii) for i in range(5) for ii in range(i, 5) if i != ii]
    # distances = Parallel(n_jobs=1)(
    #     delayed(cb_func)(result[x, :, :], result[y, :, :], k=3) for x, y in tmp2
    # )
    #
    # print(spectral_k_distance(result[0, :, :], result[1, :, :], k=3))
    #
    # print(distances)

    # print()
    # #
    # result2 = phaselockingvalue(
    #     data, filter=passband_filter, filter_opts={"fs": 128.0, "fb": [1.0, 4.0]}
    # )
    # print(result2)
