import numpy as np
import scipy
from joblib import Parallel, delayed

import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/future/")
from dyconnmap.fc import plv_fast

from dataset import Dataset, Modality
from correlation import Correlation, correlation

from timevar import TimeVarying
from slidingwindow import SlidingWindow
from phaselockingvalue import PhaseLockingValue, phaselockingvalue
from pipeline import Pipeline
from basicfilter import passband_filter
from bv import bv_parse_voi


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


import pysnooper

if __name__ == "__main__":
    voi_fname = (
        "/home/makism/Documents/BrainVoyager/SampleData/NF Pilot Study/s01/s01_V5.voi"
    )
    voi_desc, vois = bv_parse_voi(fname=voi_fname)

    print(vois[0]["NrOfVoxels"])
    print(vois[0]["NameOfVOI"])
    print(len(vois[0]["Coords"]))
    print(vois[1]["NrOfVoxels"])
    print(len(vois[1]["Coords"]))

    sys.exit(0)

    rng = np.random.RandomState(0)

    n_subjects = 1
    n_rois = 4
    n_samples = 128 * 3
    fs = 128.0
    cc = 2.0
    step = 10
    fb = [1.0, 4.0]

    data = rng.rand(n_subjects, n_rois, n_samples)

    # with pysnooper.snoop():
    ds = Dataset(data, modality=Modality.Raw, fs=fs)

    win = TimeVarying(step=step, cc=cc)
    win = SlidingWindow(window_length=10)
    conn = PhaseLockingValue(
        rois=None, filter=passband_filter, filter_opts={"fs": fs, "fb": fb}
    )

    result = conn(ds, win)
    print(result[0, :, :])
