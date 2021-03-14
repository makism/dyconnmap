"""Phase Locking Value Estimator."""
import numpy as np
import itertools

from .estimator import Estimator
from .basicfilter import analytic_signal
from joblib import Parallel, delayed
import collections


class PhaseLockingValue(Estimator):
    """Phase Locking Value (PLV)."""

    def __post_init__(self):
        super().__post_init__()
        self.dtype = np.complex
        self.requires_preprocessing = True

    def preprocess(self, data: np.ndarray, **kwargs) -> np.ndarray:
        u_phases = analytic_signal(data)

        return u_phases

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        rois = kwargs["rois"]
        samples = kwargs["samples"]
        # print("samples: ", samples)

        # Cartesian product of all ROIs.
        tmp = list(range(rois))
        rois_pairs = list(itertools.product(*[tmp, tmp]))
        # l = rois
        # rois_pairs = [(roi1, roi2) for roi1 in range(0, l) for roi2 in range(0, l)]
        # print(rois_pairs)

        if "window" not in kwargs:
            ts_size = samples
        else:
            ts_size = kwargs["window"]["window_length"]
        # print("ts_size: ", ts_size)

        # ts = np.zeros((rois, rois, ts_size), dtype=np.complex)
        avg = np.zeros((rois, rois))

        # print("np.shape(data): ", np.shape(data))
        # print("np.shape(ts): ", np.shape(ts))
        # print("np.shape(avg): ", np.shape(avg))

        for pair in rois_pairs:
            # print(f"pair: {pair}")
            u_phases1, u_phases2 = data[pair, :]
            ts_plv = np.exp(1j * (u_phases1 - u_phases2))
            avg_plv = np.abs(np.sum((ts_plv))) / float(ts_size)

            # ts[pair] = ts_plv
            avg[pair] = avg_plv

        return avg


def phaselockingvalue(ts, **kwargs):
    """Phase Locking Value (func)."""
    n_subjects, n_rois, n_samples = np.shape(ts)

    rois = kwargs.get("rois", range(n_rois))
    filter = kwargs.get("filter", None)
    filter_opts = kwargs.get("filter_opts", None)

    obj = PhaseLockingValue()
    ts = ts[0, rois, :]
    ts = np.squeeze(ts)

    if filter is not None and isinstance(filter, collections.Callable):
        ts = filter(ts, **filter_opts)

    if obj.requires_preprocessing:
        ts = obj.preprocess(ts)

    results = obj.estimate(ts, rois=len(rois), samples=n_samples)

    return results
