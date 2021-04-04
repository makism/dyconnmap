"""Phase Locking Value Estimator."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

import collections
import itertools

import numpy as np

from .basicfilter import analytic_signal
from .estimator import Estimator


class PhaseLockingValue(Estimator):
    """Phase Locking Value (PLV)."""

    def __post_init__(self):
        """Post init; setup estimator."""
        super().__post_init__()
        self.dtype = np.complex
        self.requires_preprocessing = True

    def preprocess(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocess input data; compute the analytic signal."""
        u_phases = analytic_signal(data)

        return u_phases

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Run the Phase Locking Value algorithm."""
        rois = kwargs["rois"]
        samples = kwargs["samples"]

        # Cartesian product of all ROIs.
        tmp = list(range(rois))
        rois_pairs = list(itertools.product(*[tmp, tmp]))

        if "window" not in kwargs:
            ts_size = samples
        else:
            ts_size = kwargs["window"]["window_length"]

        avg = np.zeros((rois, rois))
        for pair in rois_pairs:
            u_phases1, u_phases2 = data[pair, :]
            ts_plv = np.exp(1j * (u_phases1 - u_phases2))
            avg_plv = np.abs(np.sum((ts_plv))) / float(ts_size)

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
