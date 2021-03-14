"""Correlation Estimator."""

from .estimator import Estimator

import numpy as np
from joblib import Parallel, delayed


class Correlation(Estimator):
    """Correlation."""

    def __post_init__(self):
        super().__post_init__()
        self.dtype = np.float32

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        result = np.corrcoef(data)
        # return {"index": kwargs["subject_index"], "estimation": result}
        return result


def correlation(ts, **kwargs):
    """Correlation (func)."""
    n_subjects, n_rois, n_samples = np.shape(ts)

    rois = kwargs.get("rois", range(n_rois))

    obj = Correlation(rois=rois)
    ts = ts[:, rois, :]

    if obj.requires_preprocessing:
        ts = obj.preprocess(ts)

    results = Parallel(n_jobs=-1)(
        delayed(obj.estimate)(ts[index, :, :], subject_index=index)
        for index in np.arange(n_subjects)
    )

    return results
