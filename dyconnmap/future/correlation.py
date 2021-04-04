"""Correlation Estimator."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from joblib import Parallel, delayed

import numpy as np

from .estimator import Estimator


class Correlation(Estimator):
    """Correlation."""

    def __post_init__(self):
        """Post init; set datatype."""
        super().__post_init__()
        self.dtype = np.float32

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Estimate correlation connectivity."""
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
