"""

"""

from estimator import Estimator

import numpy as np


class Correlation(Estimator):
    """Correlation."""

    def __init__(self):
        super()
        self.dtype = np.float64

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        result = np.corrcoef(data)
        return {"index": kwargs["subject_index"], "result": result}
