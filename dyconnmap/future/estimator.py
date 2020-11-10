"""
"""
import numpy as np

import multiprocessing
from joblib import Parallel, delayed

from typing import List, Type, Union, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import itertools
import collections


@dataclass(frozen=False)
class Estimator(ABC):
    """Base estimator class."""

    dtype: np.dtype = field(default=np.float32)

    requires_preprocessing: bool = field(default=False)

    filter: Optional[Callable] = field(
        default=None,
        init=True,
        metadata={"description": "A callback function to filter the given `Dataset`."},
    )

    filter_opts: Optional[Dict] = field(
        default_factory=dict,
        init=True,
        metadata={
            "description": "A dictionaray of parameters passed to the callback filter function."
        },
    )

    # est_ts: Optional[np.ndarray] = field(
    #     init=False,
    #     default=None,
    #     metadata={
    #         "description": "The estimated time series after invoking the estimator; some methods are not able to produce these intermediate time series (i.e. correlation)."
    #     },
    # )
    #
    # est_avg: Optional[np.ndarray] = field(
    #     init=False,
    #     default=None,
    #     metadata={"description": "The resulting connectivity matrices."},
    # )

    jobs: int = field(
        default=multiprocessing.cpu_count(),
        repr=True,
        metadata={"description": "Number of parallel jobs."},
    )

    def __post_init__(self):
        if self.jobs <= 0:
            self.jobs = 1

    def preprocess(self, data: np.ndarray, **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Estimator method; all subclasses must implement this."""
        ...

    def estimate_for_slides(
        self, data: np.ndarray, pairs: List[Tuple[int, int]], **kwargs
    ) -> np.ndarray:
        for window_id, slice_start, slice_end in pairs:
            tmp = self.estimate(data[:, slice_start:slice_end], **kwargs)

    def __call__(self, dataset: "Dataset", window: Optional["DynamicWindow"] = None):
        pairs = None
        ts = dataset.data

        if self.filter is not None and isinstance(self.filter, collections.Callable):
            ts = self.filter(ts, **self.filter_opts)

        if self.requires_preprocessing:
            ts = self.preprocess(ts)

        if window is None:
            cb_func = self.estimate
        else:
            cb_func = self.estimate_for_slides
            pairs = list(window)

        results = None
        if dataset.subjects == 1:
            results = cb_func(
                ts[0, :, :], pairs=pairs, subject_index=0, **dataset.settings()
            )
        else:
            results = Parallel(n_jobs=self.jobs)(
                delayed(cb_func)(
                    ts[index, :, :],
                    pairs=pairs,
                    subject_index=index,
                    **dataset.settings()
                )
                for index in np.arange(dataset.subjects)
            )

        return results
