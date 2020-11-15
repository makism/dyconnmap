"""Base Estimator."""
import numpy as np

import multiprocessing
from joblib import Parallel, delayed

from typing import List, Type, Union, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
import itertools
import collections
from abc import ABC, abstractmethod


@dataclass(frozen=False)
class Estimator(ABC):
    """Base estimator class."""

    dtype: np.dtype = field(default=np.float32)

    requires_preprocessing: bool = field(default=False)

    rois: Optional[List[int]] = field(
        default=None,
        init=True,
        metadata={
            "description": "An optional list of ROIs indices between which the Estimator wil be employed."
        },
    )

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

    # dataset_settings: dict = None

    # window_settings: dict = None

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
        rois = kwargs["rois"]
        slides = kwargs["slides"]

        dfcg = np.zeros((slides, rois, rois), dtype=self.dtype)
        for window_id, slice_start, slice_end in pairs:
            result = self.estimate(data[:, slice_start:slice_end], **kwargs)

            dfcg[window_id, :, :] = result["estimation"]

        return dfcg

    def __call__(self, dataset: "Dataset", window: Optional["DynamicWindow"] = None):
        pairs = None
        ts = dataset.data

        # Dataset's settings; we'll pass the where needed.
        settings = dataset.settings()

        # Check if a list of ROIs is given
        if self.rois is not None:
            ts = ts[:, self.rois, :]
            settings["rois"] = len(self.rois)

        # Apply the given filter ufunc on the input timeseries.
        if self.filter is not None and isinstance(self.filter, collections.Callable):
            ts = self.filter(ts, **self.filter_opts)

        # Run (if defined) the preprocessing function defined by the Estimator.
        if self.requires_preprocessing:
            ts = self.preprocess(ts)

        if window is None:
            cb_func = self.estimate
        else:
            window.prepare(**settings)
            settings["slides"] = window.slides

            cb_func = self.estimate_for_slides
            pairs = list(window)

        if dataset.subjects == 1:
            results = cb_func(ts[0, :, :], pairs=pairs, subject_index=0, **settings)
        else:
            results = Parallel(n_jobs=self.jobs)(
                delayed(cb_func)(
                    ts[index, :, :], pairs=pairs, subject_index=index, **settings
                )
                for index in np.arange(dataset.subjects)
            )

        return results
