"""Dataset class.

An experimental Dataset container that ties together all the aspects of the
module.
"""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from dataclasses import dataclass, field
import multiprocessing
from joblib import Parallel, delayed

import numpy as np
from enum import IntEnum
from typing import List, Type, Union, Optional, Tuple

from abc import ABC, abstractmethod

import json


class Modality(IntEnum):
    """Modalities."""

    Any = 0
    Raw = 1
    fMRI = 2
    EEG = 3
    MEG = 4


class SlidingWindow:
    """Sliding Window."""

    def __init__(self, n_samples: int = 128, window_length: int = 10, step: int = 1):
        self.window_length: int = window_length
        self.step: int = step
        self.n_slides: int = np.int32(np.ceil((n_samples - window_length) / step + 1.0))
        self.n_rois: int = 32

        self.pairs: List[Tuple[int, int, int]] = [
            (
                win_id,
                np.int32(win_id * step),
                np.int32(win_id * step + self.window_length),
                # c1,
                # c2,
            )
            for win_id in range(self.n_slides)
            # for c1 in range(0, self.n_rois)
            # for c2 in range(c1, self.n_rois)
            # if c1 != c2
        ]

    def __iter__(self):
        return iter(self.pairs)


@dataclass(frozen=False)
class Estimator(ABC):
    """Base estimator class."""

    dtype: np.dtype = field(default=np.float32)

    n_jobs: int = field(
        default=multiprocessing.cpu_count(),
        repr=True,
        metadata={"description": "Number of parallel jobs."},
    )

    def __post_init__(self):
        if self.n_jobs <= 0:
            self.n_jobs = 1

    @abstractmethod
    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def estimate_pair(
        self, data: np.ndarray, against: np.ndarray, **kwargs
    ) -> np.ndarray:
        ...

    def __call__(self, dataset: "Dataset", window: Optional["SlidingWindow"] = None):
        if window is None:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.estimate)(dataset.data[index, :, :], subject_index=index)
                for index in np.arange(dataset.subjects)
            )

            return results

        else:
            for pair in window:
                slide_id, slice_start, slice_end = pair


class Correlation(Estimator):
    """Correlation."""

    def __init__(self):
        super()
        self.dtype = np.float64

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        result = np.corrcoef(data)
        return {"index": kwargs["subject_index"], "result": result}

    def estimate_pair(
        self, data: np.ndarray, against: np.ndarray, **kwargs
    ) -> np.ndarray:
        r = np.corrcoef(data, against)[0, 1]
        return r


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class Dataset:
    """Dataset."""

    version: float = field(default=1.0, init=False)
    comments: str = field(default_factory=str, init=False)

    data: np.ndarray = field(repr=False)
    modality: Modality = field(default=Modality.Raw)
    subjects: int = field(default=1)
    samples: int = field(default=0)
    rois: int = field(default=0)

    labels: List[str] = field(init=True, default_factory=list, repr=True)

    tr: float = field(
        default=None, metadata={"unit": "Seconds", "modality": [Modality.fMRI]}
    )
    fs: float = field(
        default=None,
        metadata={
            "unit": "Hertz",
            "modality": [Modality.Raw, Modality.EEG, Modality.MEG],
        },
    )

    def __post_init__(self):
        # Add an empty dimension if it's single-subject
        if len(self.data.shape) == 2:
            self.data = np.expand_dims(self.data, axis=0)

        if self.samples == 0 or self.rois == 0:
            self.subjects, self.rois, self.samples = np.shape(self.data)

    def __iadd__(self, new_data: Union["Dataset", np.ndarray]) -> "Dataset":
        """Overload operator '+='."""
        if isinstance(new_data, np.ndarray):
            new_subjects = new_rois = new_samples = 1
            if len(new_data.shape) == 2:
                new_rois, new_samples = new_data.shape
                new_data = np.expand_dims(new_data, axis=0)
            else:
                new_subjects, new_rois, new_samples = new_data.shape

            if self.rois == new_rois and self.samples == new_samples:
                self.data = np.vstack([self.data, new_data])
                self.subjects += new_subjects

        return self

    def __getitem__(self, index: int) -> Optional[np.ndarray]:
        if index > self.subjects:
            raise IndexError("Index out of boundaries.")

        return self.data[index, :, :]

    def to_json(fname: str) -> bool:
        with open("dataset1.json", "w") as fp:
            json.dump(ds, fp, cls=DatasetEncoder, indent=4)

    def from_json(fname: str) -> "Dataset":
        pass


class DatasetEncoder(json.JSONEncoder):
    """JSON encoder for our Dataset class."""

    def default(self, obj):
        if isinstance(obj, Dataset):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    ss = SlidingWindow()

    for pair in ss:
        # slide_id, start, end, roi1, roi2 = pair
        slide_id, start, end = pair

        print(pair)

    import sys

    sys.exit(0)

    n_subjects = 10
    n_rois = 32
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)
    data = rng.rand(n_rois, n_samples)
    data2 = rng.rand(n_rois, n_samples)

    ds = Dataset(data, modality=Modality.fMRI, tr=1.5)
    ds.labels = ["a", "b"]

    ds += data2

    conn = Correlation()
    conn(ds, ss)

    print(conn)
