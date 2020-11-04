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
    FMRI = 2
    EEG = 3
    MEG = 4


@dataclass
class DynamicWindow(ABC):
    """Base dynamic window class."""

    samples: int
    rois: int
    slides: int = field(init=False)
    window_length: int
    pairs: Optional[List[Tuple[int, int, int]]] = field(default=None)

    @abstractmethod
    def __post_init__(self):
        ...

    @abstractmethod
    def __iter__(self):
        ...


@dataclass
class SlidingWindow(DynamicWindow):
    """Sliding Window."""

    step: int = field(init=True, default=10)

    def __post_init__(self):
        self.samples = 128
        self.rois = 32
        self.window_length = 10
        self.step = 1
        self.slides = np.int32(
            np.ceil((self.samples - self.window_length) / self.step + 1.0)
        )

        if self.pairs is None:
            self.pairs = [
                (
                    win_id,
                    np.int32(win_id * self.step),
                    np.int32(win_id * self.step + self.window_length),
                )
                for win_id in range(self.slides)
            ]

    def __iter__(self):
        return iter(self.pairs)


@dataclass(frozen=False)
class Estimator(ABC):
    """Base estimator class."""

    dtype: np.dtype = field(default=np.float32)

    jobs: int = field(
        default=multiprocessing.cpu_count(),
        repr=True,
        metadata={"description": "Number of parallel jobs."},
    )

    def __post_init__(self):
        if self.jobs <= 0:
            self.jobs = 1

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

        if window is None:
            cb_func = self.estimate
        else:
            cb_func = self.estimate_for_slides
            pairs = list(window)

        results = Parallel(n_jobs=self.jobs)(
            delayed(cb_func)(
                dataset.data[index, :, :], pairs=pairs, subject_index=index
            )
            for index in np.arange(dataset.subjects)
        )


class Correlation(Estimator):
    """Correlation."""

    def __init__(self):
        super()
        self.dtype = np.float64

    def estimate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        result = np.corrcoef(data)
        return {"index": kwargs["subject_index"], "result": result}


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
        default=None, metadata={"unit": "Seconds", "modality": [Modality.FMRI]}
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
        """Fetch the subject's data specified by `index`."""
        if index > self.subjects:
            raise IndexError("Index out of boundaries.")

        return self.data[index, :, :]

    def to_json(fname: str) -> bool:
        """Write the Dataset to a json file."""
        with open("dataset1.json", "w") as fp:
            json.dump(ds, fp, cls=DatasetEncoder, indent=4)

    def from_json(fname: str) -> "Dataset":
        """Load Dataset from a json file."""
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

    n_subjects = 10
    n_rois = 32
    n_samples = 128

    data = rng.rand(n_subjects, n_rois, n_samples)
    data = rng.rand(n_rois, n_samples)
    data2 = rng.rand(n_rois, n_samples)

    ds = Dataset(data, modality=Modality.FMRI, tr=1.5)
    ds.labels = ["a", "b"]
    print(ds)

    ds += data2

    sw = SlidingWindow(step=10, samples=128, rois=32, window_length=10)
    print(sw)

    conn = Correlation()
    conn(ds, sw)

    print(conn)
