"""Dataset class.

An experimental Dataset container that ties together all the aspects of the
module.
"""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

import numpy as np
import os
import datetime
from dataclasses import dataclass, field
from enum import IntEnum

from typing import List, Type, Union, Optional, Tuple, Dict
import json


class Modality(IntEnum):
    """Modalities."""

    Any = 0
    Raw = 1
    FMRI = 2
    EEG = 3
    MEG = 4


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class Dataset:
    """Dataset."""

    version: float = field(default=1.0, init=False)
    date_time: datetime.datetime = field(default=datetime.datetime.now(), init=False)
    comments: str = field(default_factory=str, init=False)

    data: np.ndarray = field(repr=False)
    modality: Modality = field(default=Modality.Raw)
    subjects: int = field(default=1)
    samples: int = field(default=0)
    rois: int = field(default=0)

    labels: List[str] = field(init=True, default_factory=list)

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
        self.version = 1.0
        self.date_time = datetime.datetime.now()

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

    def __eq__(self, other: "Dataset"):
        """Test for equality against another Dataset object."""

        if not isinstance(other, Dataset):
            return False

        for key in self.__dict__.keys():
            if key == "data":
                compare = self.data == other.data
                equal_arrays = compare.all()
                if not equal_arrays:

                    return False
            elif key == "labels":
                if set(self.labels) != set(other.labels):
                    return False

            else:
                if key not in other.__dict__:
                    return False

                if self.__dict__[key] != other.__dict__[key]:
                    return False

        return True

    def settings(self) -> Dict[str, int]:
        """Return a dictionary containing important metadata about the Dataset; number of subjects, samples, etc.."""
        return {"subjects": self.subjects, "samples": self.samples, "rois": self.rois}

    def write(self, pathname: str) -> Optional[bool]:
        """Write the Dataset to the given directory."""
        try:
            if not os.path.exists(pathname):
                os.makedirs(pathname)

            json_dataset = f"{pathname}/dataset.json"
            with open(json_dataset, "w") as fp:
                json.dump(self, fp, cls=DatasetEncoder, indent=4)

            for s in range(self.subjects):
                np.savetxt(
                    f"{pathname}/data_subject{s}.csv",
                    self.data[s, :, :],
                    header=",".join(self.labels),
                    comments="",
                    delimiter=",",
                )
        except Exception as err:
            print(err)
            return False

        return True

    @classmethod
    def load(cls, pathname: str) -> "Dataset":
        """Load Dataset from a json file."""
        with open(f"{pathname}/dataset.json", "r") as fp:
            json_data = json.load(fp)

            orig_version, orig_date_time, orig_comments = (
                json_data["version"],
                json_data["date_time"],
                json_data["comments"],
            )

            del json_data["version"]
            del json_data["comments"]
            del json_data["date_time"]

            ds_data = None
            for i in range(json_data["subjects"]):
                m = np.loadtxt(
                    f"{pathname}/data_subject{i}.csv", delimiter=",", skiprows=1
                )

                if ds_data is None:
                    ds_data = m
                else:
                    ds_data = np.stack([ds_data, m], axis=0)
            json_data["data"] = ds_data

            new_ds = Dataset(**json_data)
            new_ds.date_time = datetime.datetime.fromisoformat(orig_date_time)
            new_ds.modality = Modality(new_ds.modality)

            return new_ds

        return None


class DatasetEncoder(json.JSONEncoder):
    """JSON encoder for our Dataset class."""

    def default(self, obj):
        if isinstance(obj, Dataset):
            return obj.__dict__
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            # return obj.tolist()
            return None

        return json.JSONEncoder.default(self, obj)
