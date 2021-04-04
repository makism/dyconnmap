"""Base Dynamic Window."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DynamicWindow(ABC):
    """Base dynamic window class."""

    __ready = False

    window_length: int
    samples: int = field(default=0)
    rois: int = field(default=0)
    slides: int = field(init=False, default=0)
    pairs: Optional[List[Tuple[int, int, int]]] = field(
        default=None,
        metadata={
            "description": "The timing pairs / windows for the estimator."
        },
    )

    def is_ready(self) -> bool:
        """Return the preparation status of the estimator."""
        return self.__ready

    def prepare(self, **kwargs) -> None:
        """Prepare the dynamic window and initialize the default values."""
        self.samples = kwargs.get("samples", 0)
        self.rois = kwargs.get("rois", 0)
        self.__ready = True

    def settings(self) -> dict:
        """Return the settings for the given estimator."""
        return {
            "window_length": self.window_length,
            "samples": self.samples,
            "rois": self.rois,
            "slides": self.slides,
            "pairs": self.pairs,
        }

    @abstractmethod
    def __iter__(self):
        """Return a generator to iterate over the constructed windows."""
        ...
