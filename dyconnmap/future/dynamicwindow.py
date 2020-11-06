from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Type, Union, Optional, Tuple, Dict


@dataclass
class DynamicWindow(ABC):
    """Base dynamic window class."""

    samples: int
    rois: int
    slides: int = field(init=False)
    window_length: int
    pairs: Optional[List[Tuple[int, int, int]]] = field(
        default=None,
        metadata={
            "description": "The timing pairs, in which the estimator with will fed with."
        },
    )

    @abstractmethod
    def __post_init__(self):
        ...

    @abstractmethod
    def __iter__(self):
        ...
