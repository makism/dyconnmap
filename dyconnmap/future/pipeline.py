"""Pipeline backend."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from dataclasses import dataclass
from typing import Any, List, Optional

from .dataset import Dataset


@dataclass
class Pipeline:
    """A simple pipeline; to group and stack methods together."""

    def __init__(self, stages: Optional[List[Any]] = None) -> None:
        """Construct a pipeline."""
        self.stages = stages

    def run(self, dataset: Optional["Dataset"] = None) -> None:
        """Execute the pipeline."""
        pass
