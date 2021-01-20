"""Sliwing Window."""
import numpy as np

from dynamicwindow import DynamicWindow

from typing import List, Type, Union, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class SlidingWindow(DynamicWindow):
    """Sliding window class."""

    step: int = field(default=1, init=True)

    def prepare(self, **kwargs) -> None:
        super().prepare(**kwargs)

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
