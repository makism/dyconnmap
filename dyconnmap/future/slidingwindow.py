"""Sliwing Window."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from dataclasses import dataclass, field

import numpy as np

from .dynamicwindow import DynamicWindow


@dataclass
class SlidingWindow(DynamicWindow):
    """Sliding window class."""

    step: int = field(default=1, init=True)

    def prepare(self, **kwargs) -> None:
        """Prepare the sliding window."""
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
        """Return a generator to iterate over the constructed windows."""
        return iter(self.pairs)
