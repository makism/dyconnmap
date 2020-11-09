import numpy as np

from dynamicwindow import DynamicWindow

from typing import List, Type, Union, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class TimeVarying(DynamicWindow):
    """Time Varying (Functional Connectivity Graphs)."""

    cc: float = field(
        init=True, default=2.0, metadata={"description": "The cycle-criterion."}
    )

    step: int = field(init=True, default=10)

    def __post_init__(self):
        self.samples = 128
        self.rois = 32
        self.window_length = 10
        # self.step = None
        # self.slides = None
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

        # window_length = np.int32(np.round((cc / fb[0]) * fs))
        # windows = np.int32(np.round((n_samples - window_length) / step))

    def __iter__(self):
        return iter(self.pairs)
