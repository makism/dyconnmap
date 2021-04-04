"""Time-Vayring Functional Connectivity Graphs."""
# author Avraam Marimpis <avraam.marimpis@gmail.com>

from dataclasses import dataclass, field

import numpy as np

from .dynamicwindow import DynamicWindow


@dataclass
class TimeVarying(DynamicWindow):
    """Time Varying (Functional Connectivity Graphs)."""

    cc: float = field(
        init=True,
        default=2.0,
        metadata={"description": "The cycle-criterion."},
    )

    # redefine optional
    window_length: int = field(init=False)

    step: int = field(init=True, default=10)

    def prepare(self, **kwargs) -> None:
        """Prepare the time-varying window."""
        super().prepare(**kwargs)

        if "filter_opts" not in kwargs:
            raise Exception(
                "No filter options found; please set them in your Estimator object."
            )

        fs = kwargs["filter_opts"]["fs"]
        fb = kwargs["filter_opts"]["fb"]

        self.window_length = np.int32(np.round((self.cc / fb[0]) * fs))

        if self.window_length >= self.samples:
            raise Exception(
                "The resulting window size is greater than the samples."
            )

        self.slides = np.int32(
            np.round((self.samples - self.window_length) / self.step)
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
