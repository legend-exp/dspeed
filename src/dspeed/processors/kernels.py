from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32, float32, float32[:])", "void(float64, float64, float64[:])"],
    "(),(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def t0_filter(rise: int, fall: int, kernel: np.array) -> None:
    """Apply a modified, asymmetric trapezoidal filter to the waveform.

    Parameters
    ----------
    rise
        the length of the rise section. This is the linearly increasing
        section of the filter that performs a weighted average.
    fall
        the length of the fall section. This is the simple averaging part
        of the filter.
    kernel
        the output kernel

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        t0_filter:
          function: t0_filter
          module: dspeed.processors
          args:
            - "128*ns"
            - "2*us"
            - t0_filter
          init_args:
            - "128*ns"
            - "2*us"
    """
    if rise < 0:
        raise DSPFatal("The length of the rise section must be positive")

    if fall < 0:
        raise DSPFatal("The length of the fall section must be positive")

    if len(kernel) != rise + fall:
        raise DSPFatal("The length of the output kernel must equal rise+fall")

    for i in range(int(rise)):
        kernel[i] = 2 * (int(rise) - i) / (rise * (rise + 1))
    for i in range(int(rise), len(kernel), 1):
        kernel[i] = -1 / fall


@guvectorize(
    ["void(float32[:])", "void(float64[:])"],
    "(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def moving_slope(kernel):
    """Calculates the linear slope of kernel

    Parameters
    ----------
    length
        the length of the section to calculate slope
    kernel
        the output kernel

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        kern_slopes:
          function: moving_slope
          module: dspeed.processors
          args:
            - 12
            - kern_slopes
    """
    length = len(kernel)

    sum_x = length * (length + 1) / 2
    sum_x2 = length * (length + 1) * (2 * length + 1) / 6

    kernel[:] = (np.arange(1, length + 1, 1) * length) - (np.ones(length) * sum_x)
    kernel[:] /= length * sum_x2 - sum_x * sum_x
    kernel[:] = kernel[::-1]


@guvectorize(
    ["void(float32, float32[:])", "void(float64, float64[:])"],
    "(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def step(weight_pos: int, kernel: np.array) -> None:
    """Process waveforms with a step function.

    Parameters
    ----------
    weight_pos
        relative weight of positive step side.
    kernel
        output kernel

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        kern_step:
          function: step
          module: dspeed.processors
          args:
            - 16
            - kern_step
    """

    x = np.arange(len(kernel))
    kernel[:] = np.piecewise(
        x,
        [
            ((x >= 0) & (x < len(kernel) / 4)),
            ((x >= len(kernel) / 4) & (x < 3 * len(kernel) / 4)),
            ((x >= 3 * len(kernel) / 4) & (x < len(kernel))),
        ],
        [-1, 1, -1],
    )
