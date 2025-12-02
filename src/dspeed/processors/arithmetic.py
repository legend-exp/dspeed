from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:])",
        "void(float64[:], float64[:])",
    ],
    "(n)->()",
    **nb_kwargs,
)
def sum(w_in: np.ndarray, result: float) -> None:
    """Sum the waveform values from index a to b.

    Parameters
    ----------
    w_in
        the input waveform
    result
        the sum of all values in w_in.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_sum:
          function: sum
          module: dspeed.processors
          args:
            - waveform
            - wf_sum
          unit:
            - ADC
    """
    result[0] = np.nan

    if np.isnan(w_in).any():
        return

    start = 0
    end = len(w_in) - 1

    if start < 0:
        start = 0
    if end > len(w_in) - 1:
        end = len(w_in) - 1
    if start > end:
        return

    total = 0.0
    for i in range(start, end + 1):
        total += w_in[i]

    result[0] = total


@guvectorize(
    [
        "void(float32[:], float32[:])",
        "void(float64[:], float64[:])",
    ],
    "(n)->()",
    **nb_kwargs,
)
def mean(w_in: np.ndarray, result: float) -> None:
    """Calculate the mean of waveform values.

    Parameters
    ----------
    w_in
        the input waveform.
    result
        the mean of all values in w_in.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_mean:
          function: mean
          module: dspeed.processors
          args:
            - waveform
            - wf_mean
          unit:
            - ADC
    """
    result[0] = np.nan

    if np.isnan(w_in).any():
        return

    start = 0
    end = len(w_in) - 1

    if start < 0:
        start = 0
    if end > len(w_in) - 1:
        end = len(w_in) - 1
    if start > end:
        return

    total = 0.0
    for i in range(start, end + 1):
        total += w_in[i]

    result[0] = total / (end - start + 1)
