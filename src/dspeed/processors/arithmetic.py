from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64[:])",
    ],
    "(n),(),()->()",
    **nb_kwargs,
)
def sum(w_in: np.ndarray, a: float, b: float, result: float) -> None:
    """Sum the waveform values from index a to b.

    Parameters
    ----------
    w_in
        the input waveform.
    a
        the starting index (inclusive). If NaN, defaults to 0.
    b
        the ending index (inclusive). If NaN, defaults to len(w_in) - 1.
    result
        the sum of w_in[a:b+1].

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_sum:
          function: sum
          module: dspeed.processors
          args:
            - waveform
            - "np.nan"
            - "np.nan"
            - wf_sum
          unit:
            - ADC
    """
    result[0] = np.nan

    if np.isnan(w_in).any():
        return

    start = 0 if np.isnan(a) else int(a)
    end = len(w_in) - 1 if np.isnan(b) else int(b)

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
        "void(float32[:], float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64[:])",
    ],
    "(n),(),()->()",
    **nb_kwargs,
)
def mean(w_in: np.ndarray, a: float, b: float, result: float) -> None:
    """Calculate the mean of waveform values from index a to b.

    Parameters
    ----------
    w_in
        the input waveform.
    a
        the starting index (inclusive). If NaN, defaults to 0.
    b
        the ending index (inclusive). If NaN, defaults to len(w_in) - 1.
    result
        the mean of w_in[a:b+1], which is sum(w_in[a:b+1]) / (b - a + 1).

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_mean:
          function: mean
          module: dspeed.processors
          args:
            - waveform
            - "np.nan"
            - "np.nan"
            - wf_mean
          unit:
            - ADC
    """
    result[0] = np.nan

    if np.isnan(w_in).any():
        return

    start = 0 if np.isnan(a) else int(a)
    end = len(w_in) - 1 if np.isnan(b) else int(b)

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
