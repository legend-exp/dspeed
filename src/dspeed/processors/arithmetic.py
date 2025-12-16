from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32[:])",
        "void(float64[:], float64, float64[:])",
    ],
    "(n),()->()",
    **nb_kwargs,
)
def mean_below_threshold(w_in: np.ndarray, threshold: float, result: float) -> None:
    """Calculate the mean of waveform values that are below a threshold.

    Parameters
    ----------
    w_in
        the input waveform.
    threshold
        the threshold value. Only waveform values below this threshold
        are included in the mean calculation.
    result
        the mean of all values in w_in that are below the threshold.
        Returns NaN if no values are below the threshold.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_mean_below_threshold:
          function: mean_below_threshold
          module: dspeed.processors
          args:
            - waveform
            - 100
            - wf_mean_below_threshold
          unit:
            - ADC
    """
    result[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(threshold):
        return

    total = 0.0
    count = 0

    for i in range(len(w_in)):
        if w_in[i] < threshold:
            total += w_in[i]
            count += 1

    if count == 0:
        return

    result[0] = total / count
