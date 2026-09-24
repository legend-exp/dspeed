"""Processors performing logarithmic consistency checks on waveforms."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs
from .utils import contains_nan


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def log_check(w_in: np.ndarray, w_log: np.ndarray) -> None:
    """
    Calculate the logarithm of the waveform if all its values
    are positive; otherwise, return NaN.

    Parameters
    ----------
    w_in
        the input waveform.
    w_log
        the output waveform with logged values.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_logged:
          function: log_check
          module: dspeed.processors
          args:
            - "wf_blsub[2100:]"
            - wf_logged
    """
    w_log[:] = np.nan

    if contains_nan(w_in):
        return

    for i in range(len(w_in)):
        if w_in[i] <= 0:
            return

    for i in range(len(w_in)):
        w_log[i] = np.log(w_in[i])
