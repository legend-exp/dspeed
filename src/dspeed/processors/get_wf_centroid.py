"""Processors estimating waveform centroids."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs

from ..utils import contains_nan


@guvectorize(
    [
        "void(float32[:], float32, float32[:])",
        "void(float64[:], float64, float64[:])",
    ],
    "(n),()->()",
    **nb_kwargs,
)
def get_wf_centroid(w_in: np.ndarray, shift: int, centroid: int) -> None:
    """Calculate waveform centroid.

    Note
    ----
    This processor calculate the centroid position when provided the convolution product with a step function.

    Parameters
    ----------
    w_in
        the input waveform.
    shift
        shift.
    centroid
        centroid position.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        centroid:
          function: get_wf_centroid
          module: dspeed.processors
          args:
            - waveform
            - shift
            - centroid
    """

    centroid[0] = np.nan

    if contains_nan(w_in):
        return

    if np.isnan(shift):
        raise DSPFatal("shift is nan")
    if shift < 0:
        raise DSPFatal("shift must be positive")
    if shift > len(w_in) - 1:
        raise DSPFatal("shift must be shorter than input waveform size")

    # find the first positive and the last negative sample between the first
    # minimum and the first maximum; if there is no crossing in that window
    # (e.g. noise events where the minimum comes after the maximum) the
    # centroid is undefined -> stays NaN. The old np.where(...)[0][0] on the
    # empty match read out of bounds when boundscheck is disabled.
    lo = w_in.argmin()
    hi = w_in.argmax()

    i_pos = -1
    i_neg = -1
    for i in range(lo, hi):
        if i_pos < 0 and w_in[i] > 0:
            i_pos = i
        if w_in[i] < 0:
            i_neg = i
    if i_pos < 0 or i_neg < 0:
        return

    c_a = i_pos + shift
    c_b = i_neg + shift

    centroid[0] = round((c_a + c_b) / 2)
