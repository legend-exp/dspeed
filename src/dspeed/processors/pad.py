"""Pad the start and end of a vector input to a fixed length output."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
    ["void(float32[:], int64, float32, float32, float32, float32[:])", "void(float64[:], int64, float64, float64, float64, float64[:])"],
    "(n),(),(),(),(),(m)",
    **nb_kwargs,
)
def pad(w_in: np.ndarray, len_in: int, offset: float, start_val: float, end_val: float, w_out: np.ndarray) -> None:
    """Pad the start, up to offset, and end of a vector input to a fixed length output.

    Note
    ----
    The length of the output waveform is determined by the length of `w_out`
    rather than an input parameter.

    Parameters
    ----------
    w_in
        the input variable length array
    len_in
        the length of the variable length input
    offset
        the integer offset of the start of the input; the length of the start values
    start_val
        the value to use for padding before the start
    end_val
        the value to use for padding after the end
    w_out
        the padded waveform

    Examples
    --------
    .. code-block:: yaml

        trapEftp:
          function: dspeed.processors.pad
          args:
            - vec_in
            - len(vec_in)
            - 20
            - vec_in[0]
            - vec_in[-1]
            - aoa_out(shape=100)
    """
    w_out[:] = np.nan

    if np.isnan(w_in[:len_in]).any() or np.isnan(offset):
        return

    if len_in>len(w_in):
        raise DSPFatal("Length longer than input array")

    i_beg = int(offset)
    if i_beg != offset:
        raise DSPFatal("Offset must be an integer value")
    i_end = i_beg + len_in

    w_out[:i_beg] = start_val
    w_out[i_beg:i_end] = w_in[:len_in]
    w_out[i_end:] = end_val
