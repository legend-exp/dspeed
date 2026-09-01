"""Processors for applying moving window average convolution to waveforms."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import contains_nan
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def moving_window_left(w_in: np.ndarray, length: float, w_out: np.ndarray) -> None:
    """Applies a moving average window to the waveform.

    Note
    ----
    Starts from the left and assumes that the baseline is at zero.

    Parameters
    ----------
    w_in
        the input waveform.
    length
        length of the moving window to be applied.
    w_out
        output waveform after moving window applied.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_mw:
          function: moving_window_left
          module: dspeed.processors
          args:
            - wf_pz
            - "96*ns"
            - wf_mw
    """

    w_out[:] = np.nan

    if contains_nan(w_in):
        return

    if not length >= 0 or not length < len(w_in):
        raise DSPFatal(
            "length is out of range, must be between 0 and the length of the waveform"
        )

    # carry the accumulator in a register (cast back to the element dtype
    # each step so results match the store/reload version bitwise)
    tp = w_in.dtype.type
    acc = w_in[0]
    w_out[0] = acc
    for i in range(1, int(length)):
        acc = tp(acc + (w_in[i] - w_in[0]) / length)
        w_out[i] = acc
    for i in range(int(length), len(w_in)):
        acc = tp(acc + (w_in[i] - w_in[i - int(length)]) / length)
        w_out[i] = acc


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),()->(n)",
    **nb_kwargs,
)
def moving_window_right(w_in: np.ndarray, length: float, w_out: np.ndarray) -> None:
    """Applies a moving average window to the waveform from the right.

    Parameters
    ----------
    w_in
        the input waveform.
    length
        length of the moving window to be applied.
    w_out
        output waveform after moving window applied.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_mw:
          function: moving_window_right
          module: dspeed.processors
          args:
            - wf_pz
            - "96*ns"
            - wf_mw
    """

    w_out[:] = np.nan

    if contains_nan(w_in):
        return

    if not length >= 0 or not length < len(w_in):
        raise DSPFatal(
            "length is out of range, must be between 0 and the length of the waveform"
        )

    # carry the accumulator in a register (cast back to the element dtype
    # each step so results match the store/reload version bitwise)
    tp = w_in.dtype.type
    m = len(w_in)
    acc = w_in[-1]
    w_out[-1] = acc
    for i in range(1, int(length), 1):
        acc = tp(acc + (w_in[m - 1 - i] - w_in[-1]) / length)
        w_out[m - 1 - i] = acc
    for i in range(int(length), m, 1):
        acc = tp(acc + (w_in[m - 1 - i] - w_in[m - 1 - i + int(length)]) / length)
        w_out[m - 1 - i] = acc


@guvectorize(
    [
        "void(float32[:], float32, float32, int32, float32[:])",
        "void(float64[:], float64, float64, int32, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def moving_window_multi(
    w_in: np.ndarray, length: float, num_mw: int, mw_type: int, w_out: np.ndarray
) -> None:
    """Apply a series of moving-average windows to the waveform, alternating
    its application between the left and the right.

    Parameters
    ----------
    w_in
        the input waveform.
    length
        length of the moving window to be applied.
    num_mw
        the number of moving windows.
    mw_type
        - ``0`` -- alternate moving windows right and left
        - ``1`` -- only left
        - ``2`` -- only right
    w_out
        the windowed waveform.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        curr_av:
          function: moving_window_multi
          module: dspeed.processors
          args:
            - curr
            - "96*ns"
            - 3
            - 0
            - curr_av
          unit: ADC/sample
    """
    w_out[:] = np.nan

    if contains_nan(w_in):
        return

    if np.floor(length) != length:
        raise DSPFatal("The length of the moving window must be an integer")

    if np.floor(num_mw) != num_mw:
        raise DSPFatal("The number of moving windows must be an integer")

    if int(length) < 0 or int(length) >= len(w_in):
        raise DSPFatal("The length of the moving window is out of range")

    if int(num_mw) < 0:
        raise DSPFatal("The number of moving windows much be positive")

    # ping-pong between w_out and one scratch buffer instead of copying the
    # full waveform after every pass; the start buffer is chosen so the last
    # pass always writes into w_out
    n = len(w_in)
    tp = w_in.dtype.type
    scratch = np.empty(n, w_in.dtype)
    if int(num_mw) % 2 == 1:
        dst, alt = w_out, scratch
    else:
        dst, alt = scratch, w_out

    w_buf = w_in
    for k in range(0, int(num_mw), 1):
        if ((k % 2 == 1) & (mw_type == 0)) | (mw_type == 2):
            acc = w_buf[-1]
            dst[-1] = acc
            for i in range(1, int(length), 1):
                acc = tp(acc + (w_buf[n - 1 - i] - w_buf[-1]) / length)
                dst[n - 1 - i] = acc
            for i in range(int(length), n, 1):
                acc = tp(
                    acc + (w_buf[n - 1 - i] - w_buf[n - 1 - i + int(length)]) / length
                )
                dst[n - 1 - i] = acc
        else:
            # carry the accumulator in a register (cast back to the element
            # dtype each step so results match the store/reload version
            # bitwise); indexing dst[i-1] forces a reload since numba cannot
            # prove dst and w_buf don't alias
            acc = w_buf[0]
            dst[0] = acc
            for i in range(1, int(length)):
                acc = tp(acc + (w_buf[i] - w_buf[0]) / length)
                dst[i] = acc
            for i in range(int(length), n):
                acc = tp(acc + (w_buf[i] - w_buf[i - int(length)]) / length)
                dst[i] = acc
        w_buf = dst
        dst, alt = alt, dst


@guvectorize(
    ["void(float32[:], float32, float32[:])", "void(float64[:], float64, float64[:])"],
    "(n),(),(m)",
    **nb_kwargs,
)
def avg_current(w_in: np.ndarray, length: float, w_out: np.ndarray) -> None:
    """Calculate the derivative of a waveform, averaged across `length` samples.

    Parameters
    ----------
    w_in
        the input waveform.
    length
        length of the moving window to be applied.
    w_out
        output waveform after derivation.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        curr:
          function: avg_current
          module: dspeed.processors
          args:
            - wf_pz
            - 1
            - "curr(len(wf_pz)-1, f)"
          unit: ADC/sample
    """

    w_out[:] = np.nan

    if contains_nan(w_in):
        return

    if not length >= 0 or not length < len(w_in):
        raise DSPFatal(
            "length is out of range, must be between 0 and the length of the waveform"
        )

    w_out[:] = w_in[int(length) :] - w_in[: -int(length)]
    w_out /= length
