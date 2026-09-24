"""Processors for applying moving window average convolution to waveforms."""

from __future__ import annotations

import numba
import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs
from .utils import contains_nan, nb_kwargs_util


@numba.njit(**nb_kwargs_util)
def _mw_pass(src: np.ndarray, dst: np.ndarray, length: float, reverse: bool) -> None:
    """One moving-average pass of :func:`moving_window_multi`: dst <- window(src), right to
    left if ``reverse``. The running value is kept in a local of dst's dtype (same arithmetic
    and rounding as accumulating in dst)."""
    n = len(src)
    lw = int(length)
    if reverse:
        last = src[n - 1]
        acc = last
        dst[n - 1] = acc
        for i in range(1, lw, 1):
            acc = acc + (src[n - 1 - i] - last) / length
            dst[n - 1 - i] = acc
        for i in range(lw, n, 1):
            acc = acc + (src[n - 1 - i] - src[n - 1 - i + lw]) / length
            dst[n - 1 - i] = acc
    else:
        first = src[0]
        acc = first
        dst[0] = acc
        for i in range(1, lw):
            acc = acc + (src[i] - first) / length
            dst[i] = acc
        for i in range(lw, n):
            acc = acc + (src[i] - src[i - lw]) / length
            dst[i] = acc


@numba.njit(**nb_kwargs_util)
def _moving_window_multi_core(
    w_in: np.ndarray,
    length: float,
    num_mw: int,
    mw_type: int,
    w_out: np.ndarray,
    w_buf: np.ndarray,
) -> None:
    """Body of :func:`moving_window_multi` with the scratch buffer ``w_buf``
    (same shape as ``w_in``) supplied by the caller, so that callers that
    cannot allocate (e.g. CUDA device code) can use it."""
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

    # passes alternate between w_buf and w_out (pass 0 reads w_in) so that the last one writes
    # w_out: no per-pass copy. Each pass is the same computation as before (see _mw_pass).
    nmw = int(num_mw)
    for k in range(0, nmw, 1):
        reverse = ((k % 2 == 1) & (mw_type == 0)) | (mw_type == 2)
        to_out = (nmw - 1 - k) % 2 == 0
        if k == 0:
            if to_out:
                _mw_pass(w_in, w_out, length, reverse)
            else:
                _mw_pass(w_in, w_buf, length, reverse)
        elif to_out:
            _mw_pass(w_buf, w_out, length, reverse)
        else:
            _mw_pass(w_out, w_buf, length, reverse)


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

    w_out[0] = w_in[0]
    for i in range(1, int(length)):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[0]) / length
    for i in range(int(length), len(w_in)):
        w_out[i] = w_out[i - 1] + (w_in[i] - w_in[i - int(length)]) / length


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

    w_out[-1] = w_in[-1]
    for i in range(1, int(length), 1):
        w_out[len(w_in) - 1 - i] = (
            w_out[len(w_in) - i] + (w_in[len(w_in) - 1 - i] - w_out[-1]) / length
        )
    for i in range(int(length), len(w_in), 1):
        w_out[len(w_in) - 1 - i] = (
            w_out[len(w_in) - i]
            + (w_in[len(w_in) - 1 - i] - w_in[len(w_in) - 1 - i + int(length)]) / length
        )


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
    _moving_window_multi_core(w_in, length, num_mw, mw_type, w_out, np.empty_like(w_in))


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

    for i in range(min(len(w_out), len(w_in) - int(length))):
        w_out[i] = w_in[int(length) + i] - w_in[i]
        w_out[i] /= length
