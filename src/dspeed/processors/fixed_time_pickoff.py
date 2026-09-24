"""Processors computing fixed-time pick-off metrics."""

from __future__ import annotations

import numba
import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs
from .utils import contains_nan, nb_kwargs_util


@numba.njit(**nb_kwargs_util)
def _fixed_time_pickoff_core(
    w_in: np.ndarray,
    t_in: float,
    mode_in: np.int8,
    a_out: np.ndarray,
    u: np.ndarray,
    w2: np.ndarray,
) -> None:
    """Body of :func:`fixed_time_pickoff` with the float64 scratch buffers for
    the cubic-spline mode (``u``, ``w2``, each ``len(w_in)``) supplied by the
    caller, so that callers that cannot allocate (e.g. CUDA device code) can use
    it. Modes are compared as character codes: i=105, n=110, f=102, c=99,
    l=108, h=104, s=115."""
    a_out[0] = np.nan

    if contains_nan(w_in) or np.isnan(t_in):
        return

    if t_in < 0 or t_in > len(w_in) - 1:
        return

    i_in = int(t_in)
    if i_in == t_in:
        a_out[0] = w_in[i_in]
        return

    t0 = t_in - i_in
    t1 = 1 - t0

    if mode_in == 105:  # Index
        raise DSPFatal("fixed_time_pickoff requires integer t_in when using mode 'i'")
    elif mode_in == 110:  # Nearest-neighbor
        a_out[0] = w_in[i_in] if t0 < 0.5 else w_in[i_in + 1]
    elif mode_in == 102:  # Floor
        a_out[0] = w_in[i_in]
    elif mode_in == 99:  # Ceiling
        a_out[0] = w_in[i_in + 1]
    elif mode_in == 108:  # linear
        a_out[0] = t1 * w_in[i_in] + t0 * w_in[i_in + 1]
    elif mode_in == 104:  # Cubic hermite
        m0 = w_in[1] - w_in[0] if i_in == 0 else (w_in[i_in + 1] - w_in[i_in - 1]) / 2
        m1 = (
            w_in[-1] - w_in[-2]
            if i_in == len(w_in) - 2
            else (w_in[i_in + 2] - w_in[i_in]) / 2
        )
        a_out[0] = (
            (-2 * t1**3 + 3 * t1**2) * w_in[i_in]
            + (-2 * t0**3 + 3 * t0**2) * w_in[i_in + 1]
            - (t1**3 - t1**2) * m0
            + (t0**3 - t0**2) * m1
        )
    elif mode_in == 115:  # Cubic spline
        for i in range(len(w_in)):  # u, w2 (second derivative values) start at zero
            u[i] = 0
            w2[i] = 0

        for i in range(1, len(w_in) - 1):
            p = 0.5 * w2[i - 1] + 2
            w2[i] = -0.5 / p
            u[i] = w_in[i + 1] - 2 * w_in[i] + w_in[i - 1]
            u[i] = (3 * u[i] - 0.5 * u[i - 1]) / p
        for i in range(len(w_in) - 2, i_in - 1, -1):
            w2[i] = w2[i] * w2[i + 1] + u[i]

        a_out[0] = (
            t1 * w_in[i_in]
            + t0 * w_in[i_in + 1]
            + ((t1**3 - t1) * w2[i_in] + (t0**3 - t0) * w2[i_in + 1]) / 6.0
        )
    else:
        raise DSPFatal("Unrecognized interpolation mode")


@guvectorize(
    [
        "void(float32[:], float32, char, float32[:])",
        "void(float64[:], float64, char, float64[:])",
    ],
    "(n),(),()->()",
    **nb_kwargs,
)
def fixed_time_pickoff(w_in: np.ndarray, t_in: float, mode_in: np.int8, a_out: float):
    """Pick off the waveform value at the provided time.

    For non-integral times, interpolate between samples using the method
    selected using `mode_in`. If the provided index `t_in` is out of range,
    return :any:`numpy.nan`.

    Parameters
    ----------
    w_in
        the input waveform.
    t_in
        the waveform index to pick off.
    mode_in
        character selecting which interpolation method to use. Note this
        must be passed as a ``int8``, e.g. ``ord('i')``. Options:

        * ``i`` -- integer `t_in`; equivalent to
          :func:`~.dsp.processors.fixed_sample_pickoff`
        * ``n`` -- nearest-neighbor interpolation; defined at all values,
          but not continuous
        * ``f`` -- floor, or value at previous neighbor; defined at all
          values but not continuous
        * ``c`` -- ceiling, or value at next neighbor; defined at all values,
          but not continuous
        * ``l`` -- linear interpolation; continuous at all values, but not
          differentiable
        * ``h`` -- Hermite cubic spline interpolation; continuous and
          differentiable at all values but not twice-differentiable
        * ``s`` -- natural cubic spline interpolation; continuous and
          twice-differentiable at all values. This method is much slower
          than the others because it utilizes the entire input waveform!
    a_out
        the output pick-off value.

    Examples
    --------
    .. code-block:: yaml

        trapEftp:
          function: fixed_time_pickoff
          module: dspeed.processors
          args:
            - wf_trap
            - "tp_0+10*us"
            - "'h'"
            - trapEftp
    """
    if mode_in == 115:  # "s": cubic spline needs two scratch buffers
        u = np.empty(len(w_in))
        w2 = np.empty(len(w_in))
    else:
        u = w2 = np.empty(0)
    _fixed_time_pickoff_core(w_in, t_in, mode_in, a_out, u, w2)
