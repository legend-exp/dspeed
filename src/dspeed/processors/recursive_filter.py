from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32, float32, float32[:])",
        "void(float64[:], float64[:], float64[:], float64, float64, float64[:])",
    ],
    "(n),(p),(q),(),()->(n)",
    nopython=True,
    **nb_kwargs,
)
def recursive_filter(w_in, a, b, init_in, init_out, w_out):
    r"""
    Apply a recursive filter using ``a`` and ``b`` as the
    feedforward and feedback coefficients, respectively:

    .. math::
        w_{out}[i] = (a[0]*w_{in}[i] + a[1]*w_{in}[i-1] + a[2]*w_{in}[i-2] + ... - b[1]*w_{out}[i-1] - b[2]*w_{out}[i-2] - ...)/b[0]

    When performing a z-transform, ``a`` and ``b`` represent the polynomial coefficients of the numerator and denominator, respectively:

    ..math::
        F(z) = \frac{a[0] + a[1]*z^{-1} + a[2]*z^{-2} + ...}{b[0] + b[1]*z^{-1} + b[2]*z^{-2} + ...}

    See `scipy.signal.iir_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html>`_
    for an example of how to generate these coefficients (using ``'ba'`` output).

    The initial conditions for the filter are set using ``init_in`` and ``init_out`` to pad the starts of ``w_in`` and ``w_out``.
    These should be set appropriately to minimize edge effects.

    Parameters
    ----------
    w_in
        the input waveform
    a
        feedforward (numerator) coefficients
    b
        feedback (denominator) coefficients
    init_in
        initial value to pad at front of w_in
    init_out
        initial value of w_out memory
    w_out
        the output waveform
    """

    w_out[:] = np.nan
    if (
        np.isnan(w_in).any()
        or np.isnan(a).any()
        or np.isnan(b).any()
        or np.isnan(init_in)
        or np.isnan(init_out)
    ):
        return

    if len(b) == 0:
        raise DSPFatal("b cannot be scalar")
    if len(w_in) <= len(b):
        raise DSPFatal(
            f"The length of the waveform must be larger than {len(b)} for the filter to work safely"
        )

    # circular buffer; make float64 to mitigate numerical instabilities
    circ_buf = np.full(len(b), init_out, dtype="float64")

    for i in range(len(w_out)):
        i_buf = i % len(circ_buf)

        circ_buf[i_buf] = 0
        # feed forward
        for j in range(len(a)):
            if j <= i:
                circ_buf[i_buf] += a[j] * w_in[i - j]
            else:
                circ_buf[i_buf] += a[j] * init_in

        # feed back
        for j in range(1, len(b)):
            circ_buf[i_buf] -= b[j] * circ_buf[i_buf - j]

        circ_buf[i_buf] /= b[0]

        w_out[i] = circ_buf[i_buf]
