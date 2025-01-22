from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def inject_gumbel(
    wf_in: np.ndarray, a: float, t0: float, beta: float, wf_out: np.ndarray
) -> None:
    """
    Injects a Gumbel distribution into the waveform `wf_in`, modifying it in place in `wf_out`.

    Parameters:
    - wf_in: Input waveform (1D array).
    - a: Amplitude of the Gumbel distribution.
    - t0: Temporal centroid of the Gumbel distribution.
    - beta: Scale parameter (controls spread of the Gumbel distribution).
    - wf_out: Output waveform (1D array), modified by adding the Gumbel distribution.
    """

    wf_out[:] = np.nan

    # Early exit if any of the inputs contain NaN values (invalid inputs).
    if np.isnan(wf_in).any() or np.isnan(a) or np.isnan(t0) or np.isnan(beta):
        return

    wf_out[:] = wf_in[:]

    # Define the range of indices over which the Gumbel distribution will be applied.
    # Start injecting the distribution at 2 beta below the centroid (t0).
    start = t0
    mu = t0 + (2 * beta)
    end = mu + (8 * beta)

    # Ensure the range is within valid waveform boundaries.
    if start < 0:
        start = 0
    if end > len(wf_in):
        end = len(wf_in)

    # Loop through the specified range and add the Gumbel distribution to wf_out.
    for i in range(start, end):

        z = (i - mu) / beta
        wf_out[i] += (a / beta) * np.exp(-(z + np.exp(-z)))


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),(),(),()->(n)",
    **nb_kwargs,
)
def inject_general_logistic(
    wf_in: np.ndarray,
    a: float,
    t0: float,
    rt: float,
    q: float,
    v: float,
    decay: float,
    wf_out: np.ndarray,
) -> None:
    r"""Inject sigmoid pulse into existing waveform to simulate pileup.

    .. math::

        s(t) = \frac{A}{(1 + q \exp[-4 \log(99) (t - t_0 - t_r/2) / t_r])^{1/v}}
                e^{-(t-t_0)/\tau}

    Parameters
    ----------
    wf_in
        the input waveform.
    a
        the amplitude :math:`A` of the injected waveform.
    t0
        the position :math:`t_0` of the injected waveform.
    rt
        the rise time :math:`t_r` of the injected waveform.
    q
        shaping parameter of the logistic function.
    v
        shaping parameter of the logistic function.
    decay
        the decay parameter :math:`\tau` of the injected waveform.
    wf_out
        the output waveform.
    """

    wf_out[:] = np.nan

    # Early exit if any of the inputs contain NaN values (invalid inputs).
    if (
        np.isnan(wf_in).any()
        or np.isnan(a)
        or np.isnan(t0)
        or np.isnan(rt)
        or np.isnan(q)
        or np.isnan(v)
        or np.isnan(decay)
    ):
        return

    wf_out[:] = wf_in[:]
    rise = 4 * np.log(99) / rt

    for t in range(len(wf_out)):
        wf_out[t] = wf_out[t] + a / (
            (1 + q * np.exp(-rise * (t - t0 - rt / 2))) ** (1 / v)
        ) * np.exp(-(1 / decay) * (t - t0))
