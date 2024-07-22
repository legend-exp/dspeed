from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs
from .convolutions import convolve_damped_oscillator


@guvectorize(
    [
        "void(float32[:], float64, float64, float64, float64, float32[:])",
        "void(float64[:], float64, float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),()->(n)",
    **nb_kwargs(forceobj=True),
)
def inject_damped_oscillation(
    w_in: np.ndarray,
    tau: float,
    omega: float,
    phase: float,
    frac: float,
    w_out: np.ndarray,
) -> None:
    """
    Inject a damped oscillation component/pole into the electronics response

    Parameters
    ----------
    w_in
        the input waveform.
    tau
        time constant of decay
    omega:
        angular frequency of oscillation
    phase:
        phase shift of oscillation
    frac
        fraction of amplitude in injected pole
    w_out
        output waveform after injecting decay component
    """
    if np.isnan(w_in).any():
        return

    if not 0 <= frac <= 1:
        raise DSPFatal("frac must be between zero and one.")

    curr = np.zeros_like(w_out)
    curr[0] = w_in[0]
    curr[1:] = w_in[1:] - w_in[:-1]

    convolve_damped_oscillator(curr, tau, omega, phase, w_out)
    w_out *= frac
    w_out += w_in
