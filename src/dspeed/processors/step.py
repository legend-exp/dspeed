from __future__ import annotations

from typing import Callable

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


def step(
    length: int
) -> Callable:
    """Process waveforms with a step function.
    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.
    Parameters
    ----------
    length
        length of the step function.
    JSON Configuration Example
    --------------------------
    .. code-block :: json
        "wf_step": {
            "function": "step",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "wf_step(len(waveform)-16+1, 'f')"],
            "unit": "ADC",
            "init_args": ["16"]
        }
    """

    x = np.arange(length)
    y = np.piecewise(x,
                     [((x >= 0) & (x < length/4)),
                      ((x >= length/4) & (x <= 3*length/4)),
                      ((x > 3*length/4) & (x <= length))],
                     [-1, 1, -1])

    @guvectorize(
        ["void(float32[:], float32[:])",
         "void(float64[:], float64[:])"],
        "(n),(m)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
    def step_out(w_in: np.ndarray, w_out: np.ndarray) -> None:
        """
        Parameters
        ----------
        w_in
            the input waveform.
        w_out
            the filtered waveform.
        """

        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(y) > len(w_in):
            raise DSPFatal("The filter is longer than the input waveform")
        w_out[:] = np.convolve(w_in, y, mode = 'valid')

    return step_out
