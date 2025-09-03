from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], int32, int32, float32[:])",
        "void(float64[:], float64[:], int32, int32, float64[:])",
    ],
    "(n),(m),(),(),(n)",
    **nb_kwargs,
)
def wf_correction(
    w_in: np.ndarray, w_corr: np.ndarray, start_idx: int, stop_idx: int, w_out: np.ndarray
) -> None:

    if np.isnan(w_in).any():
        return

    if np.isnan(w_corr).any():
        return

    if np.isnan(start_idx):
        raise DSPFatal("start_idx is nan")
    if start_idx < 0:
        raise DSPFatal("start_idx must be positive")
    if start_idx > len(w_in):
        raise DSPFatal("start_idx must be shorter than input waveform size")

    if np.isnan(stop_idx):
        raise DSPFatal("stop_idx is nan")
    if stop_idx <= 0:
        raise DSPFatal("stop_idx must be positive")
    if stop_idx > len(w_in):
        raise DSPFatal("stop_idx must be shorter than input waveform size")

    if start_idx >= stop_idx:
        raise DSPFatal("start_idx must be smaller than stop_idx")
    if (stop_idx - start_idx) > len(w_corr):
        raise DSPFatal("stop_idx - start_idx must be smaller than len(w_corr)")

    w_out[:] = w_in[:]

    w_out[start_idx:stop_idx] = w_in[start_idx:stop_idx] - w_corr[:stop_idx - start_idx]
