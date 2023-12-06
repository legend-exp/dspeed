from __future__ import annotations

import numpy as np
from numba import guvectorize
from scipy.signal import fftconvolve

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
        ["void(float32[:], float32[:], char, float32[:])", 
         "void(float64[:], float64[:], char, float64[:])"],
        "(n),(m),(),(p)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
def convolve_wf(w_in: np.ndarray, kernel:np.array, mode_in: np.int8, w_out: np.ndarray) -> None: #
    """
    Parameters
    ----------
    w_in
        the input waveform.
    kernel
        the kernel to convolve with
    mode 
        mode of convolution options are f : full, v : valid or s : same, 
        explained here: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    w_out
        the filtered waveform.
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return
    
    if np.isnan(kernel).any():
        return

    if len(kernel) > len(w_in):
        raise DSPFatal("The filter is longer than the input waveform")
    
    if chr(mode_in) == "f":
        mode = "full"
    elif chr(mode_in) == "v":
        mode = "valid"
    elif chr(mode_in) == "s":
        mode = "same"
    else:
        raise DSPFatal("Invalid mode")
    
    w_out[:] = np.convolve(w_in, kernel, mode=mode)


@guvectorize(
        ["void(float32[:], float32[:], char, float32[:])", 
         "void(float64[:], float64[:], char, float64[:])"],
        "(n),(m),(),(p)",
        **nb_kwargs(
            cache=False,
            forceobj=True,
        ),
    )
def fft_convolve_wf(w_in: np.ndarray, kernel:np.array, mode_in: np.int8, w_out: np.ndarray) -> None: #
    """
    Parameters
    ----------
    w_in
        the input waveform.
    kernel
        the kernel to convolve with
    mode 
        mode of convolution options are f : full, v : valid or s : same, 
        explained here: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    w_out
        the filtered waveform.
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return
    
    if np.isnan(kernel).any():
        return

    if len(kernel) > len(w_in):
        raise DSPFatal("The filter is longer than the input waveform")
    
    if chr(mode_in) == "f":
        mode = "full"
    elif chr(mode_in) == "v":
        mode = "valid"
    elif chr(mode_in) == "s":
        mode = "same"
    else:
        raise DSPFatal("Invalid mode")
    
    w_out[:] = fftconvolve(w_in, kernel, mode=mode)


