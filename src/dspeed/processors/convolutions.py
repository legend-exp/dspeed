"""Convolution-based waveform filter processors."""

from __future__ import annotations

import numba
import numpy as np
from numba import guvectorize
from scipy.signal import fftconvolve

from ..errors import DSPFatal
from ..utils import dspeed_guvectorize
from ..utils import numba_defaults_kwargs as nb_kwargs
from .utils import contains_nan, nb_kwargs_util


@guvectorize(
    [
        "void(float32[:], float32[:], char, float32[:])",
        "void(float64[:], float64[:], char, float64[:])",
    ],
    "(n),(m),(),(p)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def convolve_wf(
    w_in: np.ndarray, kernel: np.array, mode_in: np.int8, w_out: np.ndarray
) -> None:  #
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
        if len(w_out) != len(w_in) + len(kernel) - 1:
            raise DSPFatal(
                f"Output waveform has length {len(w_out)}; expect {len(w_in) + len(kernel) - 1}"
            )
    elif chr(mode_in) == "v":
        mode = "valid"
        if len(w_out) != abs(len(w_in) - len(kernel)) + 1:
            raise DSPFatal(
                f"Output waveform has length {len(w_out)}; expect {abs(len(w_in) - len(kernel)) + 1}"
            )
    elif chr(mode_in) == "s":
        mode = "same"
        if len(w_out) != max(len(w_in), len(kernel)):
            raise DSPFatal(
                "Output waveform has length {len(w_out)}; expect {max(len(w_in), len(kernel))}"
            )
    else:
        raise DSPFatal("Invalid mode")

    w_out[:] = np.convolve(w_in, kernel, mode=mode)


@dspeed_guvectorize(
    "(n),(m),(),(p)",
    ["ffbf", "ddbd"],
    vectorized=True,
    copy_out=True,
)
def fft_convolve_wf(
    w_in: np.ndarray, kernel: np.array, mode_in: np.int8, w_out: np.ndarray
) -> None:  #
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
    nan_ids = np.isnan(w_in).any(axis=-1)
    w_in[nan_ids] = 0

    if np.isnan(kernel).any():
        return

    if kernel.shape[-1] > w_in.shape[-1]:
        raise DSPFatal("The filter is longer than the input waveform")

    if chr(mode_in) == "f":
        mode = "full"
    elif chr(mode_in) == "v":
        mode = "valid"
    elif chr(mode_in) == "s":
        mode = "same"
    else:
        raise DSPFatal("Invalid mode")

    if len(kernel.shape) < len(w_in.shape):
        kernel = kernel.reshape((1, *kernel.shape))
    w_out[:] = fftconvolve(w_in, kernel, mode=mode, axes=-1)
    w_out[nan_ids] = np.nan


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(m),(p)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def reflected_convolve_wf(
    w_in: np.ndarray, kernel: np.array, w_out: np.ndarray
) -> None:
    """
    Convolve a waveform with a kernel using reflection padding at the boundaries.

    This function extends the input waveform by reflecting its edges before
    convolution to minimize boundary artifacts. The reflection length is
    determined by the kernel size.

    Parameters
    ----------
    w_in : np.ndarray
        The input waveform to be convolved.
    kernel : np.ndarray
        The convolution kernel. Must be shorter than or equal to w_in.
    w_out : np.ndarray
        Output array for the filtered waveform. Will be filled with the
        convolution result, or NaN if inputs are invalid.

    Raises
    ------
    DSPFatal
        If the kernel length exceeds the input waveform length.

    Notes
    -----
    - If either w_in or kernel contains NaN values, w_out is set to NaN.
    - Uses 'reflect' mode padding to extend the signal at boundaries.
    - The extension length is (len(kernel) // 2) + 1 on each side.
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if np.isnan(kernel).any():
        return

    if len(kernel) > len(w_in):
        raise DSPFatal("The filter is longer than the input waveform")

    extension_length = int(len(kernel) / 2) + 1

    # Extend the signal
    extended_signal = np.pad(w_in, extension_length, mode="reflect")

    w_out[:] = np.convolve(extended_signal, kernel, mode="same")[
        extension_length:-extension_length
    ]


@numba.njit(**nb_kwargs_util)
def _convolve_wf_core(w_in, kernel, mode_in, w_out) -> None:
    """:func:`convolve_wf` without np.convolve, for callers that cannot allocate or call BLAS
    (CUDA device code). It follows numba's np.convolve (correlate with the reversed kernel:
    left overlap, full overlap, right overlap) with a sequential float64 inner product, rounded
    to the output dtype; np.convolve uses BLAS dot for floats (output-dtype accumulation in a
    different order), so results can differ in the last bits. Modes as character codes: f=102,
    v=118, s=115."""
    for i in range(len(w_out)):
        w_out[i] = np.nan
    if contains_nan(w_in) or contains_nan(kernel):
        return
    n1 = len(w_in)
    n = len(kernel)
    if n > n1:
        raise DSPFatal("The filter is longer than the input waveform")
    if mode_in == 102:
        n_left = n - 1
        n_right = n - 1
    elif mode_in == 118:
        n_left = 0
        n_right = 0
    elif mode_in == 115:
        n_left = n // 2
        n_right = n - n_left - 1
    else:
        raise DSPFatal("Invalid mode")
    idx = 0
    for i in range(n_left):                 # innerprod(w_in[:k], kernel[::-1][-k:])
        k = i + n - n_left
        acc = 0.0
        for t in range(k):
            acc = acc + w_in[t] * kernel[k - 1 - t]
        w_out[idx] = acc
        idx += 1
    for i in range(n1 - n + 1):             # innerprod(w_in[i:i+n], kernel[::-1])
        acc = 0.0
        for t in range(n):
            acc = acc + w_in[i + t] * kernel[n - 1 - t]
        w_out[idx] = acc
        idx += 1
    for i in range(n_right):                # innerprod(w_in[-k:], kernel[::-1][:k])
        k = n - i - 1
        acc = 0.0
        for t in range(k):
            acc = acc + w_in[n1 - k + t] * kernel[n - 1 - t]
        w_out[idx] = acc
        idx += 1
