"""Convolution-based waveform filter processors."""

from __future__ import annotations

import numpy as np
from numba import guvectorize
from scipy.fft import irfft, next_fast_len, rfft

from ..errors import DSPFatal
from ..utils import dspeed_guvectorize
from ..utils import numba_defaults_kwargs as nb_kwargs


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
    # keepdims so nan_ids stays an ndarray that broadcasts against both
    # (block, n) blocks and plain 1-D waveforms
    nan_ids = np.isnan(w_in).any(axis=-1, keepdims=True)
    if nan_ids.any():
        # zero out NaN'd waveforms in a copy: w_in is a view into the shared
        # processing-chain buffer, which downstream processors also read
        w_in = np.where(nan_ids, 0, w_in)

    if np.isnan(kernel).any():
        return

    if kernel.shape[-1] > w_in.shape[-1]:
        raise DSPFatal("The filter is longer than the input waveform")

    if chr(mode_in) not in ("f", "v", "s"):
        raise DSPFatal("Invalid mode")

    if len(kernel.shape) < len(w_in.shape):
        kernel = kernel.reshape((1, *kernel.shape))

    # equivalent to scipy.signal.fftconvolve(w_in, kernel, mode=mode, axes=-1)
    # with the same FFT length (so results are bit-identical), but skipping
    # scipy.signal's per-call python overhead, which dominates at block size
    n = w_in.shape[-1]
    m = kernel.shape[-1]
    full = n + m - 1
    fshape = next_fast_len(full, True)
    sp = rfft(w_in, fshape, axis=-1)
    sp *= rfft(kernel, fshape, axis=-1)
    ret = irfft(sp, fshape, axis=-1)

    if chr(mode_in) == "f":
        w_out[:] = ret[..., :full]
    elif chr(mode_in) == "s":
        start = (full - n) // 2
        w_out[:] = ret[..., start : start + n]
    else:
        out_len = n - m + 1
        start = (full - out_len) // 2
        w_out[:] = ret[..., start : start + out_len]
    w_out[...] = np.where(nan_ids, np.nan, w_out)


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
