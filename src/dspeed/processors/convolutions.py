from __future__ import annotations

import numpy as np
from numba import guvectorize
from scipy.signal import fftconvolve

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
        "void(float32[:], float64, float32[:])",
        "void(float64[:], float64, float64[:])",
    ],
    "(n),()->(n)",
    **nb_kwargs,
)
def convolve_exp(w_in: np.ndarray, tau: float, w_out: np.ndarray) -> None:
    """Convolve waveform with exponential kernel.

    Notes
    -----
    kernel is normalized to have a maximum amplitude of 1. To normalize
    by area instead, divide result by tau.

    Parameters
    ----------
    w_in
        the input waveform
    tau
        decay time of exponential kernel
    w_out
        output waveform after convolution
    """
    w_out[:] = np.nan
    if np.isnan(w_in).any():
        return

    if tau == 0.0 or np.isnan(tau):
        raise DSPFatal("tau cannot be zero or NaN.")

    w_out[:] = 0
    c = np.exp(-1.0 / tau)
    cn = 1.0
    for i in range(len(w_in)):
        w_out[i:] += cn * w_in[: len(w_in) - i]
        cn *= c


@guvectorize(
    [
        "void(float32[:], float64, float64, float64, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),()->(n)",
    **nb_kwargs,
)
def convolve_damped_oscillator(
    w_in: np.ndarray, tau: float, omega: float, phase: float, w_out: np.ndarray
) -> None:
    """Convolve waveform with damped oscillator kernel.

    Notes
    -----
    kernel is normalized to have a maximum amplitude of 1. To normalize
    by area instead, divide result by tau.

    Parameters
    ----------
    w_in
        the input waveform
    tau
        decay time of exponential
    omega
        angular frequency of oscillation
    phase
        starting phase of oscillation
    w_out
        output waveform after convolution
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    if tau == 0.0 or np.isnan(tau):
        raise DSPFatal("tau cannot be zero or NaN.")
    if np.isnan(omega):
        raise DSPFatal("omega cannot be NaN.")
    if np.isnan(phase):
        raise DSPFatal("phase cannot be NaN.")

    w_out[:] = 0
    c = np.exp(-1.0 / tau + omega * 1j)
    cn = np.exp(phase * 1j)
    for i in range(len(w_in)):
        w_out[i:] += np.real(cn) * w_in[: len(w_in) - i]
        cn *= c
