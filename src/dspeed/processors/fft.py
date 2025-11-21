"""Fast fourier transform processors"""

from __future__ import annotations

import numpy as np
from numba import guvectorize, vectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], complex64[:])", "void(float64[:], complex128[:])"],
    "(n),(m)",
    **nb_kwargs(forceobj=True),
)
def fft(w_in, dft_out):
    """Perform a discrete fourier transform from a real waveform to a complex
    fourier spectrum

    Parameters
    ----------
    w_in
        input waveform
    dft_out
        output discrete fourier transform

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        dft:
            function: dspeed.processors.fft
            args:
              - wf
              - dft(len(wf)//2+1, period=1/wf.period/len(wf)
    """
    if not len(w_in) // 2 + 1 == len(dft_out):
        raise DSPFatal(f"Size of fft must be len(w_in)//2+1 = {len(w_in)//2+1}")

    dft_out[:] = np.nan
    if np.isnan(w_in).any():
        return

    np.fft.rfft(w_in, out=dft_out)


@guvectorize(
    ["void(complex64[:], float32[:])", "void(complex128[:], float64[:])"],
    "(n),(m)",
    **nb_kwargs(forceobj=True),
)
def ifft(dft_in, w_out):
    """Perform an inverse discrete fourier transform from a complex discrete
    fourier spectrum to a real waveform to a complex

    Parameters
    ----------
    dft_in
        input discrete fourier transform
    w_out
        output waveform

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        waveform:
            function: dspeed.processors.ifft
            args:
              - dft
              - waveform((len(dft)-1)*2, period=2/dft.period/len(dft)
    """
    if not (len(dft_in) - 1) * 2 == len(w_out):
        raise DSPFatal(f"Size of wf must be (len(dft_in)-1)*2 = {(len(dft_in)-1)*2}")

    w_out[:] = np.nan
    if np.isnan(dft_in).any():
        return

    np.fft.irfft(dft_in, out=w_out)


@vectorize(["float64(complex128, uint32)", "float32(complex64, uint32)"], **nb_kwargs)
def abs2norm(x, norm):
    """Helper for psd"""
    return (x.real * x.real + x.imag * x.imag) / norm


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n),(m)",
    **nb_kwargs(forceobj=True),
)
def psd(w_in, psd_out):
    """Perform a discrete fourier transform from a real waveform and
    extract the power spectral density

    Parameters
    ----------
    w_in
        input waveform
    psd_out
        output discrete power spectrum

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        psd:
            function: dspeed.processors.psd
            args:
              - wf
              - psd(len(wf)//2+1, period=1/wf.period/len(wf)
    """
    if not len(w_in) // 2 + 1 == len(psd_out):
        raise DSPFatal(f"Size of psd must be len(w_in)//2+1 = {len(w_in)//2+1}")

    psd_out[:] = np.nan
    if np.isnan(w_in).any():
        return

    abs2norm(np.fft.rfft(w_in), len(w_in), out=psd_out)
