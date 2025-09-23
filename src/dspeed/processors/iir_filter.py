from __future__ import annotations

from collections.abc import Collection
import numpy as np
from pint import Quantity
import scipy.signal as sg
from ..errors import DSPFatal
from ..processors import recursive_filter
from ..utils import GUFuncWrapper
from ..processing_chain import ProcChainVar

# Generate recursive filters using scipy.signal

def iir_filter(
    freq: Quantity | float | Collection,
    order: int,
    rp: float = None,
    rs: float = None,
    f_samp: Quantity | float | ProcChainVar = None,
    ftype: str = "butter",
    btype: str = "lowpass",
):
    """Generator function for an iir filter based on
    ''scipy.signal.iirfilter''.

    Parameters
    ----------
    freq
        critical frequency(s) of filter. If no f_samp 
        is provided, express as fraction of nyquist frequency.
        If bandpass or bandstop, provide array of 2 values
    order
        order of filter
    rp
        for Chebyshev and elliptical, maximum ripple in the passband (dB)
    rs
        for Chebyshev and elliptical, minimum attenuation in the stopband (dB)
    f_samp
        sampling frequency for input waveform or ProcessingChain
        variable representing waveform
    ftype
        design of filter: {'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'}
        (default to butter)
    btype 
        type of filter: {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
        (default to lowpass)

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_lp": {
            "function": "iir_filter",
            "module": "dspeed.processors",
            "args_in": ["15*MHz", 4, "wf", "ftype=butter", "btype=lowpass"],
            "args": ["wf", "wf_lp(unit=ADC)"],
        }
    """
    # convert units as needed, check inputs are valid
    if isinstance(f_samp, ProcChainVar):
        f_samp = 1/f_samp.period

    if btype in ('lowpass', 'highpass'):
        if isinstance(freq, Collection):
            raise DSPFatal(f"{btype} filter requires one freq value")
        if not f_samp is None:
            f_c = float(2*freq/f_samp)
        else:
            f_c = freq
        if not 0 <= f_c <= 1:
            raise DSPFatal("Critical frequency must be positive and < nyquist frequency")
    elif btype in ('bandpass', 'bandstop'):
        if not (isinstance(freq, Collection) and len(freq)==2):
            raise DSPFatal(f"{btype} filter requires two freq values")
        if not f_samp is None:
            f_c = [float(2*f/f_samp) for f in freq]
        else:
            f_c = freq
        if not all(0 <= f <= 1 for f in f_c):
            raise DSPFatal("Critical frequency must be positive and < nyquist frequency")
    else:
        raise DSPFatal("Invalid type of filter")

    # design filter and initial filter conditions
    a, b = sg.iirfilter(
        order,
        f_c,
        rp=rp,
        rs=rs,
        btype=btype,
        ftype=ftype
    )
    gain = sum(a)/sum(b)
    
    return GUFuncWrapper(
        lambda w_in, w_out: recursive_filter(
            w_in, a, b, w_in[..., 0], gain*w_in[..., 0], w_out
        ),
        signature="(n)->(n)",
        types=["ff->f","dd->d"],
        name=f"{ftype}({freq}, {order}, {btype})",
        vectorized=True,
        copy_out=False
    )


def notch_filter(
    freq: Quantity | float,
    bandwidth: Quantity | float,
    f_samp: Quantity | float | ProcChainVar = None
):
    """Generator function for a notch filter

    Parameters
    ----------
    freq
        frequency of notch filter. If no f_samp is provided,
        express as fraction of nyquist frequency
    bandwidth
        bandwidth of notch filter
    f_samp
        sampling frequency for input waveform or ProcessingChain
        variable representing waveform

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_notch": {
            "function": "notch_filter",
            "module": "dspeed.processors",
            "args_in": ["15*MHz", "1.5*MHz", "wf"],
            "args": ["wf", "wf_notch(unit=ADC)"],
        }
    """
    if isinstance(f_samp, ProcChainVar):
        f_samp = 1/f_samp.period
    if not f_samp is None:
        f_c = float(2*freq/f_samp)
    else:
        f_c = freq
    Q = float(freq/bandwidth)

    if not 0 <= f_c <= 1:
        raise DSPFatal("Critical frequency must be positive and < nyquist frequency")

    a, b = sg.iirnotch(f_c, Q)
    return GUFuncWrapper(
        lambda w_in, w_out: recursive_filter(
            w_in, a, b, w_in[..., 0], w_in[..., 0], w_out
        ),
        signature="(n)->(n)",
        types=["ff->f","dd->d"],
        name=f"notch(f_samp, bandwidth)",
        vectorized=True,
        copy_out=False
    )

def peak_filter(
    freq: Quantity | float,
    bandwidth: Quantity | float,
    f_samp: Quantity | float | ProcChainVar = None
):
    """Generator function for a peak filter

    Parameters
    ----------
    freq
        frequency of peak filter. If no f_samp is provided,
        express as fraction of nyquist frequency
    bandwidth
        bandwidth of peak filter
    f_samp
        sampling frequency for input waveform or ProcessingChain
        variable representing waveform

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wf_peak": {
            "function": "peak_filter",
            "module": "dspeed.processors",
            "args_in": ["15*MHz", "1.5*MHz", "wf"],
            "args": ["wf", "wf_peak(unit=ADC)"],
        }
    """
    if isinstance(f_samp, ProcChainVar):
        f_samp = 1/f_samp.period
    if not f_samp is None:
        f_c = float(2*freq/f_samp)
    else:
        f_c = freq
    Q = float(freq/bandwidth)

    if not 0 <= f_c <= 1:
        raise DSPFatal("Critical frequency must be positive and < nyquist frequency")

    a, b = sg.iirpeak(f_c, Q)
    return GUFuncWrapper(
        lambda w_in, w_out: recursive_filter(
            w_in, a, b, w_in[..., 0], 0, w_out
        ),
        signature="(n)->(n)",
        types=["ff->f","dd->d"],
        name=f"peak(f_samp, bandwidth)",
        vectorized=True,
        copy_out=False
    )
