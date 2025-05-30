from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32, float32[:])",
        "void(float64[:], float64[:], float64, float64[:])",
    ],
    "(n),(p),(),(n)",
    **nb_kwargs,
)
def inl_correction(
    w_in: np.ndarray, inl: np.ndarray, factor: float, w_out: np.ndarray
) -> None:
    """INL correction.

    Note
    ----
    This processor correct the input waveform by applying the INL.

    Parameters
    ----------
    w_in
        the input waveform.
    inl
        inl correction array.
    factor
        factor to apply inl correction.
    w_out
        corrected waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

      "wf_corr": {
          "function": "inl_correction",
          "module": "dspeed.processors",
          "args": ["w_in", "inl", "factor", "w_out"],
          "unit": "ADC"
       }
    """

    w_out[:] = np.nan

    if np.isnan(w_in).any() or np.isnan(inl).any():
        return

    if np.isnan(factor):
        raise DSPFatal("INL factor is nan")

    for i in range(len(w_in)):
        adc_code = int(w_in[i])
        if 0 <= adc_code < len(inl):
            w_out[i] = w_in[i] + factor * inl[adc_code]
        else:
            raise DSPFatal(
                f"ADC code {adc_code} out of range for INL array of length {len(inl)}"
            )
