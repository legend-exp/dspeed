from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(int32[:], float32[:], float32[:])",
        "void(int32[:], float64[:], float64[:])",
    ],
    "(n),(p)->(n)",
    **nb_kwargs,
)
def inl_correction(w_in: np.ndarray, inl: np.ndarray, w_out: np.ndarray) -> None:
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
    w_out
        corrected waveform.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

      "wf_corr": {
          "function": "inl_correction",
          "module": "dspeed.processors",
          "args": ["w_in", "inl", "w_out"],
          "unit": "ADC"
       }
    """

    w_out[:] = np.nan

    if np.isnan(inl).any():
        return

    for i in range(len(w_in)):
        adc_code = w_in[i]
        if 0 <= adc_code < len(inl):
            w_out[i] = w_in[i] + inl[adc_code]
        else:
            raise DSPFatal(
                f"ADC code {adc_code} out of range for INL array of length {len(inl)}"
            )
