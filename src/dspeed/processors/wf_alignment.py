from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs
        

@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n), (), (), (), (m)",
    **nb_kwargs,
)
def wf_alignment(
    w_in: np.ndarray,
    centroid: int,
    shift: int,
    size: int,
    w_out: np.ndarray
) -> None:
    """Calculate waveform centroid.
    Parameters
    ----------
    w_in
        the input waveform.
    centroid
        centroid.
    shift
        shift.
    size
        size of output waveform.
    w_out
        aligned waveform.
    JSON Configuration Example
    --------------------------
    .. code-block :: json
        "wf_align": {
          "function": "wf_alignment",
          "module": "pygama.dsp.processors",
          "args": ["waveform", "centroid", "shift", "centroid", "size", "wf_align"],
          "unit": "ADC"
        }
    """
    w_out[:] = np.nan
    
    if (centroid >= size/2) and (centroid < len(w_in) - size/2):
        w_out[:] = w_in[int(centroid - size/2):int(centroid + size/2)]
    elif (centroid > size/2-shift) and (centroid < size/2):
        ss = int((size+1)/2 - centroid)
        w_out[:ss] = w_in[0]
        w_out[ss:] = w_in[:int(centroid + size/2)]
    else:
        w_out[:] = w_in[:size]
