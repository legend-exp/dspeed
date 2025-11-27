from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    **nb_kwargs,
)
def sort_wf(w_in: np.ndarray, w_out: np.ndarray) -> None:
    """Return a sorted array using :func:`numpy.sort`.

    Parameters
    ----------
    w_in
        the input waveform.
    w_out
        the output sorted waveform.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_sorted:
          function: sort_wf
          module: dspeed.processors
          args:
            - waveform
            - wf_sorted
    """
    w_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    w_out[:] = np.sort(w_in)
