from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs

@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:],float32[:],float32)",
        "void(float64[:], float64[:], float64[:], float64[:],float64[:],float64)",
    ],
    "(n),(m),(),(),(),()",
    **nb_kwargs,
)
def histogram_stats(
    weights_in: np.ndarray,
    edges_in: np.ndarray,
    mode_out: int,
    max_out: float,
    fwhm_out: float,
    max_in: float,
) -> None:
    """Compute useful histogram-related quantities.

    Parameters
    ----------
    weights_in
        histogram weights.
    edges_in
        histogram bin edges.
    max_in
        if not :any:`numpy.nan`, the mode is derived as the bin edge
        closest to `max_in`.  Otherwise the mode is computed automatically.
    mode_out
        The index of the mode.
    max_out
        the computed mode of the histogram. If `max_in` is not :any:`numpy.nan`
        then the closest edge to `max_in` is returned.
    fwhm_out
        is actually the HALF width at half maximum (HWHM) of the histogram.
        The calculations starts from the mode and descends left and right, taking
        the smallest HWHM found in either direction.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "hwhm, idx_out, mode_out": {
            "function": "histogram_stats",
            "module": "dspeed.processors.histogram",
            "args": ["hist_weights","hist_borders","idx_out","mode_out","hwhm","np.nan"],
            "unit": ["ADC", "none", "ADC"]
        }

    See Also
    --------
    .histogram
    """

    fwhm_out[0] = np.nan
    mode_out[0] = np.nan
    max_out[0] = np.nan

    if np.isnan(weights_in).any():
        return

    if len(weights_in) + 1 != len(edges_in):
        raise DSPFatal("length edges_in must be exactly 1 + length of weights_in")

    # find global maximum search from left to right
    max_index = 0
    if np.isnan(max_in):
        for i in range(0, len(weights_in), 1):
            if weights_in[i] > weights_in[max_index]:
                max_index = i

    # is user specifies mean justfind mean index
    else:
        if max_in > edges_in[-2]:
            max_index = len(weights_in) - 1
        else:
            for i in range(0, len(weights_in), 1):
                if abs(max_in - edges_in[i]) < abs(max_in - edges_in[max_index]):
                    max_index = i

    mode_out[0] = max_index
    # returns left bin edge
    max_out[0] = edges_in[max_index]

    # and the approx fwhm
    for i in range(max_index, len(weights_in), 1):
        if weights_in[i] <= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break

    # look also into the other direction
    for i in range(0, max_index, 1):
        if weights_in[i] >= 0.5 * weights_in[max_index] and weights_in[i] != 0:
            if fwhm_out[0] < abs(max_out[0] - edges_in[i]):
                fwhm_out[0] = abs(max_out[0] - edges_in[i])
            break