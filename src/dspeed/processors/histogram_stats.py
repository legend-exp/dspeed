from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32, int32, int32, float32[:], float32[:])",
        "void(float64[:], float64[:], float64, int32, int32, float64[:], float64[:])",
    ],
    "(n),(m),(),(),()->(),()",
    **nb_kwargs,
)
def histogram_peakstats(
    weights_in: np.ndarray,
    edges_in: np.ndarray,
    max_in: float,
    skip_zeroes: int,
    width_type: int,
    mode_out: float,
    width_out: float,
) -> None:
    """
    Compute peak statistics for a histogram, including mode and width.
    The mode is determined either by the global maximum or by a user-specified value.
    The width is computed according to the selected width_type (FWHM or a HWHM option).
    Best to use the histogram from histogram_around_mode.

    Parameters
    ----------
    weights_in
        Histogram weights (bin contents).
    edges_in
        Histogram bin edges.
    max_in
        If not :any:`numpy.nan`, the mode is derived as the bin center closest to `max_in`.
        Otherwise, the mode is computed automatically.
    skip_zeroes
        If 1, zero bins are skipped when computing the width.
        If 0, zeroes are not skipped; not possible if aliasing is present; use histogram_around_mode
        instead of histogram then!
    width_type
        Determines the method for width calculation:
        0: FWHM
        1: Minimum of HWHM
        2: Maximum of HWHM
        3: left HWHM (towards lower ADC)
        4: right HWHM (towards higher ADC)
    mode_out
        Output: the computed mode value; always in the center of a bin.
    width_out
        Output: the computed width value.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        mode_out, width_out:
          function: histogram_peakstats
          module: dspeed.processors
          args:
            - hist_weights
            - hist_borders
            - np.nan
            - 0
            - 0
            - mode_out
            - width_out
          unit:
            - ADC
            - ADC
    See Also
    --------
    .histogram_around_mode
    """

    mode_out[0] = np.nan
    width_out[0] = np.nan

    if np.isnan(weights_in).any():
        raise DSPFatal("nan in input weights")

    n_bins = len(weights_in)
    if n_bins + 1 != len(edges_in):
        raise DSPFatal("length edges_in must be exactly 1 + length of weights_in")

    # find global maximum search
    max_index = 0
    if np.isnan(max_in):
        for i in range(0, n_bins):
            if weights_in[i] > weights_in[max_index]:
                max_index = i

    # is user specifies mean justfind mean index
    else:
        if max_in > edges_in[-1]:
            max_index = n_bins - 1
        elif max_in < edges_in[0]:
            max_index = 0
        else:
            for i in range(0, n_bins):
                if edges_in[i] <= max_in < edges_in[i + 1]:
                    max_index = i
                    break

    # bin center
    mode_out[0] = edges_in[max_index] + 0.5 * (
        edges_in[max_index + 1] - edges_in[max_index]
    )

    hwhm_left = np.nan
    hwhm_right = np.nan

    for i in range(max_index, n_bins):  # to the right
        if skip_zeroes and weights_in[i] == 0:
            continue
        if weights_in[i] <= 0.5 * weights_in[max_index]:
            hwhm_right = abs(
                mode_out[0] - edges_in[i]
            )  # take the left edge of the first bin below threshold
            break
    else:
        hwhm_right = abs(mode_out[0] - edges_in[-1])

    for i in range(max_index, -1, -1):  # to the left
        if skip_zeroes and weights_in[i] == 0:
            continue
        if weights_in[i] <= 0.5 * weights_in[max_index]:
            hwhm_left = abs(
                mode_out[0] - edges_in[i + 1]
            )  # take the right edge of the first bin below threshold
            break
    else:
        hwhm_left = abs(mode_out[0] - edges_in[0])

    if width_type == 0:  # FWHM
        width_out[0] = hwhm_left + hwhm_right
    elif width_type == 1:  # Minimum of HWHM
        width_out[0] = min(hwhm_left, hwhm_right)
    elif width_type == 2:  # Maximum of HWHM
        width_out[0] = max(hwhm_left, hwhm_right)
    elif width_type == 3:  # left HWHM (towards lower ADC)
        width_out[0] = hwhm_left
    elif width_type == 4:  # right HWHM (towards higher ADC )
        width_out[0] = hwhm_right
    else:
        raise DSPFatal(f"Unknown width_type {width_type}, must be [0...4]")


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
    Be careful, since outputs are biased, as the computed mode is
    aligned with the left edge of the bin.

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
        the largest HWHM found in either direction.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        hwhm, idx_out, mode_out:
          function: histogram_stats
          module: dspeed.processors
          args:
            - hist_weights
            - hist_borders
            - idx_out
            - mode_out
            - hwhm
            - np.nan
          unit:
            - ADC
            - none
            - ADC
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
