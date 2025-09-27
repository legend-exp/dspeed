from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(m),(p)",
    **nb_kwargs,
)
def histogram(
    w_in: np.ndarray, weights_out: np.ndarray, borders_out: np.ndarray
) -> None:
    """Produces and returns an histogram of the waveform.

    Parameters
    ----------
    w_in
        Data to be histogrammed.
    weights_out
        The output histogram weights.
    borders_out
        The output histogram bin edges of the histogram. Length must be len(weights_out)+1

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        hist_weights, hist_borders:
          function: histogram
          module: dspeed.processors.histogram
          args:
            - waveform
            - "hist_weights(100)"
            - "hist_borders(101)"
          unit:
            - none
            - ADC
    Note
    ----
    This implementation is significantly faster than just wrapping
    :func:`numpy.histogram`.

    See Also
    --------
    .histogram_stats
    """

    if len(weights_out) + 1 != len(borders_out):
        raise DSPFatal("length borders_out must be exactly 1 + length of weights_out")

    weights_out[:] = 0
    borders_out[:] = np.nan

    if np.isnan(w_in).any():
        return

    wf_min = min(w_in)
    wf_max = max(w_in)

    # create the bin borders
    borders_out[0] = wf_min
    delta = 0

    # number of bins
    bin_in = len(weights_out)

    # define the bin edges
    delta = (wf_max - wf_min) / (bin_in)
    for i in range(0, bin_in, 1):
        borders_out[i + 1] = wf_min + delta * (i + 1)

    # make the histogram
    for i in range(0, len(w_in), 1):
        for k in range(1, len(borders_out), 1):
            if (w_in[i] - borders_out[k]) < 0:
                weights_out[k - 1] += 1
                break


@guvectorize(
    [
        "void(float32[:], float32, float32, float32[:], float32[:])",
        "void(float64[:], float64, float64, float64[:], float64[:])",
    ],
    "(n),(),(),(m),(p)",
    **nb_kwargs,
)
def histogram_around_mode(
    w_in: np.ndarray,
    center: float,
    bin_width: float,
    weights_out: np.ndarray,
    borders_out: np.ndarray,
) -> None:
    """Creates a histogram centered at a specified point.
    If no point given, use the mode of the input data, resulting in two histogramming passes:
    1. using the full min-max range to determine the mode
    2. histogramming around the mode to determine the final output histogram.
    If a valid center is given, only step 2 is performed.
    The number of samples is determined from the size of the output arrays.
    The histogram will always be aligned such that the center is in the center of a bin,
    even if an even number of bins is requested.

    Parameters
    ----------
    w_in
        Data to be histogrammed.
    center
        center of the final output histogram; best a integer if histogramming ADC values.
        If NAN, use the mode instead
    bin_width
        the bin width of the output histogram (not used for determining the mode).
        Best to use integer values to avoid aliasing.
    weights_out
        The output histogram weights.
    borders_out
        The output histogram bin edges of the histogram. Length must be len(weights_out)+1

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        hist_weights, hist_borders:
          function: histogram
          module: dspeed.processors.histogram
          args:
            - waveform
            - np.nan
            - 1
            - "hist_weights(101)"
            - "hist_borders(102)"
          unit:
            - none
            - ADC
    See Also
    --------
    .histogram
    """

    if len(weights_out) + 1 != len(borders_out):
        raise DSPFatal("length borders_out must be exactly 1 + length of weights_out")

    # borders_out[:] = np.nan

    if np.isnan(w_in).any():
        raise DSPFatal("input data contains nan")

    # number of bins
    n_bins = len(weights_out)

    if np.isnan(center):
        weights_out[:] = 0

        # find the mode
        wf_min = min(w_in)
        wf_max = max(w_in)

        # create the bin borders
        borders_out[0] = wf_min
        delta = 0

        # define the bin edges
        delta = (wf_max - wf_min) / (n_bins)
        for i in range(0, n_bins):
            borders_out[i + 1] = wf_min + delta * (i + 1)

        # make the histogram
        for i in range(0, len(w_in)):
            for k in range(0, n_bins):
                if (w_in[i] - borders_out[k + 1]) < 0:
                    weights_out[k] += 1
                    break

        # find the mode
        center = borders_out[np.argmax(weights_out)] + 0.5 * delta
        # align center to bin_width
        center = np.round(center / bin_width) * bin_width

    # (re)set
    weights_out[:] = 0

    hist_min = center - bin_width * (n_bins // 2) - 0.5 * bin_width
    # hist_max = hist_min + bin_width * n_bins

    # create the bin borders
    for i in range(0, n_bins + 1):
        borders_out[i] = hist_min + bin_width * i
    # make the histogram
    for i in range(0, len(w_in)):
        # we might be below the histogram range, so we need to check
        if w_in[i] < borders_out[0]:
            continue
        for k in range(0, n_bins):
            if (w_in[i] - borders_out[k + 1]) < 0:
                weights_out[k] += 1
                break
