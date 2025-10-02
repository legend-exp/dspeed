from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n)->(),(),(),()",
    **nb_kwargs,
)
def linear_slope_fit(
    w_in: np.ndarray, mean: float, stdev: float, slope: float, intercept: float
) -> None:
    """
    Calculate the mean and standard deviation of the waveform using
    Welford's method as well as the slope an intercept of the waveform
    using linear regression.

    Parameters
    ----------
    w_in
        the input waveform.
    mean
        the mean of the waveform.
    stdev
        the standard deviation of the waveform.
    slope
        the slope of the linear fit.
    intercept
        the intercept of the linear fit.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        bl_mean, bl_std, bl_slope, bl_intercept:
          function: linear_slope_fit
          module: dspeed.processors
          args:
            - "wf_blsub[0:round(44.5*us, wf_blsub.period)]"
            - bl_mean
            - bl_std
            - bl_slope
            - bl_intercept
          unit:
            - ADC
            - ADC
            - ADC
            - ADC
    """
    mean[0] = np.nan
    stdev[0] = np.nan
    slope[0] = np.nan
    intercept[0] = np.nan

    if np.isnan(w_in).any():
        return

    sum_x = sum_x2 = sum_xy = sum_y = mean[0] = stdev[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = w_in[i] - mean
        mean += temp / (i + 1)
        stdev += temp * (
            w_in[i] - mean
        )  # Welford's method is difference between updated mean and old mean
        # (x_i -mean_i) * (x_i - mean_i-1)

        # linear regression
        sum_x += i
        sum_x2 += i * i
        sum_xy += w_in[i] * i
        sum_y += w_in[i]

    stdev /= isum - 1
    np.sqrt(stdev, stdev)

    slope[0] = (isum * sum_xy - sum_x * sum_y) / (isum * sum_x2 - sum_x * sum_x)
    intercept[0] = (sum_y - sum_x * slope[0]) / isum


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(),()->(),()",
    **nb_kwargs,
)
def linear_slope_diff(
    w_in: np.ndarray, slope: float, intercept: float, mean: float, rms: float
) -> None:
    """
    Calculate the mean and rms of the waveform after subtracting out the
    provided slope and intercept.

    Uses Welford's method and linear regression.

    Parameters
    ----------
    w_in
        the input waveform.
    slope
        the slope of the linear fit.
    intercept
        the intercept of the linear fit.
    mean
        the mean of the waveform after subtracting the slope/intercept.
    stdev
        the standard deviation of the waveform after subtracting the slope/intercept.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        bl_slope_diff , bl_slope_rms:
          description: "finds mean and rms relative to linear fit of the baseline section"
          function: linear_slope_diff
          module: dspeed.processors
          args:
            - "wf_presum[0: round(44.5*us, wf_presum.period)]"
            - bl_slope
            - bl_intercept
            - bl_slope_diff
            - bl_slope_rms
          unit:
            - ADC
            - ADC
    """
    mean[0] = np.nan
    rms[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(slope) or np.isnan(intercept):
        return

    mean[0] = rms[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = w_in[i] - (slope * i + intercept)
        mean += temp / (i + 1)
        rms += temp * temp

    rms /= isum - 1
    np.sqrt(rms, rms)
