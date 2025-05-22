import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs
from .fixed_time_pickoff import fixed_time_pickoff


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(m),(m)",
    **nb_kwargs,
    forceobj=True,
)
def multi_a_filter(w_in, vt_maxs_in, va_max_out):
    """Finds the maximums in a waveform and returns the amplitude of the wave
    at those points.

    Parameters
    ----------
    w_in
        the array of data within which amplitudes of extrema will be found.
    vt_maxs_in
        the array of max positions for each waveform.
    va_max_out
        an array (in-place filled) of the amplitudes of the maximums of the waveform.
    """

    # Initialize output parameters

    va_max_out[:] = np.nan

    # Check inputs
    if np.isnan(w_in).any():
        return

    if not len(vt_maxs_in) < len(w_in):
        raise DSPFatal(
            "The length of your return array must be smaller than the length of your waveform"
        )

    nan_mask = np.isnan(vt_maxs_in)
    if nan_mask.all() or len(vt_maxs_in) == 0:
        return
    first_nan = np.where(nan_mask)[0]
    if len(first_nan) == 0:
        first_nan = None
    else:
        first_nan = first_nan[0]
        if ~np.isnan(vt_maxs_in[first_nan:]).all():
            first_nan = None
    fixed_time_pickoff(w_in, vt_maxs_in[:first_nan], ord("i"), va_max_out[:first_nan])
