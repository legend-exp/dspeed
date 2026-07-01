"""Processors for getting values from arrays"""

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        f"void({t}[:], int64, {t}[:])"
        for t in [
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]
    ],
    "(n),()->()",
    **nb_kwargs,
    forceobj=True,
)
def get(v_in, i, a_out):
    """
    Get value at position ``i`` of array ``v_in``. If ``i`` is
    out of range return ``np.nan``.

    parameters
    ----------
    v_in
        input array
    i
        input index
    a_out
        output value
    """
    if i >= 0 and i < len(v_in):
        a_out[:] = v_in[i]
    else:
        a_out[:] = np.nan


@guvectorize(
    [
        f"void({t}[:], int64, {t}, {t}[:])"
        for t in [
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]
    ],
    "(n),(),()->()",
    **nb_kwargs,
    forceobj=True,
)
def get_default(v_in, i, default, a_out):
    """
    Get value at position ``i`` of array ``v_in``. If ``i`` is
    out of range or value is ``np.nan``, return ``default``

    parameters
    ----------
    v_in
        input array
    i
        input index
    default
        input value to return if value is not found
    a_out
        output value
    """
    if i >= 0 and i < len(v_in) and not np.isnan(v_in[i]):
        a_out[:] = v_in[i]
    else:
        a_out[:] = default
