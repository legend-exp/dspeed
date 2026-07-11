"""Processors for getting values from arrays"""

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        f"void({t}[:], int64, {t}[:])"
        for t in [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
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
def get(a_in, i, a_out):
    """Get value at position ``i`` of array ``a_in``. Negative indices
    will get position ``i`` before the end. If ``i`` is out of range,
    raise ``DSPFatal``.

    parameters
    ----------
    a_in
        input array
    i
        input index
    a_out
        output value
    """
    if i >= -len(a_in) and i < len(a_in):
        a_out[:] = a_in[i]
    else:
        raise DSPFatal("i is out of range")


@guvectorize(
    [
        f"void({t}[:], int64, {t}, {t}[:])"
        for t in [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
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
def get_default(a_in, i, default, a_out):
    """Get value at position ``i`` of array ``a_in``. Negative indices
    will get position ``i`` before the end. If ``i`` is out of range,
    or value is ``NaN``, return ``default``

    parameters
    ----------
    a_in
        input array
    i
        input index
    default
        input value to return if value is not found
    a_out
        output value
    """
    if i >= -len(a_in) and i < len(a_in) and not np.isnan(a_in[i]):
        a_out[:] = a_in[i]
    else:
        a_out[:] = default
