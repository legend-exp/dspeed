"""
util functions for use by other processors. To call functions from inside
guvectorized functions, these will typically be jit and njit functions!
"""

import numba
import numpy as np
from ..utils import numba_defaults_kwargs as nb_kwargs

# filter out default kwargs that don't work in jit/njit
nb_kwargs_util = { k:v for k,v in nb_kwargs.items() if k not in {"target"} }

@numba.njit(**nb_kwargs_util)
def contains_nan(w: np.ndarray) -> bool:
    """Return whether any element of `w` is NaN."""
    # Process in fixed blocks to benefit from SIMD
    i = 0
    while i+16 <= len(w):
        found = False
        for j in range(i, i + 16):
            found |= np.isnan(w[j])
        if found:
            return True
        i+=16

    # Process final block
    for v in w[i:]:
        if np.isnan(v):
            return True

    return False
