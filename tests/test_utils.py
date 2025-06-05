import lgdo
import numpy as np

from dspeed.utils import numba_defaults


def test_numba_defaults_loading():
    numba_defaults.cache = False
    numba_defaults.boundscheck = True


def isclose(lhs, rhs, rtol=1e-5, atol=1e-8, equal_nan=True):
    # an is close comparison for LGDO structures

    if isinstance(lhs, lgdo.Struct) and isinstance(rhs, lgdo.Struct):
        if set(lhs) != set(rhs) or lhs.attrs != rhs.attrs:
            return False

        for k in lhs:
            if not isclose(lhs[k], rhs[k], rtol=rtol, atol=atol, equal_nan=equal_nan):
                return False
        return True

    elif isinstance(lhs, lgdo.Array) and isinstance(rhs, lgdo.Array):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return np.all(np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan))

    elif isinstance(lhs, lgdo.VectorOfVectors) and isinstance(
        rhs, lgdo.VectorOfVectors
    ):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return lhs.cumulative_length == rhs.cumulative_length and np.all(
            np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    return False
