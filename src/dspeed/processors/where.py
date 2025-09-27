from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


# wrapper for numpy.where following gufunc protocol
@guvectorize(
    [
        f"void(b1, {t}, {t}, {t}[:])"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    "(),(),()->()",
    nopython=True,
    **nb_kwargs,
)
def where(condition, a, b, out):
    """Return elements chosen from x or y depending on condition.

    Parameters:
        condition:
            boolean array to select which input to use
        a:
            input to use if condition is ``True``
        b:
            input to use if condition is ``False``
        output:
            array containing output values

    YAML Configuration Example
    --------------------------

    Note: this processor is aliased using `where` in ProcessingChain and
    the `a if b else c` syntax. The following examples are equivalent.

    .. code-block:: yaml

        a_or_b:
          function: dspeed.processors.where
          args:
            - condition
            - a
            - b
            - a_or_b
        a_or_b: "where(condition, a, b)"
        a_or_b: "a if condition else b"
    """

    out[:] = np.where(condition, a, b)
