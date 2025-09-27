"""Conditional selection processor that wraps :func:`numpy.where`."""

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
    """Return elements chosen from ``a`` or ``b`` depending on ``condition``.

    Parameters
    ----------
    condition
        Boolean mask that selects which input to use.
    a
        Value to emit wherever ``condition`` is ``True``.
    b
        Value to emit wherever ``condition`` is ``False``.
    out
        Destination array that receives the selected values.

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
