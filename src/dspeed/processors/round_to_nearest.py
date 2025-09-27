from __future__ import annotations

import numpy as np
from numba import vectorize

from ..utils import numba_defaults_kwargs as nb_kwargs


@vectorize(
    [
        f"{t}({t}, {t})"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def round_to_nearest(val: np.ndarray, to_nearest: int | float) -> None:
    """
    Round value to nearest multiple of to_nearest.

    Parameters
    ----------
    val
        value to be rounded
    to_nearest
        round to multiple of this
    out
        rounded value

    YAML Configuration Example
    --------------------------

    Note: this processor is aliased using `round` in ProcessingChain.
    The following two examples are equivalent.

    .. code-block:: yaml

      # Equivalent forms:
      t_rounded:
        function: round_to_nearest
        module: dspeed.processors
        args:
          - t_in
          - "1*us"
          - t_rounded
        unit:
          - ns

      t_rounded: "round(t_in, 1*us)"
    """

    if np.isnan(val):
        return np.nan
    else:
        return to_nearest * round(val / to_nearest)


@vectorize(
    [
        f"{t}({t}, {t})"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def floor_to_nearest(val: np.ndarray, to_nearest: int | float) -> None:
    """
    Return largest multiple of to_nearest that is <= val

    Parameters
    ----------
    val
        value to be floored
    to_nearest
        floor to multiple of this
    out
        floored value

    YAML Configuration Example
    --------------------------

    Note: this processor is aliased using `floor` in ProcessingChain.
    The following two examples are equivalent.

    .. code-block:: yaml

      # Equivalent forms:
      t_floor:
        function: floor_to_nearest
        module: dspeed.processors
        args:
          - t_in
          - "1*us"
          - t_floor
        unit:
          - ns

      t_floor: "floor(t_in, 1*us)"
    """

    if np.isnan(val):
        return np.nan
    else:
        return to_nearest * np.floor(val / to_nearest)


@vectorize(
    [
        f"{t}({t}, {t})"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def ceil_to_nearest(val: np.ndarray, to_nearest: int | float) -> None:
    """
    Return smallest multiple of to_nearest that is >= val.

    Parameters
    ----------
    val
        value to be ceiled
    to_nearest
        ceil to multiple of this
    out
        ceiled value

    YAML Configuration Example
    --------------------------

    Note: this processor is aliased using `ceil` in ProcessingChain.
    The following two examples are equivalent.

    .. code-block:: yaml

      # Equivalent forms:
      t_ceil:
        function: ceil_to_nearest
        module: dspeed.processors
        args:
          - t_in
          - "1*us"
          - t_ceil
        unit:
          - ns

      t_ceil: "ceil(t_in, 1*us)"
    """

    if np.isnan(val):
        return np.nan
    else:
        return to_nearest * np.ceil(val / to_nearest)


@vectorize(
    [
        f"{t}({t}, {t})"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def trunc_to_nearest(val: np.ndarray, to_nearest: int | float) -> None:
    """
    Return multiple of to_nearest that is closest to val, towards zero

    Parameters
    ----------
    val
        value to be truncated
    to_nearest
        truncate to multiple of this
    out
        truncated value

    YAML Configuration Example
    --------------------------

    Note: this processor is aliased using `ceil` in ProcessingChain.
    The following two examples are equivalent.

    .. code-block:: yaml

      # Equivalent forms:
      t_trunc:
        function: trunc_to_nearest
        module: dspeed.processors
        args:
          - t_in
          - "1*us"
          - t_trunc
        unit:
          - ns

      t_trunc: "trunc(t_in, 1*us)"
    """

    if np.isnan(val):
        return np.nan
    else:
        return to_nearest * np.trunc(val / to_nearest)
