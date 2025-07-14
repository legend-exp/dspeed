from __future__ import annotations

import numpy as np
from numba import vectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs

# Functions used for converting between different coordinate systems
# in ProcessingChain. Convert timing values knowing the offset
# in each coordinate frame, and the ratio of periods


@vectorize(
    [f"{t}({t}, f8, f8, f8)" for t in ["f4", "f8"]],
    **nb_kwargs,
)
def convert(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    return (buf_in + offset_in) * period_ratio - offset_out


@vectorize(
    [f"{t}({t}, f8, f8, f8)" for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8"]],
    **nb_kwargs,
)
def convert_int(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    tmp = (buf_in + offset_in) * period_ratio - offset_out
    ret = np.rint(tmp)
    if np.abs(tmp - ret) < 1.0e-5:
        return ret
    else:
        raise DSPFatal("Cannot convert to integer. Use round or astype")


@vectorize(
    [
        f"{t}({t}, f8, f8, f8)"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def convert_round(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    return np.rint((buf_in + offset_in) * period_ratio - offset_out)


@vectorize(
    [
        f"{t}({t}, f8, f8, f8)"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def convert_floor(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    return np.floor((buf_in + offset_in) * period_ratio - offset_out)


@vectorize(
    [
        f"{t}({t}, f8, f8, f8)"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def convert_ceil(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    return np.ceil((buf_in + offset_in) * period_ratio - offset_out)


@vectorize(
    [
        f"{t}({t}, f8, f8, f8)"
        for t in ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f4", "f8"]
    ],
    **nb_kwargs,
)
def convert_trunc(buf_in, offset_in, offset_out, period_ratio):  # noqa: N805
    return np.trunc((buf_in + offset_in) * period_ratio - offset_out)
