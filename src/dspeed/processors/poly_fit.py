"""Polynomial fitting processors for waveforms."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..utils import GUFuncWrapper
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float64[::1,::1], float32[:])",
        "void(float64[:], float64[::1,::1], float64[:])",
    ],
    "(n),(m,m)->(m)",
    **nb_kwargs,
)
def _poly_fitter(w_in: np.ndarray, inv: np.ndarray, poly_pars: np.ndarray) -> None:
    """Helper function that fits w_in to order `len(poly_pars)-1` polynomial,
    while providing necessary inverse matrix.
    """
    if np.isnan(w_in).any():
        return

    arr = np.zeros(len(poly_pars), dtype="float")
    for i in range(0, len(w_in), 1):
        for j in range(len(poly_pars)):
            arr[j] += w_in[i] * (i**j)

    poly_pars[:] = inv @ arr


def poly_fit(length, deg):
    """Factory function for generating a polynomial fitter for an input of length
    `length` to a polynomial of order `deg`."""

    vals_array = np.zeros(2 * deg + 1, dtype="float64")

    for i in range(length):
        # linear regression
        for j in range(2 * deg + 1):
            vals_array[j] += i**j

    mat = np.zeros((deg + 1, deg + 1), dtype="float")
    for i in range(deg + 1):
        mat[i, :] = vals_array[i : deg + 1 + i]

    inv = np.linalg.inv(mat)

    return GUFuncWrapper(
        lambda w_in, poly_pars: _poly_fitter(w_in, inv, poly_pars),
        "(n),(m)",
        ["ff", "dd"],
        name="poly_fitter",
        vectorized=True,
        copy_out=False,
        doc_string=f"Fit w_in to order {deg} polynomial.",
    )


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(m)->(),()",
    **nb_kwargs,
)
def poly_diff(
    w_in: np.ndarray,
    poly_pars: np.ndarray,
    mean: float,
    rms: float,
) -> None:
    """ """
    mean[0] = np.nan
    rms[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(poly_pars).any():
        return

    mean[0] = rms[0] = 0
    isum = len(w_in)

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = 0.0
        for j in range(len(poly_pars)):
            temp += poly_pars[j] * i**j
        temp = w_in[i] - temp
        mean += temp / (i + 1)
        rms += temp * temp

    rms /= isum - 1
    np.sqrt(rms, rms)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(m)->(),()",
    **nb_kwargs,
)
def poly_exp_rms(
    w_in: np.ndarray, poly_pars: np.ndarray, mean: float, rms: float
) -> None:
    """ """

    mean[0] = np.nan
    rms[0] = np.nan

    if np.isnan(w_in).any() or np.isnan(poly_pars).any():
        return

    mean[0] = rms[0] = 0

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = 0.0
        for j in range(len(poly_pars)):
            temp += poly_pars[j] * i**j
        mean += (w_in[i] - np.exp(temp)) / (i + 1)
        rms += (w_in[i] - np.exp(temp)) ** 2

    rms /= len(w_in) - 1
    np.sqrt(rms, rms)
