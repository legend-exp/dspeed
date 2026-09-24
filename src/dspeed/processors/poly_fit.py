"""Polynomial fitting processors for waveforms."""

from __future__ import annotations

import numba
import numpy as np
from numba import guvectorize

from ..utils import GUFuncWrapper
from ..utils import numba_defaults_kwargs as nb_kwargs
from .utils import contains_nan, nb_kwargs_util



@numba.njit(**nb_kwargs_util)
def _poly_moments(w_in: np.ndarray, arr: np.ndarray) -> None:
    """arr[j] = sum_i w_in[i] * i**j, for j < len(arr)."""
    for j in range(len(arr)):
        arr[j] = 0
    for i in range(0, len(w_in), 1):
        for j in range(len(arr)):
            arr[j] += w_in[i] * (i**j)


@numba.njit(**nb_kwargs_util)
def _poly_fitter_core(
    w_in: np.ndarray, inv: np.ndarray, poly_pars: np.ndarray, arr: np.ndarray
) -> None:
    """:func:`_poly_fitter` for callers that cannot allocate or call BLAS (e.g.
    CUDA device code): the float64 scratch ``arr`` (length ``len(poly_pars)``) is
    supplied by the caller and the matrix-vector product is written out. The
    product's summation order differs from BLAS, so results can differ from
    :func:`_poly_fitter` in the last bit."""
    if contains_nan(w_in):
        return
    _poly_moments(w_in, arr)
    for a in range(len(poly_pars)):
        acc = 0.0
        for b in range(len(poly_pars)):
            acc += inv[a, b] * arr[b]
        poly_pars[a] = acc


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
    if contains_nan(w_in):
        return

    arr = np.zeros(len(poly_pars), dtype="float")
    _poly_moments(w_in, arr)
    poly_pars[:] = inv @ arr



def poly_fit(length, deg):
    """Factory function for generating a polynomial fitter for an input of length
    `length` to a polynomial of order `deg`.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        fit_pars:
            function: dspeed.processors.poly_fit
            args: ["wf_logged", "fit_pars(shape=4)"]
            init_args: ["len(wf_logged)", 3]
    """

    vals_array = np.zeros(2 * deg + 1, dtype="float64")

    for i in range(length):
        # linear regression
        for j in range(2 * deg + 1):
            vals_array[j] += i**j

    mat = np.zeros((deg + 1, deg + 1), dtype="float")
    for i in range(deg + 1):
        mat[i, :] = vals_array[i : deg + 1 + i]

    inv = np.linalg.inv(mat)

    fitter = GUFuncWrapper(
        lambda w_in, poly_pars: _poly_fitter(w_in, inv, poly_pars),
        "(n),(m)",
        ["ff", "dd"],
        name="poly_fitter",
        vectorized=True,
        copy_out=False,
        doc_string=f"Fit w_in to order {deg} polynomial.",
    )
    # exposed for callers that run _poly_fitter_core themselves (e.g. on a GPU)
    fitter.inv = inv
    return fitter


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

    if contains_nan(w_in) or contains_nan(poly_pars):
        return

    mean[0] = rms[0] = 0
    isum = len(w_in)
    dt = mean.dtype.type

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = 0.0
        for j in range(len(poly_pars)):
            temp += poly_pars[j] * i**j
        temp = w_in[i] - temp
        # length-1 outputs as scalars, keeping the former array expressions' typing:
        # each update is evaluated in float64 and stored back in the output dtype
        mean[0] = dt(np.float64(mean[0]) + temp / (i + 1))
        rms[0] = dt(np.float64(rms[0]) + temp * temp)

    rms[0] = rms[0] / dt(isum - 1)
    rms[0] = np.sqrt(rms[0])


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

    if contains_nan(w_in) or contains_nan(poly_pars):
        return

    mean[0] = rms[0] = 0
    dt = mean.dtype.type

    for i in range(0, len(w_in), 1):
        # the mean and standard deviation
        temp = 0.0
        for j in range(len(poly_pars)):
            temp += poly_pars[j] * i**j
        mean[0] = dt(np.float64(mean[0]) + (w_in[i] - np.exp(temp)) / (i + 1))
        rms[0] = dt(np.float64(rms[0]) + (w_in[i] - np.exp(temp)) ** 2)

    rms[0] = rms[0] / dt(len(w_in) - 1)
    rms[0] = np.sqrt(rms[0])
