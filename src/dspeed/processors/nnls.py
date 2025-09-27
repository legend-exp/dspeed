from __future__ import annotations

import numpy as np
from numba import guvectorize

from ..errors import DSPFatal
from ..utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:,::1], float32[::1], float32, float32, boolean, float32, float32[::1])",
        "void(float64[:,::1], float64[::1], float64, float32, boolean, float32, float64[::1])",
    ],
    "(m,n),(m),(),(),(),(),(n)",
    nopython=True,
    **nb_kwargs,
)
def optimize_nnls(
    a: np.ndarray,
    b: np.ndarray,
    maxiter: int,
    tol: float,
    allow_singularity: bool,
    min_value: float,
    x: np.ndarray,
) -> None:
    """Solve ``argmin_x || ax - b ||_2`` for ``x>=0``.
    Based on :func:`scipy.optimize.nnls` implementation. Which in turn is based on
    the algorithm in Bro, R. and De Jong, S. (1997), A fast non-negativity-constrained least squares algorithm. J. Chemometrics, 11: 393-401

    Parameters
    ----------
    a : (m, n) ndarray
        Coefficient matrix
    b : (m,) ndarray, float
        Right-hand side vector.
    maxiter: int
        Maximum number of iterations.
    tol: float
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(a.T @ (a x - b)`` entries. Increasing this
        value relaxes the solution constraints.
    allow_singularity: bool
        If matrix is not solvable (e.g. because of non full rank caused by
        float precision), no error is raised but all elements of the
        solution vector are set NaN
    x : ndarray
        Solution vector.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        nnls_solution:
          function: optimize_nnls
          module: dspeed.processors
          args:
            - db.coefficient_matrix
            - wf_blsub
            - 1000
            - 1e-6
            - True
            - nnls_solution
    """

    def numba_ix(arr: np.array, rows: np.array, cols: np.array) -> np.array:
        """
        Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
        from https://github.com/numba/numba/issues/5894#issuecomment-974701551
        :param arr: 2D array to be indexed
        :param rows: Row indices
        :param cols: Column indices
        :return: 2D array with the given rows and columns of the input array
        """
        one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
        for i, r in enumerate(rows):
            start = i * len(cols)
            one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

        arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
        slice_1d = np.take(arr_1d, one_d_index)
        return slice_1d.reshape((len(rows), len(cols)))

    def is_singular(matrix):
        """
        Returns True if matrix det = 0 i.e. matrix is singular.
        """
        det = np.linalg.det(matrix)
        return abs(det) < np.finfo(np.float64).eps

    m, n = a.shape

    if n != len(x):
        raise DSPFatal(
            "n dimension of coefficient axis doesn't match solution vector length."
        )
    if m != len(b):
        raise DSPFatal(
            "m dimension of coefficient axis doesn't match right-hand vector length."
        )

    ata = np.transpose(a) @ a
    atb = b @ a  # Result is 1D - let NumPy figure it out

    # Initialize vars
    x[:] = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    # Inactive constraint switches
    p = np.zeros(n, dtype=np.bool_)
    pidx = np.arange(0, len(p), 1, dtype=np.int32)

    # Projected residual
    w = atb.copy().astype(np.float64)  # x=0. Skip (-ata @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0
    while (not p.all()) and (w[~p] > tol).any():
        # Get the "most" active coeff index and move to inactive set
        k = np.argmax(w * (~p))
        p[k] = True

        # Iteration solution
        s[:] = 0.0
        mat = numba_ix(ata, pidx[p], pidx[p])

        # check if matrix has full rank before solving
        if is_singular(mat) and allow_singularity:
            x[:] = np.nan
            return None

        s[p] = np.linalg.solve(mat, atb[p])

        # Inner loop
        while (iter < maxiter) and (s[p].min() <= min_value):
            iter += 1
            inds = p * (s <= min_value)
            alpha = (x[inds] / (x[inds] - s[inds])).min()
            x *= 1 - alpha
            x += alpha * s
            p[x <= tol] = False

            mat = numba_ix(ata, pidx[p], pidx[p])
            if is_singular(mat) and allow_singularity:
                x[:] = np.nan
                return None

            s[p] = np.linalg.solve(mat, atb[p])
            s[~p] = 0

        x[:] = s[:]
        w[:] = atb - ata @ x

        if iter == maxiter:
            return None
