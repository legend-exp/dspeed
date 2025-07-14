import numpy as np

from dspeed.processors import (
    ceil_to_nearest,
    floor_to_nearest,
    round_to_nearest,
    trunc_to_nearest,
)


def test_round_to_nearest(compare_numba_vs_python):
    w_in = np.array([-2.0, -1.5, -1.0, -0.75, -0.1, 0.0, 0.1, 0.75, 1.0, 1.5, 2.0])

    assert np.all(
        compare_numba_vs_python(round_to_nearest, w_in, 1.5)
        == np.array([-1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5])
    )

    assert np.all(
        compare_numba_vs_python(floor_to_nearest, w_in, 1.5)
        == np.array([-3.0, -1.5, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5])
    )

    assert np.all(
        compare_numba_vs_python(ceil_to_nearest, w_in, 1.5)
        == np.array([-1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 3.0])
    )

    assert np.all(
        compare_numba_vs_python(trunc_to_nearest, w_in, 1.5)
        == np.array([-1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5])
    )

    w_in = np.nan
    assert np.isnan(compare_numba_vs_python(round_to_nearest, w_in, 1.5))
    assert np.isnan(compare_numba_vs_python(floor_to_nearest, w_in, 1.5))
    assert np.isnan(compare_numba_vs_python(ceil_to_nearest, w_in, 1.5))
    assert np.isnan(compare_numba_vs_python(trunc_to_nearest, w_in, 1.5))
