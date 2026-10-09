import numpy as np

from dspeed.processors.utils import contains_nan


def test_contains_nan(compare_numba_vs_python):
    # lengths around the 64-sample block boundary of the vectorized scan
    for dtype in (np.float32, np.float64):
        for n in (1, 63, 64, 65, 200):
            w = np.zeros(n, dtype=dtype)
            assert not compare_numba_vs_python(contains_nan, w)
            for pos in (0, n // 2, n - 1):
                w = np.zeros(n, dtype=dtype)
                w[pos] = np.nan
                assert compare_numba_vs_python(contains_nan, w)
    assert not compare_numba_vs_python(contains_nan, np.zeros(0, dtype=np.float64))
    assert not compare_numba_vs_python(contains_nan, np.array([np.inf, -np.inf]))
