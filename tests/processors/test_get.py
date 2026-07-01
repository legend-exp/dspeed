import numpy as np
import pytest

from dspeed import build_dsp
from dspeed.errors import DSPFatal
from dspeed.processors import get, get_default


def test_get(compare_numba_vs_python):
    v_in = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    a_out = compare_numba_vs_python(get, v_in, np.array([0, 1, 2]))
    assert np.all(a_out == np.array([1, 5, 9]))

    a_out_ob = compare_numba_vs_python(get, v_in, np.array([1, 3, 5]))
    assert a_out_ob[0] == 2 and np.all(np.isnan(a_out_ob[1:]))

def test_get_default(compare_numba_vs_python):
    v_in = np.array([
        [1., 2., 3., np.nan],
        [4., 5., np.nan, np.nan],
        [6., 7., 8., 9.]
    ])
    a_out = compare_numba_vs_python(get_default, v_in, np.array([0, 1, 2]), 0.0)
    assert np.all(a_out == np.array([1, 5, 8]))

    a_out_ob = compare_numba_vs_python(get_default, v_in, np.array([1, 3, 5]), 0.0)
    assert np.all(a_out_ob == np.array([2., 0., 0.]))