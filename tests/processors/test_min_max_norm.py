import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import min_max_norm


def test_min_max_norm(compare_numba_vs_python):
    """Testing function for the max_min_norm processor."""

    # set up values to use for each test case
    len_wf = 10

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    w_out = np.empty(len_wf)
    assert np.isnan(compare_numba_vs_python(min_max_norm, w_in, -1, 1, w_out))
    
    # test for division by 0 
    w_in = np.ones(len_wf)
    w_out = np.ones(len_wf)
    assert np.all(compare_numba_vs_python(min_max_norm, w_in, 0, 0, w_out))
    
    # test for abs(a_max) > abs(a_min)
    w_in = np.ones(len_wf)
    a_max = 2
    a_min = -1
    w_out = np.ones(len_wf)
    w_out_expected = np.ones(len_wf)/abs(a_max)
    assert np.allclose(
        compare_numba_vs_python(min_max_norm, w_in, a_min, a_max, w_out),
        w_out_expected,
    )

    # test for abs(a_max) < abs(a_min)
    w_in = np.ones(len_wf)
    a_max = 1
    a_min = -2
    w_out = np.ones(len_wf)
    w_out_expected = np.ones(len_wf)/abs(a_min)
    assert np.allclose(
        compare_numba_vs_python(min_max_norm, w_in, a_min, a_max, w_out),
        w_out_expected,
    )