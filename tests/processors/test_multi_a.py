import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import multi_a_filter


def test_multi_a_filter(compare_numba_vs_python):
    """Testing function for the multi_a_filter."""

    len_wf = 20

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(
        compare_numba_vs_python(multi_a_filter, w_in, np.array([1, 2]))
    ).all()

    w_in = np.ones(len_wf)

    # test vt_maxs_in > wf len
    with pytest.raises(DSPFatal):
        maxs_in = np.arange(21)
        multi_a_filter(w_in, maxs_in, np.zeros(21))

    # test vt_max_len is 0
    assert len(compare_numba_vs_python(multi_a_filter, w_in, np.array([]))) == 0

    # test all nan
    assert np.isnan(
        compare_numba_vs_python(multi_a_filter, w_in, np.array([np.nan]))
    ).all()

    # test output
    assert compare_numba_vs_python(multi_a_filter, w_in, np.array([10]))[0] == 1

    # test output with remaining nans
    out = compare_numba_vs_python(multi_a_filter, w_in, np.array([10, np.nan, np.nan]))
    assert out[0] == 1
    assert np.isnan(out[1:]).all()

    # test output with 1 nan
    out = compare_numba_vs_python(multi_a_filter, w_in, np.array([10, np.nan, 12]))
    assert out[0] == 1 and out[2] == 1
    assert np.isnan(out[1])
