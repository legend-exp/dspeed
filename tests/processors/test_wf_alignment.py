import numpy as np

from dspeed.processors import wf_alignment


def test_wf_alignment(compare_numba_vs_python):
    len_wf = 20
    size = 10

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    w_out = np.empty(size)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, size, w_out))

    # test if nan is passed to shift
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, np.nan, 1, size, w_out))

    # test if shift is negative
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, -1, 1, size, w_out))

    # test if shift is too large
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, len_wf, 1, size, w_out))

    # test if nan is passed to centroid
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, np.nan, size, w_out))

    # test if centroid is negative
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, -1, size, w_out))

    # test if centroid is too large
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, len_wf, size, w_out))

    # test for nan is passed to size
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, np.nan, w_out))

    # test if size is negative
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, -1, w_out))

    # test if size is zero
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, 0, w_out))

    # test for nan if size is too large
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, len_wf, w_out))
