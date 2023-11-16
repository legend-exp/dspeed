import numpy as np

from dspeed.processors import wf_alignment


def test_wf_alignment(compare_numba_vs_python):
    len_wf = 20

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, 10))

    # test for nan if nan is passed to shift
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, np.nan, 1, 10))

    # test for nan if nan is passed to centroid
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, np.nan, 10))

    # test for nan if nan is passed to size
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, np.nan))

    # test for nan if shift is negative
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, -1, 1, 10))

    # test for nan if shift is too large
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, len_wf, 1, 10))

    # test for nan if centroid is negative
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, -1, 10))

    # test for nan if centroid is too large
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, len_wf, 10))

    # test for nan if size is negative
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, -1))

    # test for nan if size is zero
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, 0))

    # test for nan if size is too large
    w_in = np.ones(len_wf)
    assert np.isnan(compare_numba_vs_python(wf_alignment, w_in, 1, 1, len_wf))
