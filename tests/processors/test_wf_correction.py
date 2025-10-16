import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import wf_correction


def test_wf_correction(compare_numba_vs_python):
    """Testing function for the inl_correction processor."""

    len_wf = 8
    len_corr = 2

    # Case 1: w_in contains NaN
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    w_in[1] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 2)))

    # Case 2: w_corr contains NaN
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    w_corr[1] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 2)))

    # Case 3,4,5: start_idx is NaN, negative, larger then w_in length
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, np.nan, 2)))

    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, -1, 2)))

    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 9, 2)))

    # Case 6,7,8: stop_idx is NaN, negative, larger then w_in length
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, np.nan)))

    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, -1)))

    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 9)))

    # Case 9: start_idx is larger then stop_idx
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 3, 2)))

    # Case 10: stop_idx - start_idx is larger then w_corr length
    w_in = np.ones(len_wf, dtype=np.int32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    assert np.all(np.isnan(compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 4)))
