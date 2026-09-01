import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import get_wf_centroid


def test_get_wf_centroid(compare_numba_vs_python):
    len_wf = 20

    # test for nan if w_in has a nan
    w_in = np.ones(len_wf)
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(get_wf_centroid, w_in, 1))

    # test for nan if nan is passed to shift
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, np.nan)

    # test for nan if shift is negative
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, -1)

    # test for nan if shift is too large
    w_in = np.ones(len_wf)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get_wf_centroid, w_in, len_wf)


def test_get_wf_centroid_no_crossing(compare_numba_vs_python):
    # noise-like waveform whose first minimum comes after its first maximum:
    # the argmin:argmax window is empty, so no zero crossing exists and the
    # centroid is undefined. The old implementation indexed an empty
    # np.where result here (an out-of-bounds read with boundscheck off).
    w_in = np.zeros(20)
    w_in[15] = -1.0  # first (and only) minimum, after...
    w_in[5] = 1.0  # ...the maximum
    assert np.isnan(compare_numba_vs_python(get_wf_centroid, w_in, 1))

    # window non-empty but all samples negative -> no positive crossing
    w_in = np.full(20, -0.5)
    w_in[2] = -2.0
    w_in[18] = 2.0
    assert np.isnan(compare_numba_vs_python(get_wf_centroid, w_in, 1))
