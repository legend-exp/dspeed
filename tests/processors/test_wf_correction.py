import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import wf_correction


def test_wf_correction(compare_numba_vs_python):
    """Testing function for the wf_correction processor."""

    len_wf = 8
    len_corr = 4

    # w_in contains NaN
    w_in = np.ones(len_wf, dtype=np.float32)
    w_corr = np.zeros(len_corr, dtype=np.float32)
    w_out = np.zeros(len_wf, dtype=np.float32)
    w_in[1] = np.nan
    result = compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 2, w_out)[-1]
    assert np.all(np.isnan(result))

    # w_corr contains NaN
    w_in = np.ones(len_wf, dtype=np.float32)
    w_corr[1] = np.nan
    result = compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 2, w_out)[-1]
    assert np.all(np.isnan(result))

    # tests on start_idx
    w_corr = np.zeros(len_corr, dtype=np.float32)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, -1, 2)[-1]
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, 9, 2)[-1]

    # tests on stop_idx
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, 0, -1)[-1]
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 9)[-1]

    # Case 9: start_idx is larger then stop_idx
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, 3, 2)[-1]

    # Case 10: stop_idx - start_idx is larger then w_corr length
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(wf_correction, w_in, w_corr, 0, 5)[-1]
