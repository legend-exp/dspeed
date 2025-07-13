import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import inl_correction


def test_inl_correction(compare_numba_vs_python):
    """Testing function for the inl_correction processor."""

    len_wf = 8
    len_inl = 256

    # Case 1: inl contains NaN
    w_in = np.ones(len_wf, dtype=np.int32)
    inl = np.zeros(len_inl, dtype=np.float32)
    inl[1] = np.nan
    assert np.all(np.isnan(compare_numba_vs_python(inl_correction, w_in, inl)))

    # Case 2: normal correction (inl = +0.5 for all codes)
    w_in = np.arange(len_wf, dtype=np.int32)
    inl = np.full(len_inl, 0.5, dtype=np.float32)
    expected = w_in.astype(np.float32) + 0.5
    assert np.allclose(compare_numba_vs_python(inl_correction, w_in, inl), expected)

    # Case 3: ADC code out of range
    w_in = np.array([300], dtype=np.int32)
    inl = np.zeros(len_inl, dtype=np.float32)
    with pytest.raises(DSPFatal, match="ADC code 300 out of range"):
        compare_numba_vs_python(inl_correction, w_in, inl)

    # Case 4: inl with variable correction
    w_in = np.array([0, 1, 2, 3], dtype=np.int32)
    inl = np.array([0.1, -0.1, 0.2, -0.2] + [0.0] * (len_inl - 4), dtype=np.float32)
    expected = w_in.astype(np.float32) + inl[:4]
    assert np.allclose(compare_numba_vs_python(inl_correction, w_in, inl), expected)
