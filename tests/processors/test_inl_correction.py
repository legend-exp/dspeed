import numpy as np

from dspeed.processors import inl_correction


def test_inl_correction(compare_numba_vs_python):
    """Testing function for the inl_correction processor."""

    len_wf = 8
    len_inl = 256

    # Case 1: w_in contains NaN
    w_in = np.ones(len_wf, dtype=np.float32)
    w_in[3] = np.nan
    inl = np.zeros(len_inl, dtype=np.float32)
    w_out = np.empty(len(w_in))
    compare_numba_vs_python(inl_correction, w_in, inl, w_out)
    assert np.all(np.isnan(w_out))

    # Case 2: inl contains NaN
    w_in = np.ones(len_wf, dtype=np.float32)
    inl = np.zeros(len_inl, dtype=np.float32)
    inl[1] = np.nan
    w_out = np.empty(len(w_in))
    compare_numba_vs_python(inl_correction, w_in, inl, w_out)
    assert np.all(np.isnan(w_out))

    # Case 3: normal correction (inl = +0.5 for all codes)
    w_in = np.arange(len_wf, dtype=np.float32)
    inl = np.full(len_inl, 0.5, dtype=np.float32)
    expected = w_in + 0.5
    w_out = np.empty(len(w_in))
    compare_numba_vs_python(inl_correction, w_in, inl, w_out)
    assert np.allclose(w_out, expected)

    # Case 4: ADC code out of range (e.g. w_in = [300] but inl = len 256)
    w_in = np.array([300], dtype=np.float32)
    inl = np.zeros(len_inl, dtype=np.float32)
    w_out = np.empty(len(w_in))
    error_raised = False
    try:
        compare_numba_vs_python(inl_correction, w_in, inl, w_out)
    except Exception:
        error_raised = True
    assert error_raised, "Expected error not raised for out-of-bounds ADC code"

    # Case 5: inl with variable correction
    w_in = np.array([0, 1, 2, 3], dtype=np.float32)
    inl = np.array([0.1, -0.1, 0.2, -0.2] + [0.0] * (len_inl - 4), dtype=np.float32)
    expected = w_in + inl[:4]
    w_out = np.empty(len(w_in))
    compare_numba_vs_python(inl_correction, w_in, inl, w_out)
    assert np.allclose(w_out, expected)
