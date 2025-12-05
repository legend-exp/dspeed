import numpy as np

from dspeed.processors import sort


def test_sort(compare_numba_vs_python):
    """Testing function for the sort processor."""

    # test basic sorting functionality
    w_in = np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])
    w_out_expected = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 9.0])
    assert np.allclose(compare_numba_vs_python(sort, w_in), w_out_expected)

    # test with negative values
    w_in = np.array([3.0, -1.0, 2.0, -5.0, 0.0])
    w_out_expected = np.array([-5.0, -1.0, 0.0, 2.0, 3.0])
    assert np.allclose(compare_numba_vs_python(sort, w_in), w_out_expected)

    # test with already sorted array
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w_out_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(compare_numba_vs_python(sort, w_in), w_out_expected)

    # test with reverse sorted array
    w_in = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    w_out_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(compare_numba_vs_python(sort, w_in), w_out_expected)

    # test that nan in w_in produces all nans in output
    w_in = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.all(np.isnan(compare_numba_vs_python(sort, w_in)))
