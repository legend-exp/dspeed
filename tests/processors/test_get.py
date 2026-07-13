import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import get, get_default


def test_get(compare_numba_vs_python):
    v_in = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a_out = compare_numba_vs_python(get, v_in, np.array([0, 1, 2]))
    assert np.all(a_out == np.array([1, 5, 9]))

    with pytest.raises(DSPFatal):
        compare_numba_vs_python(get, v_in, np.array([1, 3, 5]))


def test_get_default(compare_numba_vs_python):
    v_in = np.array(
        [[1.0, 2.0, 3.0, np.nan], [4.0, 5.0, np.nan, np.nan], [6.0, 7.0, 8.0, 9.0]]
    )
    a_out = compare_numba_vs_python(get_default, v_in, np.array([0, 1, 2]), 0.0)
    assert np.all(a_out == np.array([1, 5, 8]))

    a_out_ob = compare_numba_vs_python(get_default, v_in, np.array([1, 3, 5]), 0.0)
    assert np.all(a_out_ob == np.array([2.0, 0.0, 0.0]))
