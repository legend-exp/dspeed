import inspect

import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import time_point_thresh, interpolated_time_point_thresh


def test_time_point_thresh(compare_numba_vs_python):
    """Testing function for the time_point_thresh processor."""

    # test for nan if w_in has a nan
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(time_point_thresh, w_in, 1, 11, 0,))

    # test for nan if nan is passed to a_threshold
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(time_point_thresh, w_in, np.nan, 11, 0,))

    # test for nan if nan is passed to t_start
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(time_point_thresh, w_in, 1, np.nan, 0,))
    
    # test for nan if nan is passed to walk_forward
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(time_point_thresh, w_in, 1, 11, np.nan,))

    # test for error if t_start non integer
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
        time_point_thresh(w_in, 1, 10.5, 0, np.array([0.]))

    # test for error if walk_forward non integer
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
        time_point_thresh(w_in, 1, 11, 0.5, np.array([0.]))
        
    # test for error if t_start out of range
    with pytest.raises(DSPFatal):
        w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
        time_point_thresh(w_in, 1, 12, 0, np.array([0.]))
        
    # test walk backward
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(time_point_thresh, w_in, 1, 11, 0) == 8.
    
    # test walk forward
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(time_point_thresh, w_in, 3, 0, 1) == 4.

def test_interpolated_time_point_thresh(compare_numba_vs_python):
    """Testing function for the interpolated_time_point_thresh processor."""

    # test for nan if w_in has a nan
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    w_in[4] = np.nan
    assert np.isnan(compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1., 11., 0, 105))

    # test for nan if nan is passed to a_threshold
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(interpolated_time_point_thresh, w_in, np.nan, 11., 0,105))

    # test for nan if nan is passed to t_start
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1., np.nan, 0,105))
    
    # test for nan if t_start out of range
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert np.isnan(compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1., 12, 0,105))
     
        
    # test walk backward mode 'i'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 105) == 7.
    
    # test walk forward mode 'i'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 105) == 4.
    
    # test walk backward mode 'f'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 102) == 8.
    
    # test walk forward mode 'f'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 102) == 5.
    
    # test walk backward mode 'f'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 99) == 7.
    
    # test walk forward mode 'f'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 99) == 4.
    
    # test walk backward mode 'n'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1, 11, 0, 110) == 7.5
    
    # test walk forward mode 'n'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3, 0, 1, 110) == 4.5
    
    # test walk backward mode 'l'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 1.5, 11, 0, 108) == 8.5
    
    # test walk forward mode 'l'
    w_in = np.concatenate([np.arange(-1,5,1), np.arange(-1,5,1)], dtype="float")
    assert compare_numba_vs_python(interpolated_time_point_thresh, w_in, 3.5, 0, 1, 108) == 4.5