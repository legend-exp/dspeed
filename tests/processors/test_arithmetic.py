import numpy as np
import pytest

from dspeed.processors import mean_below_threshold


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_below_threshold_basic(compare_numba_vs_python):
    """Test basic mean_below_threshold functionality."""
    # Values below 4.0: 1, 2, 3. Mean = (1 + 2 + 3) / 3 = 2
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 4.0)
    assert np.isclose(result, 2.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_below_threshold_all_above(compare_numba_vs_python):
    """Test mean_below_threshold when all values are above threshold."""
    # All values >= 10.0, should return NaN
    w_in = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 10.0)
    assert np.isnan(result)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_below_threshold_all_below(compare_numba_vs_python):
    """Test mean_below_threshold when all values are below threshold."""
    # All values < 100.0. Mean = 15 / 5 = 3
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 100.0)
    assert np.isclose(result, 3.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_below_threshold_with_nan_input(compare_numba_vs_python):
    """Test mean_below_threshold returns nan if input contains nan."""
    w_in = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 4.0)
    assert np.isnan(result)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_below_threshold_with_nan_threshold(compare_numba_vs_python):
    """Test mean_below_threshold returns nan if threshold is nan."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, np.nan)
    assert np.isnan(result)


def test_mean_below_threshold_negative_values(compare_numba_vs_python):
    """Test mean_below_threshold with negative values."""
    # Values below 0.0: -2, -1. Mean = (-2 + -1) / 2 = -1.5
    w_in = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 0.0)
    assert np.isclose(result, -1.5)


def test_mean_below_threshold_single_element_below(compare_numba_vs_python):
    """Test mean_below_threshold with single element below threshold."""
    w_in = np.array([42.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 50.0)
    assert np.isclose(result, 42.0)


def test_mean_below_threshold_single_element_above(compare_numba_vs_python):
    """Test mean_below_threshold with single element above threshold."""
    w_in = np.array([42.0])
    result = compare_numba_vs_python(mean_below_threshold, w_in, 30.0)
    assert np.isnan(result)
