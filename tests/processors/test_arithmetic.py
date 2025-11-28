import numpy as np
import pytest

from dspeed.processors import mean, sum


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_sum_basic(compare_numba_vs_python):
    """Test basic sum functionality."""
    # Test sum of entire array
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(sum, w_in, np.nan, np.nan)
    assert np.isclose(result, 15.0)


def test_sum_with_range(compare_numba_vs_python):
    """Test sum with specified range."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Sum from index 1 to 3 (inclusive) = 2 + 3 + 4 = 9
    result = compare_numba_vs_python(sum, w_in, 1.0, 3.0)
    assert np.isclose(result, 9.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_sum_with_nan_input(compare_numba_vs_python):
    """Test sum returns nan if input contains nan."""
    w_in = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = compare_numba_vs_python(sum, w_in, np.nan, np.nan)
    assert np.isnan(result)


def test_sum_empty_range(compare_numba_vs_python):
    """Test sum returns nan when start > end."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(sum, w_in, 3.0, 2.0)
    assert np.isnan(result)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_basic(compare_numba_vs_python):
    """Test basic mean functionality."""
    # Mean of entire array: 15/(5-1) = 15/4 = 3.75
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(mean, w_in, np.nan, np.nan)
    assert np.isclose(result, 3.75)


def test_mean_with_range(compare_numba_vs_python):
    """Test mean with specified range."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Mean from index 1 to 3 (inclusive) = (2 + 3 + 4) / (3-1) = 9/2 = 4.5
    result = compare_numba_vs_python(mean, w_in, 1.0, 3.0)
    assert np.isclose(result, 4.5)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_mean_with_nan_input(compare_numba_vs_python):
    """Test mean returns nan if input contains nan."""
    w_in = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = compare_numba_vs_python(mean, w_in, np.nan, np.nan)
    assert np.isnan(result)


def test_mean_empty_range(compare_numba_vs_python):
    """Test mean returns nan when start > end."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compare_numba_vs_python(mean, w_in, 3.0, 2.0)
    assert np.isnan(result)


def test_sum_boundary_conditions(compare_numba_vs_python):
    """Test sum with boundary conditions."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Test with negative start (should be clamped to 0)
    result = compare_numba_vs_python(sum, w_in, -1.0, 2.0)
    # Sum of indices 0, 1, 2 = 1 + 2 + 3 = 6
    assert np.isclose(result, 6.0)

    # Test with end past array length (should be clamped to len-1)
    result = compare_numba_vs_python(sum, w_in, 2.0, 10.0)
    # Sum of indices 2, 3, 4 = 3 + 4 + 5 = 12
    assert np.isclose(result, 12.0)


def test_mean_boundary_conditions(compare_numba_vs_python):
    """Test mean with boundary conditions."""
    w_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Test with negative start (should be clamped to 0)
    result = compare_numba_vs_python(mean, w_in, -1.0, 2.0)
    # Mean of indices 0, 1, 2 = (1 + 2 + 3) / (2-0) = 6/2 = 3
    assert np.isclose(result, 3.0)

    # Test with end past array length (should be clamped to len-1)
    result = compare_numba_vs_python(mean, w_in, 2.0, 10.0)
    # Mean of indices 2, 3, 4 = (3 + 4 + 5) / (4-2) = 12/2 = 6
    assert np.isclose(result, 6.0)
