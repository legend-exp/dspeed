import numpy as np

from dspeed.processors import poly_fit


def test_poly_fit(compare_numba_vs_python):
    """Test polynomial fitter"""

    # cubic polynomial coefficients + values
    coeffs = np.array([5.0, 3.0, 1.0, -1.0])
    x = np.arange(10)
    y = sum(c * x**i for i, c in enumerate(coeffs))

    # generate processor
    filt = poly_fit(len(x), len(coeffs) - 1)

    coeffs_out = np.zeros_like(coeffs)
    compare_numba_vs_python(filt, y, coeffs_out)
    assert np.all(np.isclose(coeffs_out, coeffs))
