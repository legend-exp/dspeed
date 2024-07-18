import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import inject_damped_oscillation


def test_inject_damped_oscillation(compare_numba_vs_python):
    """
    Test that convolve_exp works with test convolutions with known results.
    """

    def test(
        wf_len: int,
        t_offset: int,
        amp: float,
        tau: float,
        omega: float,
        phase: float,
        frac: float,
    ):
        step = np.zeros(wf_len, "float64")
        step[t_offset:] = amp

        out = compare_numba_vs_python(
            inject_damped_oscillation, step, tau, omega, phase, frac
        )

        exp = np.zeros_like(step)
        exp[t_offset:] = (
            frac
            * amp
            * np.exp(-np.arange(wf_len - t_offset) / tau)
            * np.cos(np.arange(wf_len - t_offset) * omega + phase)
        )
        exp += step

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test(1000, 100, 10, 100, 0.1, 0, 1.0)
    test(1000, 100, 10, 100, 0.1, 0, 0.0)
    test(1000, 100, 10, 100, 0.1, 0, 0.5)
    test(1000, 500, 10, 100, 0.1, 0, 0.5)
    test(1000, 100, 10, 1, 0.1, 0, 0.5)
    test(1000, 100, 10, 10000, 0.1, 0, 0.5)
    test(1000, 100, 10, 100, 0.9, 0, 0.5)
    test(1000, 100, 10, 100, 0.001, 0, 0.5)
    test(1000, 100, 10, 100, 0.1, np.pi / 4, 0.5)
    test(1000, 100, 10, 100, 0.1, np.pi / 2, 0.5)

    with pytest.raises(DSPFatal):
        test(100, 10, 10, 0, 0.1, 0, 0.5)
        test(1000, 100, 10, np.nan, 0.1, 0, 0.5)
        test(100, 10, 10, 100, np.nan, 0, 0.5)
        test(1000, 100, 10, 100, 0.1, np.nan, 0.5)
        test(100, 10, 10, 100, 0.1, 0, -0.1)
        test(100, 10, 10, 100, 0.1, 0, 1.1)
        test(100, 10, 10, 100, 0.1, 0, np.nan)
