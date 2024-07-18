import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import convolve_damped_oscillator, convolve_exp


def test_convolve_exp(compare_numba_vs_python):
    """
    Test that convolve_exp works with test convolutions with known results.
    """

    def test_with_delta(wf_len: int, t_offset: int, amp: float, tau: float):
        delta = np.zeros(wf_len, "float32")
        delta[t_offset] = amp
        out = compare_numba_vs_python(convolve_exp, delta, tau)

        exp = np.zeros_like(delta)
        exp[t_offset:] = amp * np.exp(-np.arange(wf_len - t_offset) / tau)

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_delta(1000, 100, 10, 100)
    test_with_delta(1000, 500, 10, 100)
    test_with_delta(1000, 100, 10, 1)
    test_with_delta(1000, 100, 10, 10000)

    with pytest.raises(DSPFatal):
        test_with_delta(100, 10, 10, 0)
        test_with_delta(100, 10, 10, np.nan)

    def test_with_step(wf_len: int, t_offset: int, amp: float, tau: float):
        step = np.zeros(wf_len, "float64")
        step[t_offset + 1 :] = amp
        out = compare_numba_vs_python(convolve_exp, step, tau)

        exp = np.zeros_like(step)
        exp[t_offset:] = (
            amp
            * (1 - np.exp(-np.arange(wf_len - t_offset) / tau))
            / (1 - np.exp(-1 / tau))
        )

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_step(1000, 0, 10, 100)
    test_with_step(1000, 500, 10, 100)
    test_with_step(1000, 500, 10, 1)
    test_with_step(1000, 500, 10, 10000)


def test_convolve_damped_oscillator(compare_numba_vs_python):
    """
    Test that convolve_exp works with test convolutions with known results.
    """

    def test_with_delta(
        wf_len: int, t_offset: int, amp: float, tau: float, omega: float, phase: float
    ):
        delta = np.zeros(wf_len, "float32")
        delta[t_offset] = amp
        out = compare_numba_vs_python(
            convolve_damped_oscillator, delta, tau, omega, phase
        )

        exp = np.zeros_like(delta)
        exp[t_offset:] = (
            amp
            * np.exp(-np.arange(wf_len - t_offset) / tau)
            * np.cos(np.arange(wf_len - t_offset) * omega + phase)
        )

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_delta(1000, 100, 10, 100, 0.1, 0)
    test_with_delta(1000, 500, 10, 100, 0.1, 0)
    test_with_delta(1000, 100, 10, 1, 0.1, 0)
    test_with_delta(1000, 100, 10, 10000, 0.1, 0)
    test_with_delta(1000, 100, 10, 100, 0.9, 0)
    test_with_delta(1000, 100, 10, 100, 0.001, 0)
    test_with_delta(1000, 100, 10, 100, 0.1, np.pi / 4)
    test_with_delta(1000, 100, 10, 100, 0.1, np.pi / 2)

    with pytest.raises(DSPFatal):
        test_with_delta(100, 10, 10, 0, 0.1, 0)
        test_with_delta(1000, 100, 10, np.nan, 0.1, 0)
        test_with_delta(100, 10, 10, 100, np.nan, 0)
        test_with_delta(1000, 100, 10, 100, 0.1, np.nan)

    def test_with_step(
        wf_len: int, t_offset: int, amp: float, tau: float, omega: float, phase: float
    ):
        step = np.zeros(wf_len, "float64")
        step[t_offset + 1 :] = amp
        out = compare_numba_vs_python(
            convolve_damped_oscillator, step, tau, omega, phase
        )

        exp = np.zeros_like(step)
        exp[t_offset:] = np.real(
            amp
            * np.exp(phase * 1j)
            * (1 - np.exp(np.arange(wf_len - t_offset) * (-1 / tau + omega * 1j)))
            / (1 - np.exp(-1 / tau + omega * 1j))
        )

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_step(1000, 100, 10, 100, 0.1, 0)
    test_with_step(1000, 500, 10, 100, 0.1, 0)
    test_with_step(1000, 100, 10, 1, 0.1, 0)
    test_with_step(1000, 100, 10, 10000, 0.1, 0)
    test_with_step(1000, 100, 10, 100, 0.9, 0)
    test_with_step(1000, 100, 10, 100, 0.001, 0)
    test_with_step(1000, 100, 10, 100, 0.1, np.pi / 4)
    test_with_step(1000, 100, 10, 100, 0.1, np.pi / 2)
