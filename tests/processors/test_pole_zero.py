import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import (
    convolve_damped_oscillator,
    convolve_exp,
    double_pole_zero,
    inject_damped_oscillation,
    pole_zero,
)


def test_pole_zero(compare_numba_vs_python):
    """
    Test that the pole-zero filter can correct an RC decay pulse into a step function
    """

    # Create a single exponential pulse to pole-zero correct
    tau = 30000
    ts = np.arange(0, 8192, dtype=np.float64)
    amplitude = 17500
    pulse_in = np.zeros(len(ts) + 20, dtype=np.float32)
    pulse_in[20:] = amplitude * np.exp(-ts / tau, dtype=np.float32)

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(pulse_in.size)
    w_in[4] = np.nan

    assert np.all(np.isnan(compare_numba_vs_python(pole_zero, w_in, tau)))

    # ensure that a valid input gives the expected output when comparing the pole-zero correction with the pole-zero processor
    step = np.full(len(ts), amplitude)
    w_out_expected = np.insert(step, 0, np.zeros(20))

    # Check that it works at float32 precision
    pulse_in = np.zeros(len(ts) + 20, dtype=np.float32)
    pulse_in[20:] = amplitude * np.exp(-ts / tau)
    result = compare_numba_vs_python(pole_zero, pulse_in, tau)
    assert result.dtype == np.float32
    assert np.allclose(result, w_out_expected, rtol=1e-06)

    # Check that it works at float64 precision
    pulse_in = np.zeros(len(ts) + 20, dtype=np.float64)
    pulse_in[20:] = amplitude * np.exp(-ts / tau)
    result = compare_numba_vs_python(pole_zero, pulse_in, tau)
    assert result.dtype == np.float64
    assert np.allclose(result, w_out_expected, rtol=1e-07)


def test_double_pole_zero(compare_numba_vs_python):
    """
    Test that the double pole-zero filter can correct the sum of two RC decays into a step function
    """

    # Create a double exponential pulse to double-pole-zero correct
    wf_len = 8192
    tp0 = 20
    amplitude = 17500
    tau1 = 1000
    tau2 = 30000
    frac = 0.98
    ts = np.arange(0, wf_len - tp0, dtype=np.float64)
    ys = amplitude * (1 - frac) * np.exp(-ts / tau1) + amplitude * frac * np.exp(
        -ts / tau2
    )

    # ensure the DSPFatal is raised if the waveform is too short
    pulse_out = np.zeros(2, dtype=np.float64)
    with pytest.raises(DSPFatal):
        double_pole_zero(np.ones(2), tau1, tau2, frac, pulse_out)

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(wf_len)
    w_in[4] = np.nan

    assert np.all(
        np.isnan(compare_numba_vs_python(double_pole_zero, w_in, tau1, tau2, frac))
    )

    # ensure that a valid input gives the expected output when comparing the pole-zero correction with the pole-zero processor
    w_out_expected = np.full(wf_len, amplitude)
    w_out_expected[:tp0] = 0

    pulse_in = np.zeros(wf_len, dtype=np.float64)
    pulse_in[tp0:] = ys
    result = compare_numba_vs_python(double_pole_zero, pulse_in, tau1, tau2, frac)
    assert result.dtype == np.float64
    assert np.allclose(result, w_out_expected, rtol=1e-7)

    # Make sure that the processor also works for float32 precision
    pulse_in = np.zeros(wf_len, dtype=np.float32)
    pulse_in[tp0:] = ys
    result = compare_numba_vs_python(double_pole_zero, pulse_in, tau1, tau2, frac)
    assert result.dtype == np.float32
    assert np.allclose(result, w_out_expected, rtol=1e-6)


def test_convolve_exp(compare_numba_vs_python):
    """
    Test that convolve_exp works with test convolutions with known results.
    """

    def test_with_delta(wf_len: int, t_offset: int, amp: float, tau: float):
        delta = np.zeros(wf_len, "float32")
        delta[t_offset] = amp
        out = compare_numba_vs_python(convolve_exp, delta, tau)

        if np.isnan(tau):
            assert np.isnan(out).all()
            return

        if tau != 0:
            exp = np.zeros_like(delta)
            exp[t_offset:] = amp * np.exp(-np.arange(wf_len - t_offset) / tau)
        else:
            exp = delta

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_delta(1000, 100, 10, 100)
    test_with_delta(1000, 500, 10, 100)
    test_with_delta(1000, 100, 10, 1)
    test_with_delta(1000, 100, 10, 10000)
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

        if np.isnan(tau) or np.isnan(omega) or np.isnan(phase):
            assert np.isnan(out).all()
            return

        if tau != 0:
            exp = np.zeros_like(delta)
            exp[t_offset:] = (
                amp
                * np.exp(-np.arange(wf_len - t_offset) / tau)
                * np.cos(np.arange(wf_len - t_offset) * omega + phase)
            )
        else:
            exp = delta

        assert np.max(np.abs(out - exp)) < 1.0e-6

    test_with_delta(1000, 100, 10, 100, 0.1, 0)
    test_with_delta(1000, 500, 10, 100, 0.1, 0)
    test_with_delta(1000, 100, 10, 1, 0.1, 0)
    test_with_delta(1000, 100, 10, 10000, 0.1, 0)
    test_with_delta(1000, 100, 10, 100, 0.9, 0)
    test_with_delta(1000, 100, 10, 100, 0.001, 0)
    test_with_delta(1000, 100, 10, 100, 0.1, np.pi / 4)
    test_with_delta(1000, 100, 10, 100, 0.1, np.pi / 2)

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

        if np.isnan(tau) or np.isnan(omega) or np.isnan(phase) or np.isnan(frac):
            assert np.isnan(out).all()
            return

        if tau != 0:
            exp = np.zeros_like(step)
            exp[t_offset:] = (
                frac
                * amp
                * np.exp(-np.arange(wf_len - t_offset) / tau)
                * np.cos(np.arange(wf_len - t_offset) * omega + phase)
            )
            exp += step
        else:
            exp = np.copy(step)
            exp[t_offset] += frac * amp * np.cos(phase)

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

    test(100, 10, 10, 0, 0.1, 0, 0.5)
    test(1000, 100, 10, np.nan, 0.1, 0, 0.5)
    test(100, 10, 10, 100, np.nan, 0, 0.5)
    test(1000, 100, 10, 100, 0.1, np.nan, 0.5)
    with pytest.raises(DSPFatal):
        test(100, 10, 10, 100, 0.1, 0, -0.1)
    with pytest.raises(DSPFatal):
        test(100, 10, 10, 100, 0.1, 0, 1.1)
