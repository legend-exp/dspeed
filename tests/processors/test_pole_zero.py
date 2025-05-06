import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import double_pole_zero, pole_zero


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
