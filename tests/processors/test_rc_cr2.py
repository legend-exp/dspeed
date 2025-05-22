import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import rc_cr2


def test_rc_cr2(compare_numba_vs_python):
    # Create a single exponential pulse to RC-CR^2 filter
    zeta = 30000
    w_len = 8192
    ts = np.arange(0, w_len, dtype=np.float64)
    amplitude = 17500
    tau = 500

    # ensure that if there is a nan in w_in, all nans are outputted
    w_in = np.ones(w_len)
    w_in[4] = np.nan

    assert np.all(np.isnan(compare_numba_vs_python(rc_cr2, w_in, tau)))

    # Ensure that if the input waveform is less than 3 samples, an error is raised
    with pytest.raises(DSPFatal):
        rc_cr2(np.zeros(3), tau, np.zeros(3))

    # ensure that a valid input gives the expected output
    pulse_in = np.zeros(w_len, dtype=np.float64)
    pulse_in[w_len // 2 :] = amplitude * np.exp(-ts[: w_len // 2] / zeta)
    out_pulse = np.zeros_like(pulse_in)
    rc_cr2(pulse_in, tau, out_pulse)

    t = np.arange(8192 // 2 + 1, dtype=np.float64)
    # This is the exact form of an RC-CR^2 filter applied to an exponential pulse...
    w_out_expected = np.zeros_like(pulse_in)
    w_out_expected[8192 // 2 - 1 :] = (
        -zeta * t**2 * np.exp(-t / tau) / (2 * tau * (zeta - tau))
        + t * (zeta**2 - 2 * zeta * tau) * np.exp(-t / tau) / (zeta - tau) ** 2
        + zeta * tau**3 * np.exp(-t / zeta) / (zeta - tau) ** 3
        - zeta * tau**3 * np.exp(-t / tau) / (zeta - tau) ** 3
    )
    w_out_expected *= amplitude
    # rescale because we abandoned gain scaling in the filter
    w_out_expected *= np.amax(out_pulse) / np.amax(w_out_expected)

    result = compare_numba_vs_python(rc_cr2, pulse_in, tau)
    assert result.dtype == np.float64
    assert np.allclose(
        compare_numba_vs_python(rc_cr2, pulse_in, tau), w_out_expected, rtol=1e-01
    )

    # Check that it works at float32 precision
    pulse_in = np.zeros(w_len, dtype=np.float32)
    pulse_in[w_len // 2 :] = amplitude * np.exp(-ts[: w_len // 2] / zeta)
    result = compare_numba_vs_python(rc_cr2, pulse_in, tau)
    assert result.dtype == np.float32
    assert np.allclose(result, w_out_expected, rtol=1e-01)
