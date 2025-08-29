import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import fft, ifft, psd


def test_fft(compare_numba_vs_python):
    w_in = np.zeros(20, dtype="float64")

    w_in[::2] = 1
    w_in[1::2] = -1

    dft_out = np.zeros(10, dtype="complex128")
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(fft, w_in, dft_out)

    dft_out = np.zeros(11, dtype="complex128")
    compare_numba_vs_python(fft, w_in, dft_out)
    assert all(dft_out[:-1] == 0) and dft_out[-1] == len(w_in)

    w_in[1] = np.nan
    compare_numba_vs_python(fft, w_in, dft_out)
    assert np.all(np.isnan(dft_out))


def test_ifft(compare_numba_vs_python):
    dft_in = np.zeros(11, dtype="complex128")
    dft_in[0] = 1

    w_out = np.zeros(21, dtype="float64")
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(ifft, dft_in, w_out)

    w_out = np.zeros(20, dtype="float64")
    compare_numba_vs_python(ifft, dft_in, w_out)
    assert all(w_out[:-1] == 1 / len(w_out))

    dft_in[1] = np.nan
    compare_numba_vs_python(ifft, dft_in, w_out)
    assert np.all(np.isnan(w_out))


def test_psd(compare_numba_vs_python):
    w_in = np.zeros(20, dtype="float64")

    w_in[::2] = 1
    w_in[1::2] = -1

    psd_out = np.zeros(10, dtype="float64")
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(psd, w_in, psd_out)

    psd_out = np.zeros(11, dtype="float64")
    compare_numba_vs_python(psd, w_in, psd_out)
    assert all(psd_out[:-1] == 0) and psd_out[-1] == len(w_in)

    w_in[1] = np.nan
    compare_numba_vs_python(psd, w_in, psd_out)
    assert np.all(np.isnan(psd_out))
