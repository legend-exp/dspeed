import numpy as np
import pytest
import scipy.signal as sg

from dspeed.errors import DSPFatal
from dspeed.processors import iir_filter, notch_filter, peak_filter
from dspeed.units import unit_registry as ureg


def test_iir_filter(compare_numba_vs_python):
    """
    Testing function for all types of iir_filter processor.
    Compare against scipy.signal
    """
    f_samp = 100 * ureg.MHz
    f_c1 = 0.1
    f_c2 = 0.75
    rp = 5
    rs = 60
    order = 4
    bl = 10
    w_in = bl + np.sin(np.pi * 0.5 * np.arange(200))
    w_out = np.zeros_like(w_in)

    for ftype in ["butter", "cheby1", "cheby2", "ellip", "bessel"]:
        for btype in ["lowpass", "highpass", "bandpass", "bandstop"]:
            if btype in ("lowpass", "highpass"):
                f_c = f_c1
                freq_c = f_c1 * f_samp / 2
            else:
                f_c = [f_c1, f_c2]
                freq_c = [f_c1 * f_samp / 2, f_c2 * f_samp / 2]
            # get expected output using scipy
            b, a = sg.iirfilter(order, f_c, rp, rs, btype=btype, ftype=ftype)
            zi = sg.lfilter_zi(b, a)
            exp_out = sg.lfilter(b, a, w_in, zi=zi * w_in[0])[0]

            filt = iir_filter(f_c, order, rp, rs, btype=btype, ftype=ftype)
            compare_numba_vs_python(filt, w_in, w_out)
            assert np.allclose(w_out, exp_out)

            # Same filter as above, construct with units
            filt = iir_filter(
                freq_c, order, rp, rs, f_samp=f_samp, btype=btype, ftype=ftype
            )
            compare_numba_vs_python(filt, w_in, w_out)
            assert np.allclose(w_out, exp_out)

            # Check for error
            if btype in ("lowpass", "highpass"):
                with pytest.raises(DSPFatal):
                    filt = iir_filter(
                        [f_c1, f_c2], order, rp, rs, btype=btype, ftype=ftype
                    )
                with pytest.raises(DSPFatal):
                    filt = iir_filter(1.1, order, rp, rs, btype=btype, ftype=ftype)
            else:
                with pytest.raises(DSPFatal):
                    filt = iir_filter(f_c1, order, rp, rs, btype=btype, ftype=ftype)
                with pytest.raises(DSPFatal):
                    filt = iir_filter(
                        [1.1, 1.2], order, rp, rs, btype=btype, ftype=ftype
                    )

    with pytest.raises(DSPFatal):
        filt = iir_filter([1.1, 1.2], order, btype="foo")


def test_notch_filter(compare_numba_vs_python):
    """
    Testing function for the notch_filter processor.
    Compare against scipy.signal
    """

    f_c = 0.5
    bw = 0.05
    bl = 10
    w_in = bl + np.sin(np.pi * f_c * np.arange(200))
    w_out = np.zeros_like(w_in)

    # get expected output using scipy
    b, a = sg.iirnotch(f_c, f_c / bw)
    zi = sg.lfilter_zi(b, a)
    exp_out = sg.lfilter(b, a, w_in, zi=zi * w_in[0])[0]

    filt = notch_filter(f_c, bw)
    compare_numba_vs_python(filt, w_in, w_out)
    assert np.allclose(w_out, exp_out)

    # Same filter as above, construct with units
    f_samp = 100 * ureg.MHz
    f_notch = 25 * ureg.MHz
    bandwidth = 2.5 * ureg.MHz
    filt = notch_filter(f_notch, bandwidth, f_samp)
    compare_numba_vs_python(filt, w_in, w_out)
    assert np.allclose(w_out, exp_out)

    # Check for error
    f_notch = 55 * ureg.MHz
    with pytest.raises(DSPFatal):
        filt = notch_filter(f_notch, bandwidth, f_samp)


def test_peak_filter(compare_numba_vs_python):
    """
    Testing function for the peak_filter processor.
    Compare against scipy.signal
    """

    f_c = 0.5
    bw = 0.05
    bl = 10
    w_in = bl + np.sin(np.pi * f_c * np.arange(200))
    w_out = np.zeros_like(w_in)

    # get expected output using scipy
    b, a = sg.iirpeak(f_c, f_c / bw)
    zi = sg.lfilter_zi(b, a)
    exp_out = sg.lfilter(b, a, w_in, zi=zi * w_in[0])[0]

    filt = peak_filter(f_c, bw)
    compare_numba_vs_python(filt, w_in, w_out)
    assert np.allclose(w_out, exp_out)

    # Same filter as above, construct with units
    f_samp = 100 * ureg.MHz
    f_peak = 25 * ureg.MHz
    bandwidth = 2.5 * ureg.MHz
    filt = peak_filter(f_peak, bandwidth, f_samp)
    compare_numba_vs_python(filt, w_in, w_out)
    assert np.allclose(w_out, exp_out)

    # Check for error
    f_peak = 55 * ureg.MHz
    with pytest.raises(DSPFatal):
        filt = peak_filter(f_peak, bandwidth, f_samp)
