import numpy as np
import pytest

from dspeed import build_dsp
from dspeed.errors import DSPFatal
from dspeed.processors.histogram import histogram, histogram_around_mode


def test_histogram_fixed_width(compare_numba_vs_python):
    vals = np.arange(100) * 2 / 3
    with pytest.raises(DSPFatal):
        histogram(vals, np.zeros(10), np.zeros(10))

    hist_weights = np.zeros(66)
    hist_edges = np.zeros(67)
    histogram(vals, hist_weights, hist_edges)
    assert all(hist_edges == np.arange(67))
    assert all(hist_weights[0::2] == 2) and all(hist_weights[1::2] == 1)


def test_histogram_around_mode_basic(compare_numba_vs_python):
    # Create a simple waveform with a clear mode
    w_in = np.array([1, 2, 2, 2, 3, 4, 5], dtype=np.float32)
    n_bins = 11
    bin_width = 1.0
    weights_out = np.zeros(n_bins, dtype=np.float32)
    borders_out = np.zeros(n_bins + 1, dtype=np.float32)

    # apparently I need to do this in order for codecov to recognize testing
    def wrapped(w_in, center, bin_width):
        _, _, _, w_o, b_o = compare_numba_vs_python(
            histogram_around_mode, w_in, center, bin_width, weights_out, borders_out
        )
        return w_o, b_o

    # Center is nan, so mode will be computed
    weights_out, borders_out = wrapped(w_in, np.nan, bin_width)

    # The mode should be near 2, so the center bin should contain the most entries
    mode_bin = np.argmax(weights_out)
    assert weights_out[mode_bin] == 3
    # The center of the histogram should be aligned with the mode
    center = borders_out[mode_bin] + 0.5 * (
        borders_out[mode_bin + 1] - borders_out[mode_bin]
    )
    assert np.isclose(center, 2.0, atol=0.5)

    # Check that all entries are binned
    assert np.sum(weights_out) == len(w_in)

    w_in = np.array([1, 2, 2, 2, 3, 4, 5, 100], dtype=np.float32)
    weights_out, borders_out = wrapped(w_in, np.nan, bin_width)
    # histogram does not span the whole range, so not all entries are binned
    assert np.sum(weights_out) < len(w_in)

    w_in = np.array([1, 2, 2, 2, 3, 4, 5, -100], dtype=np.float32)
    weights_out, borders_out = wrapped(w_in, np.nan, bin_width)
    # same here
    assert np.sum(weights_out) < len(w_in)

    w_in = np.array([1, 2, 2, 2, 3, 4, 5, 100], dtype=np.float32)
    weights_out, borders_out = wrapped(w_in, 100, bin_width)
    assert np.sum(weights_out) == 1  # Only the last entry should be binned

    w_in[3] = np.nan
    with pytest.raises(DSPFatal) as excinfo:
        weights_out, borders_out = wrapped(w_in, np.nan, bin_width)
    assert "input data contains nan" in str(excinfo.value)

    borders_out = np.zeros(n_bins, dtype=np.float32)
    with pytest.raises(DSPFatal) as excinfo:
        bin_weights, borders_out = wrapped(w_in, np.nan, bin_width)
    assert "length borders_out must be exactly 1 + length of weights_out" in str(
        excinfo.value
    )


def test_histogram_around_mode_dsp(lgnd_test_data, tmptestdir):
    dsp_config = {
        "outputs": ["hist_weights", "hist_borders"],
        "processors": {
            "hist_weights , hist_borders": {
                "function": "histogram_around_mode",
                "module": "dspeed.processors.histogram",
                "args": [
                    "waveform",
                    "np.nan",
                    "1",
                    "hist_weights(101)",
                    "hist_borders(102)",
                ],
                "unit": ["none", "ADC"],
            }
        },
    }
    df = build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        dsp_config=dsp_config,
    )["geds/dsp"]

    # Check bin count and alignment
    assert len(df["hist_weights"][0]) + 1 == len(df["hist_borders"][0])
    # Check that the center bin is near the mode
    mode_bin = np.argmax(df["hist_weights"][0])
    center = df["hist_borders"][0][mode_bin] + 0.5 * (
        df["hist_borders"][0][mode_bin + 1] - df["hist_borders"][0][mode_bin]
    )
    # The mode should be within the waveform range
    assert df["hist_borders"][0][0] <= center <= df["hist_borders"][0][-1]
