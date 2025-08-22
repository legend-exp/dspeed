import os
import numpy as np
import pytest
from dspeed.processors.histogram_stats import histogram_peakstats, histogram_stats
from dspeed.errors import DSPFatal

from lgdo import lh5
from dspeed import build_dsp

def test_histogram_peakstats_basic():
    # Simple histogram: peak at bin 2, bins are [0,1,2,3,4]
    weights = np.array([0, 1, 3, 2, 0], dtype=np.float32)
    edges = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    mode_out = np.full(1, np.nan, dtype=np.float32)
    width_out = np.full(1, np.nan, dtype=np.float32)
    # Use FWHM (width_type=0), skip_zeroes=0, auto mode
    histogram_peakstats(weights, edges, np.nan, 0, 0, mode_out, width_out)
    # Mode should be at bin center 2.5
    assert np.isclose(mode_out[0], 2.5, atol=0.5)
    assert np.isclose(width_out[0], 2.0, atol=0.1)

    histogram_peakstats(weights, edges, np.nan, 0, 1, mode_out, width_out)
    assert np.isclose(width_out[0], 0.5, atol=0.1)
    histogram_peakstats(weights, edges, np.nan, 0, 2, mode_out, width_out)
    assert np.isclose(width_out[0], 1.5, atol=0.1)
    histogram_peakstats(weights, edges, np.nan, 0, 3, mode_out, width_out)
    assert np.isclose(width_out[0], 0.5, atol=0.1)
    histogram_peakstats(weights, edges, np.nan, 0, 4, mode_out, width_out)
    assert np.isclose(width_out[0], 1.5, atol=0.1)

def test_histogram_peakstats_user_max():
    # User supplies max_in near bin edge 3
    weights = np.array([0, 1, 3, 2, 0], dtype=np.float32)
    edges = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    mode_out = np.full(1, np.nan, dtype=np.float32)
    width_out = np.full(1, np.nan, dtype=np.float32)
    # max_in = 3.1, should select bin [3,4], center 3.5
    histogram_peakstats(weights, edges, 3.1, 0, 0, mode_out, width_out)
    assert np.isclose(mode_out[0], 3.5, atol=0.1)

def test_histogram_peakstats_wrong_size():
    weights = np.array([0, 1, 3, 1, 0], dtype=np.float32)
    # Wrong edges size: should be len(weights)+1, here it's too short
    edges = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    mode_out = np.full(1, np.nan, dtype=np.float32)
    width_out = np.full(1, np.nan, dtype=np.float32)
    with pytest.raises(DSPFatal) as excinfo:
        histogram_peakstats(weights, edges, np.nan, 0, 0, mode_out, width_out)
    assert "length edges_in must be exactly 1 + length of weights_in" in str(excinfo.value)

def test_histogram_peakstats_nan():
    weights = np.array([0, 1, np.nan, 1, 0], dtype=np.float32)
    # Wrong edges size: should be len(weights)+1, here it's too short
    edges = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    mode_out = np.full(1, np.nan, dtype=np.float32)
    width_out = np.full(1, np.nan, dtype=np.float32)
    with pytest.raises(DSPFatal) as excinfo:
        histogram_peakstats(weights, edges, np.nan, 0, 0, mode_out, width_out)
    assert "nan in input weights" in str(excinfo.value)

def test_histogram_peakstats_skipzeroes():
    # Simple histogram: peak at bin 2, bins are [0,1,2,3,4]
    weights = np.array([0, 1, 3, 0, 2, 1, 0], dtype=np.float32)
    edges = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    mode_out = np.full(1, np.nan, dtype=np.float32)
    width_out = np.full(1, np.nan, dtype=np.float32)
    # Use FWHM (width_type=0), skip_zeroes=0, auto mode
    histogram_peakstats(weights, edges, np.nan, 0, 0, mode_out, width_out)
    # Mode should be at bin center 2.5
    assert np.isclose(mode_out[0], 2.5, atol=0.5)
    assert np.isclose(width_out[0], 1.0, atol=0.1) # no zeroes skipped, peak ends early
    histogram_peakstats(weights, edges, np.nan, 1, 0, mode_out, width_out)
    assert np.isclose(width_out[0], 3.0, atol=0.1) # no zero btw 3 and 2 skipped


def test_histogram_peakstats_dsp(lgnd_test_data, tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp_peakstats.lh5"
    dsp_config = {
        "outputs": ["hist_weights", "hist_borders", "mode_out", "width_out"],
        "processors": {
            "hist_weights , hist_borders": {
                "function": "histogram_around_mode",
                "module": "dspeed.processors.histogram",
                "args": ["waveform", "np.nan", "1", "hist_weights(101)", "hist_borders(102)"],
                "unit": ["none", "ADC"],
            },
            "mode_out , width_out": {
                "function": "histogram_peakstats",
                "module": "dspeed.processors.histogram_stats",
                "args": ["hist_weights", "hist_borders", "np.nan", 0, 0, "mode_out", "width_out"],
                "unit": ["ADC", "ADC"],
            },
        },
    }
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=dsp_config,
        write_mode="r",
    )
    assert os.path.exists(dsp_file)

    df = lh5.read_as(
        "geds/dsp/", dsp_file, "pd", field_mask=["mode_out", "width_out"]
    )

    # Check that mode_out and width_out are finite and positive for the first event
    assert np.isfinite(df["mode_out"][0])
    assert np.isfinite(df["width_out"][0])
    assert df["width_out"][0] > 0

def test_histogram_stats_dsp(lgnd_test_data, tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp_histogram_stats.lh5"
    dsp_config = {
        "outputs": ["hist_weights", "hist_borders", "mode_out", "max_out", "fwhm_out"],
        "processors": {
            "hist_weights , hist_borders": {
                "function": "histogram_around_mode",
                "module": "dspeed.processors.histogram",
                "args": ["waveform", "np.nan", "1", "hist_weights(101)", "hist_borders(102)"],
                "unit": ["none", "ADC"],
            },
            "mode_out , max_out , fwhm_out": {
                "function": "histogram_stats",
                "module": "dspeed.processors.histogram_stats",
                "args": ["hist_weights", "hist_borders", "mode_out", "max_out", "fwhm_out", "np.nan"],
                "unit": ["none", "ADC", "ADC"],
            },
        },
    }
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=dsp_config,
        write_mode="r",
    )
    assert os.path.exists(dsp_file)

    df = lh5.read_as(
        "geds/dsp/", dsp_file, "pd", field_mask=["mode_out", "max_out", "fwhm_out"]
    )

    # Check that mode_out, max_out, and fwhm_out are finite for the first event
    assert np.isfinite(df["mode_out"][0])
    assert np.isfinite(df["max_out"][0])
    assert np.isfinite(df["fwhm_out"][0])