import lgdo
import pytest
import numpy as np

from dspeed.processing_chain import build_processing_chain


def test_waveform_slicing(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_blsub"],
        "processors": {
            "wf_blsub": {
                "function": "bl_subtract",
                "module": "dspeed.processors",
                "args": ["waveform[0:100]", "baseline", "wf_blsub"],
                "unit": "ADC",
            },
        },
    }
    proc_chain, _, tbl_out = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    assert list(tbl_out.keys()) == ["wf_blsub"]
    assert isinstance(tbl_out["wf_blsub"], lgdo.WaveformTable)
    assert tbl_out["wf_blsub"].wf_len == 100


def test_processor_none_arg(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", 1, None, "wf_cum"],
                "kwargs": {"signature": "(n),(),()->(n)", "types": ["fii->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    dsp_config["processors"]["wf_cum"]["args"][2] = "None"
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_processor_kwarg_assignment(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", "axis=1", "out=wf_cum"],
                "kwargs": {"signature": "(n),()->(n)", "types": ["fi->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)

    dsp_config["processors"]["wf_cum"]["args"][1] = "dtypo=None"
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    with pytest.raises(TypeError):
        proc_chain.execute(0, 1)


def test_processor_dtype_arg(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_cum"],
        "processors": {
            "wf_cum": {
                "function": "cumsum",
                "module": "numpy",
                "args": ["waveform", "axis=0", "dtype='int32'", "out=wf_cum"],
                "kwargs": {"signature": "(n),(),()->(n)", "types": ["fiU->i"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_scipy_gauss_filter(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_gaus"],
        "processors": {
            "wf_gaus": {
                "function": "gaussian_filter1d",
                "module": "scipy.ndimage",
                "args": [
                    "waveform",
                    "0.1*us",
                    "mode='reflect'",
                    "truncate=3",
                    "output=wf_gaus",
                ],
                "kwargs": {"signature": "(n),(),(),()->(n)", "types": ["ffUf->f"]},
                "unit": "ADC",
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_histogram_processor_fixed_width(spms_raw_tbl):
    dsp_config = {
        "outputs": ["hist_weights", "hist_borders"],
        "processors": {
            "hist_weights , hist_borders": {
                "function": "histogram",
                "module": "dspeed.processors.histogram",
                "args": ["waveform", "hist_weights(100)", "hist_borders(101)"],
                "unit": ["none", "ADC"],
            }
        },
    }
    proc_chain, _, _ = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


def test_processor_variable_array_output(spms_raw_tbl):
    dsp_config = {
        "outputs": ["vt_max_out"],
        "processors": {
            "vt_max_out, vt_min_out, n_max_out, n_min_out": {
                "function": "get_multi_local_extrema",
                "module": "dspeed.processors",
                "args": [
                    "waveform",
                    5,
                    5,
                    0,
                    10,
                    0,
                    "vt_max_out(10)",
                    "vt_min_out(10)",
                    "n_max_out",
                    "n_min_out",
                ],
                "unit": "ADC",
            }
        },
    }

    proc_chain, _, _ = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)


# Test that timing variables can be converted between multiple coordinate
# grids correctly. Also tests slicing with a stride. Pickoff a time from
# a windowed wf and a down-sampled waveform; they should be the same
def test_proc_chain_coordinate_grid(spms_raw_tbl):
    dsp_config = {
        "outputs": ["a_window", "a_downsample"],
        "processors": {
            "a_window": {
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": [
                    "waveform[2625:4025]",
                    "51.2*us + waveform.offset",
                    "'i'",
                    "a_window",
                ],
                "unit": ["ADC"],
            },
            "a_downsample": {
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": [
                    "waveform[0:8000:8]",
                    "51.2*us + waveform.offset",
                    "'i'",
                    "a_downsample",
                ],
                "unit": ["ADC"],
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["a_window"][0] == lh5_out["a_downsample"][0]

def test_proc_chain_round(spms_raw_tbl):
    dsp_config = {
        "outputs": ["waveform_round"],
        "processors": {
            "waveform_round": "round(waveform, 4)"
        }
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert(np.all(np.rint(spms_raw_tbl["waveform"].values[0]/4)*4 == lh5_out["waveform_round"].values[0]))

def test_proc_chain_as_type(spms_raw_tbl):
    dsp_config = {
        "outputs": ["waveform_32"],
        "processors": {
            "waveform_32": "astype(waveform, 'float32')"
        }
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert(np.all(spms_raw_tbl["waveform"].values[0] == lh5_out["waveform_32"].values[0]))

