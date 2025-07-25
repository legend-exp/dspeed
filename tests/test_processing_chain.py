import lgdo
import numpy as np
import pytest

from dspeed.errors import ProcessingChainError
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


def test_proc_chain_unit_conversion(spms_raw_tbl):
    dsp_config = {
        "outputs": ["a_unitless", "a_ns", "a_us", "a_ghz"],
        "processors": {
            "a_unitless": {
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": ["waveform", 100, "'n'", "a_unitless"],
            },
            "a_ns": {
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": ["waveform", "1600*ns", "'n'", "a_ns"],
            },
            "a_us": {
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": ["waveform", "1.6*us", "'n'", "a_us"],
            },
            "a_ghz": {  # note this doesn't really make sense, but I want to test if it will convert inverse units
                "function": "fixed_time_pickoff",
                "module": "dspeed.processors",
                "args": ["waveform", "6.25*GHz", "'n'", "a_ghz"],
            },
        },
    }
    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["a_unitless"][0] == lh5_out["a_ns"][0]
    assert lh5_out["a_unitless"][0] == lh5_out["a_us"][0]
    assert lh5_out["a_unitless"][0] == lh5_out["a_ghz"][0]


# Test that timing variables can be converted between multiple coordinate
# grids correctly. Also tests slicing with a stride. Pickoff a time from
# a windowed wf and a down-sampled waveform; they should be the same
def test_proc_chain_coordinate_grid(spms_raw_tbl):
    dsp_config = {
        "outputs": ["a_window", "a_downsample", "tp", "tp_window", "tp_downsample"],
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
            "tp": {
                "function": "time_point_thresh",
                "module": "dspeed.processors",
                "args": ["waveform", "a_window", "52.48*us+waveform.offset", 0, "tp"],
                "unit": "ns",
            },
            "tp_window": {
                "function": "time_point_thresh",
                "module": "dspeed.processors",
                "args": [
                    "waveform[2625:4025]",
                    "a_window",
                    "52.48*us+waveform.offset",
                    0,
                    "tp_window",
                ],
                "unit": "ns",
            },
            "tp_downsample": {
                "function": "time_point_thresh",
                "module": "dspeed.processors",
                "args": [
                    "waveform[0:8000:8]",
                    "a_window",
                    "52.48*us+waveform.offset",
                    0,
                    "tp_downsample",
                ],
                "unit": "ns",
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["a_window"][0] == lh5_out["a_downsample"][0]
    assert lh5_out["tp_window"][0] == lh5_out["tp"][0]
    assert -128 < lh5_out["tp_downsample"][0] - lh5_out["tp"][0] < 128


def test_proc_chain_round(spms_raw_tbl):
    dsp_config = {
        "outputs": ["w_round", "w_floor", "w_ceil", "w_trunc"],
        "processors": {
            "w_round": "round(waveform, 4)",
            "w_floor": "floor(waveform, 4)",
            "w_ceil": "ceil(waveform, 4)",
            "w_trunc": "trunc(waveform, 4)",
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    wf = spms_raw_tbl["waveform"].values[0]
    assert np.all(np.rint(wf / 4) * 4 == lh5_out["w_round"].values[0])
    assert np.all(np.floor(wf / 4) * 4 == lh5_out["w_floor"].values[0])
    assert np.all(np.ceil(wf / 4) * 4 == lh5_out["w_ceil"].values[0])
    assert np.all(np.trunc(wf / 4) * 4 == lh5_out["w_trunc"].values[0])


def test_proc_chain_where(spms_raw_tbl):
    # test with variable and const
    dsp_config = {
        "outputs": [
            "tp_min",
            "tp_max",
            "wf_min",
            "wf_max",
            "test1",
            "test2",
            "test3",
            "test4",
            "test5",
            "test6",
        ],
        "processors": {
            "tp_min, tp_max, wf_min, wf_max": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min", "tp_max", "wf_min", "wf_max"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            },
            "test1": "where(waveform<0, 0, waveform)",
            "test2": "where(waveform<0, waveform, 0)",
            "test3": "where(eventnumber==0, tp_min, 1*ns)",
            "test4": "where(eventnumber==0, tp_min, 1*us)",
            "test5": "where(eventnumber==0, 1*ns, tp_min)",
            "test6": "where(eventnumber==0, 1*us, tp_min)",
            "test7": "where(eventnumber==0, tp_min, wf_min)",
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 2)
    wf = spms_raw_tbl["waveform"].values[0]
    assert np.all(np.where(wf < 0, 0, wf) == lh5_out["test1"].values[0])
    assert np.all(np.where(wf < 0, wf, 0) == lh5_out["test2"].values[0])
    tp_min = lh5_out["tp_min"].nda
    assert lh5_out["test3"].attrs["units"] == "nanosecond"
    assert lh5_out["test3"].nda[0] == tp_min[0] and lh5_out["test3"].nda[1] == 1
    assert lh5_out["test4"].attrs["units"] == "nanosecond"
    assert lh5_out["test4"].nda[0] == tp_min[0] and lh5_out["test4"].nda[1] == 1000
    assert lh5_out["test5"].attrs["units"] == "nanosecond"
    assert lh5_out["test5"].nda[0] == 1 and lh5_out["test5"].nda[1] == tp_min[1]
    assert lh5_out["test6"].attrs["units"] == "nanosecond"
    assert lh5_out["test6"].nda[0] == 1000 and lh5_out["test6"].nda[1] == tp_min[1]
    with pytest.raises(ProcessingChainError):
        proc_chain, _, lh5_out = build_processing_chain(
            spms_raw_tbl, dsp_config, outputs=["test7"]
        )

    # test with variable and variable
    dsp_config = {
        "processors": {
            "w_downsample": "waveform[::2]",
            "w_win1": "waveform[:len(waveform)//2]",
            "w_win2": "waveform[len(waveform)//2:]",
            "tp_min, tp_max, wf_min, wf_max": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min", "tp_max", "wf_min", "wf_max"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            },
            "delta_t": "tp_max - tp_min",
            "test1": "where(eventnumber==0, w_downsample, w_win1)",  # fail due to different periods
            "test2": "where(eventnumber==0, w_win1, w_win2)",  # different offsets
            "test3": "where(eventnumber==0, tp_max, delta_t)",  # fail to different is_coord
            "test4": "where(eventnumber==0, tp_min, tp_max)",
        }
    }

    with pytest.raises(ProcessingChainError):
        proc_chain, _, lh5_out = build_processing_chain(
            spms_raw_tbl, dsp_config, outputs=["test1"]
        )

    proc_chain, _, lh5_out = build_processing_chain(
        spms_raw_tbl, dsp_config, outputs=["test2", "w_win1", "w_win2"]
    )
    proc_chain.execute(0, 2)
    assert (
        np.all(lh5_out["test2"].values[0] == lh5_out["w_win1"].values[0])
        and lh5_out["test2"].t0[0] == lh5_out["w_win1"].t0[0]
    )
    assert (
        np.all(lh5_out["test2"].values[1] == lh5_out["w_win2"].values[1])
        and lh5_out["test2"].t0[1] == lh5_out["w_win2"].t0[1]
    )

    with pytest.raises(ProcessingChainError):
        proc_chain, _, lh5_out = build_processing_chain(
            spms_raw_tbl, dsp_config, outputs=["test3"]
        )

    proc_chain, _, lh5_out = build_processing_chain(
        spms_raw_tbl, dsp_config, outputs=["test4", "tp_min", "tp_max"]
    )
    proc_chain.execute(0, 2)
    assert lh5_out["test4"].attrs["units"] == "nanosecond"
    assert np.all(lh5_out["test4"].nda[0] == lh5_out["tp_min"].nda[0])
    assert np.all(lh5_out["test4"].nda[1] == lh5_out["tp_max"].nda[1])

    # test with choosing between constant values
    dsp_config = {
        "processors": {
            "test1": "where(eventnumber==0, 10*ns, 1*us, dtype='f')",
            "test2": "where(eventnumber==0, 10*ns, 1000, dtype='f')",
            "test3": "where(eventnumber==0, 1000, 10*ns, dtype='f')",
            "test4": "where(eventnumber==0, 10, 1000, dtype='f')",
            "test5": "where(eventnumber==0, 10*ns, 10*m, dtype='f')",
        }
    }
    proc_chain, _, lh5_out = build_processing_chain(
        spms_raw_tbl, dsp_config, outputs=["test1", "test2", "test3", "test4"]
    )
    proc_chain.execute(0, 2)
    assert lh5_out["test1"].attrs["units"] == "nanosecond"
    assert lh5_out["test1"].nda[0] == 10 and lh5_out["test1"].nda[1] == 1000
    assert lh5_out["test2"].attrs["units"] == "nanosecond"
    assert lh5_out["test2"].nda[0] == 10 and lh5_out["test2"].nda[1] == 1000
    assert lh5_out["test3"].attrs["units"] == "nanosecond"
    assert lh5_out["test3"].nda[0] == 1000 and lh5_out["test3"].nda[1] == 10
    assert lh5_out["test4"].nda[0] == 10 and lh5_out["test4"].nda[1] == 1000
    with pytest.raises(ProcessingChainError):
        proc_chain, _, lh5_out = build_processing_chain(
            spms_raw_tbl, dsp_config, outputs=["test5"]
        )

    # test a if b else c form
    dsp_config = {
        "outputs": ["test"],
        "processors": {
            "test": "0 if waveform<0 else waveform",
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    wf = spms_raw_tbl["waveform"].values[0]
    assert np.all(np.where(wf < 0, 0, wf) == lh5_out["test"].values[0])


def test_proc_chain_isnan():
    dsp_config = {
        "outputs": ["test_nan", "test_finite"],
        "processors": {
            "test_nan": "isnan(input)",
            "test_finite": "isfinite(input)",
        },
    }

    tb = lgdo.Table(
        {"input": lgdo.Array(np.array([1.0, 0.0, np.inf, -np.inf, np.nan]))}
    )
    proc_chain, _, lh5_out = build_processing_chain(tb, dsp_config)
    proc_chain.execute()
    assert np.all(
        np.array([False, False, False, False, True]) == lh5_out["test_nan"].nda
    )
    assert np.all(
        np.array([True, True, False, False, False]) == lh5_out["test_finite"].nda
    )


def test_proc_chain_as_type(spms_raw_tbl):
    dsp_config = {
        "outputs": ["waveform_32"],
        "processors": {"waveform_32": "astype(waveform, 'float32')"},
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert np.all(
        spms_raw_tbl["waveform"].values[0] == lh5_out["waveform_32"].values[0]
    )


def test_output_types(spms_raw_tbl):
    dsp_config = {
        "outputs": ["wf_out", "vov_max_out", "n_max_out", "aoa_out"],
        "processors": {
            "wf_out": "-waveform",
            "aoa_out": "n_max_out + [1, 3, 5, 7, 9]",
            "vov_max_out, vov_min_out, n_max_out, n_min_out": {
                "function": "get_multi_local_extrema",
                "module": "dspeed.processors.get_multi_local_extrema",
                "args": [
                    "waveform",
                    5,
                    0.1,
                    1,
                    10,
                    0,
                    "vov_max_out(20, vector_len=n_max_out)",
                    "vov_min_out(20, vector_len=n_min_out)",
                    "n_max_out",
                    "n_min_out",
                ],
                "unit": ["ns", "ns", "none", "none"],
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(spms_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert isinstance(lh5_out["n_max_out"], lgdo.Array)
    assert isinstance(lh5_out["wf_out"], lgdo.WaveformTable)
    assert isinstance(lh5_out["aoa_out"], lgdo.ArrayOfEqualSizedArrays)
    assert isinstance(lh5_out["vov_max_out"], lgdo.VectorOfVectors)


def test_output_attrs(geds_raw_tbl):
    dsp_config = {
        "outputs": ["wf_blsub"],
        "processors": {
            "wf_blsub": {
                "function": "bl_subtract",
                "module": "dspeed.processors",
                "args": ["waveform[0:100]", "baseline", "wf_blsub"],
                "unit": "ADC",
                "lh5_attrs": {"test_attr": "This is a test"},
            }
        },
    }
    proc_chain, _, lh5_out = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["wf_blsub"].attrs["test_attr"] == "This is a test"


def test_database_params(geds_raw_tbl):
    dsp_config = {
        "outputs": ["test"],
        "processors": {
            "dbabc": "waveform[0]*0",
            "redberry": "dbabc+1",
            "test": {
                "function": "db.a + dbabc + redberry + db.b*db.c",
                "defaults": {"db.a": 1, "db.b": 2, "db.c": 3},
            },
        },
    }

    proc_chain, _, lh5_out = build_processing_chain(geds_raw_tbl, dsp_config)
    proc_chain.execute(0, 1)
    assert lh5_out["test"][0] == 8

    proc_chain, _, lh5_out = build_processing_chain(
        geds_raw_tbl, dsp_config, db_dict={"a": 2, "c": 0}
    )
    proc_chain.execute(0, 1)
    assert lh5_out["test"][0] == 3
