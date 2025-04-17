import lgdo
import numpy as np

from dspeed.processing_chain import build_processing_chain


def test_comparators():
    dsp_config = {
        "outputs": ["eq", "neq", "gt", "gte", "lt", "lte"],
        "processors": {
            "eq": "w_in == 5",
            "neq": "w_in != 5",
            "gt": "w_in > 5",
            "gte": "w_in >= 5",
            "lt": "w_in < 5",
            "lte": "w_in <= 5",
        },
    }
    w_in = np.arange(10)
    tbl_in = lgdo.types.Table(
        {"w_in": lgdo.types.ArrayOfEqualSizedArrays(nda=w_in.reshape((1, 10)))}
    )
    proc_chain, _, tbl_out = build_processing_chain(dsp_config, tbl_in)
    proc_chain.execute(0, 1)

    assert list(tbl_out.keys()) == ["eq", "neq", "gt", "gte", "lt", "lte"]
    assert all([tbl_out[k].nda.dtype == np.dtype("bool") for k in tbl_out.keys()])
    assert all(tbl_out["eq"].nda[0] == (w_in == 5))
    assert all(tbl_out["neq"].nda[0] == (w_in != 5))
    assert all(tbl_out["gt"].nda[0] == (w_in > 5))
    assert all(tbl_out["gte"].nda[0] == (w_in >= 5))
    assert all(tbl_out["lt"].nda[0] == (w_in < 5))
    assert all(tbl_out["lte"].nda[0] == (w_in <= 5))


def test_waveform_slicing(geds_raw_tbl):
    dsp_config = {
        "outputs": ["waveform", "wf_sample", "wf_slice", "wf_slice_stride"],
        "processors": {
            "wf_sample": {"function": "waveform[50]"},
            "wf_slice": {"function": "waveform[50:100]"},
            "wf_slice_stride": {"function": "waveform[50:100:2]"},
        },
    }
    proc_chain, _, tbl_out = build_processing_chain(dsp_config, geds_raw_tbl)
    proc_chain.execute(0, 10)

    assert isinstance(tbl_out.waveform, lgdo.WaveformTable)
    assert isinstance(tbl_out.wf_sample, lgdo.Array)
    assert isinstance(tbl_out.wf_slice, lgdo.WaveformTable)
    assert isinstance(tbl_out.wf_slice_stride, lgdo.WaveformTable)

    assert np.all(tbl_out.waveform.values[:, 50] == tbl_out.wf_sample)
    assert np.all(tbl_out.waveform.values[:, 50:100] == tbl_out.wf_slice.values)
    assert np.all(
        tbl_out.waveform.t0.nda + 50 * tbl_out.waveform.dt.nda
        == tbl_out.wf_slice.t0.nda
    )
    assert np.all(tbl_out.waveform.dt.nda == tbl_out.wf_slice.dt.nda)
    assert np.all(
        tbl_out.waveform.values[:, 50:100:2] == tbl_out.wf_slice_stride.values
    )
    assert np.all(
        tbl_out.waveform.t0.nda + 50 * tbl_out.waveform.dt.nda
        == tbl_out.wf_slice_stride.t0.nda
    )
    assert np.all(tbl_out.waveform.dt.nda == tbl_out.wf_slice_stride.dt.nda / 2)
