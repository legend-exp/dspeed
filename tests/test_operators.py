import lgdo
import numpy as np

from dspeed.processing_chain import build_processing_chain


def test_waveform_slicing():
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
    proc_chain, _, tbl_out = build_processing_chain(tbl_in, dsp_config)
    proc_chain.execute(0, 1)

    assert list(tbl_out.keys()) == ["eq", "neq", "gt", "gte", "lt", "lte"]
    assert all([tbl_out[k].nda.dtype == np.dtype("bool") for k in tbl_out.keys()])
    assert all(tbl_out["eq"].nda[0] == (w_in == 5))
    assert all(tbl_out["neq"].nda[0] == (w_in != 5))
    assert all(tbl_out["gt"].nda[0] == (w_in > 5))
    assert all(tbl_out["gte"].nda[0] == (w_in >= 5))
    assert all(tbl_out["lt"].nda[0] == (w_in < 5))
    assert all(tbl_out["lte"].nda[0] == (w_in <= 5))
