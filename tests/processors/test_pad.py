import numpy as np
import pytest

from lgdo import VectorOfVectors, Table

from dspeed import build_dsp
from dspeed.errors import DSPFatal
from dspeed.processors import pad

def test_pad(compare_numba_vs_python):
    v_in = np.arange(100., dtype="float32")
    a_out = np.zeros(1000, dtype="float32")

    compare_numba_vs_python(pad, v_in, 100, 500, 0, 100, a_out)
    assert np.all(a_out[:500] == 0)
    assert np.all(a_out[500:600] == v_in)
    assert np.all(a_out[600:] == 100)

    compare_numba_vs_python(pad, v_in, 50, 500, 0, 50, a_out)
    assert np.all(a_out[:500] == 0)
    assert np.all(a_out[500:550] == v_in[:50])
    assert np.all(a_out[550:] == 50)

    with pytest.raises(DSPFatal):
        compare_numba_vs_python(pad, v_in, 150, 500.5, 0, 100, a_out)

    compare_numba_vs_python(pad, v_in, 100, np.nan, 0, 100, a_out)
    assert np.isnan(a_out).all()

    with pytest.raises(DSPFatal):
        compare_numba_vs_python(pad, v_in, 100, 500.5, 0, 100, a_out)

    v_in[10] = np.nan
    compare_numba_vs_python(pad, v_in, 100, 500, 0, 100, a_out)
    assert np.isnan(a_out).all()

    tb_in = Table({
        "vov_in": VectorOfVectors(flattened_data=np.arange(100, dtype='float32'), cumulative_length=[100])
    })
    dsp_config = {
        "outputs": ["a_out"],
        "processors": {
            "a_out": {
                "function": "dspeed.processors.pad",
                "args": ["vov_in(shape=150)", "len(vov_in)", 500, "vov_in[0]", "vov_in[-1]+1", "a_out(shape=1000)"],
            }
        }
    }
    tb_out = build_dsp(tb_in, dsp_config=dsp_config, n_entries=1)
    print(tb_out["a_out"].nda)
    assert np.all(tb_out["a_out"].nda[0,:500] == 0)
    assert np.all(tb_out["a_out"].nda[0,500:600] == tb_in["vov_in"][0])
    assert np.all(tb_out["a_out"].nda[0,600:] == 100)
