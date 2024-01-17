import os
from pathlib import Path

import numpy as np
from lgdo import lh5

from dspeed import build_dsp

config_dir = Path(__file__).parent / "configs"


def test_list_parsing(lgnd_test_data, tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"
    dsp_config = {
        "outputs": ["wf_out"],
        "processors": {
            "wf_out": {
                "function": "add",
                "module": "numpy",
                "args": ["[1,2,3,4,5]", "[6,7,8,9,10]", "out=wf_out"],
                "unit": "ADC",
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

    st = lh5.LH5Store()
    df = st.read("geds/dsp/", dsp_file, n_rows=5, field_mask=["wf_out"])[0].view_as(
        "pd"
    )

    assert np.all(df["wf_out"][:] == np.array([7, 9, 11, 13, 15]))
