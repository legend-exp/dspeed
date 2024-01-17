import os
from pathlib import Path

import numpy as np
from lgdo import lh5

from dspeed import build_dsp

config_dir = Path(__file__).parent / "configs"


def test_build_dsp(lgnd_test_data, tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"
    build_dsp(
        f_raw=lgnd_test_data.get_path(
            "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
        ),
        f_dsp=dsp_file,
        dsp_config=f"{config_dir}/numpy-parsing.json",
        write_mode="r",
    )
    assert os.path.exists(dsp_file)


def test_numpy_math_constants_dsp(tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"
    st = lh5.LH5Store()
    df = st.read(
        "geds/dsp/", dsp_file, field_mask=["timestamp", "calc1", "calc2", "calc3"]
    )[0].view_as("pd")

    a1 = df["timestamp"] - df["timestamp"] - np.pi * df["timestamp"]
    a2 = df["timestamp"] - df["timestamp"] - np.pi
    a3 = df["timestamp"] - df["timestamp"] - np.pi * np.e

    f1 = df["calc1"]
    f2 = df["calc2"]
    f3 = df["calc3"]

    assert (a1 == f1).all()
    assert (a2 == f2).all()
    assert (a3 == f3).all()


def test_numpy_infinity_and_nan_dsp(tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"
    st = lh5.LH5Store()
    df = st.read("geds/dsp/", dsp_file, field_mask=["calc4", "calc5", "calc6"])[
        0
    ].view_as("pd")

    assert (np.isnan(df["calc4"])).all()
    assert (np.isneginf(df["calc5"])).all()
    assert (np.isnan(df["calc6"])).all()
