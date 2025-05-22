import os

from lgdo import lh5

from dspeed import build_dsp


def test_histogram_fixed_width(lgnd_test_data, tmptestdir):
    dsp_file = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds__numpy_test_dsp.lh5"
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
        "geds/dsp/", dsp_file, "pd", field_mask=["hist_weights", "hist_borders"]
    )

    assert len(df["hist_weights"][0]) + 1 == len(df["hist_borders"][0])
    for i in range(2, len(df["hist_borders"][0])):
        a = df["hist_borders"][0][i - 1] - df["hist_borders"][0][i - 2]
        b = df["hist_borders"][0][i] - df["hist_borders"][0][i - 1]
        assert round(a, 2) == round(b, 2)
