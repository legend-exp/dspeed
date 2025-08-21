from pathlib import Path

from dspeed.vis import WaveformAndHistBrowser

config_dir = Path(__file__).parent / "configs"


def test_basics(lgnd_test_data):
    wb = WaveformAndHistBrowser(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        "/geds/raw",
        dsp_config=f"{config_dir}/hpge-dsp-histo-config.yaml",
        lines=["waveform", "wf_mode"],
        styles="seaborn-v0.8",
        hist_values_edges=("wf_hist", "wf_borders"),
        hist_styles=[{"color": ["red", "green"]}],
    )

    wb.draw_next()
    wb.draw_entry(24)
    wb.draw_entry([2, 24])


def test_solo_and_log(lgnd_test_data):
    wb = WaveformAndHistBrowser(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        "/geds/raw",
        dsp_config=f"{config_dir}/hpge-dsp-histo-config.yaml",
        lines=[],
        hist_values_edges=("wf_hist", "wf_borders"),
        hist_styles=[{"color": ["red", "green"]}],
        hist_log=True,
        vertical_hist=True,
    )

    wb.draw_next()
    wb.draw_entry(24)
    wb.draw_entry([2, 24])
