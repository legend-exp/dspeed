import os
import subprocess
from pathlib import Path

config_dir = Path(__file__).parent / "configs"


def test_cli(lgnd_test_data, tmptestdir):
    subprocess.check_call(["dspeed", "--help"])
    subprocess.check_call(
        [
            "dspeed",
            "--overwrite",
            "--config",
            f"{config_dir}/icpc-dsp-config.json",
            "--max-rows",
            "10",
            "--output",
            f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        ]
    )

    assert os.path.exists(f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5")
