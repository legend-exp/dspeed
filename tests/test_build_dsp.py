import os
from pathlib import Path

import numpy as np
import pytest
from lgdo import Struct, Table, VectorOfVectors, lh5
from test_utils import isclose

from dspeed import build_dsp

config_dir = Path(__file__).parent / "configs"


@pytest.fixture(scope="session")
def dsp_test_file_geds(lgnd_test_data, tmptestdir):
    out_name = f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        out_name,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert os.path.exists(out_name)

    return out_name


def test_build_dsp_yaml(lgnd_test_data, dsp_test_file_geds):
    dsp_out = build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        dsp_config=f"{config_dir}/icpc-dsp-config-yaml.yaml",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert isinstance(dsp_out, Struct)

    # TODO: make sure the configs are the same for json and yaml so we can do this
    # dsp_file = lh5.read("geds/dsp", dsp_test_file_geds)
    # assert isclose(dsp_out, dsp_file)


def test_build_dsp_errors(lgnd_test_data, tmptestdir):
    with pytest.raises(FileExistsError):
        build_dsp(
            lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
            f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
        )

    with pytest.raises(FileNotFoundError):
        build_dsp(
            "non-existent-file.lh5",
            f"{tmptestdir}/LDQTA_r117_20200110T105115Z_cal_geds_dsp.lh5",
            dsp_config=f"{config_dir}/icpc-dsp-config.json",
            write_mode="r",
        )


# test different input types
def test_dsp_in_types(lgnd_test_data):
    # input from file
    raw_path = lgnd_test_data.get_path(
        "lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"
    )
    dsp_file = build_dsp(
        raw_path,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        lh5_tables="geds/raw",
        database={"pz": {"tau": 27460.5}},
    )
    assert isinstance(dsp_file, Struct)
    assert "geds" in dsp_file and "dsp" in dsp_file["geds"]

    # input iterator directly
    raw_it = lh5.LH5Iterator(raw_path, "geds/raw")
    dsp_it = build_dsp(
        raw_it,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        lh5_tables="geds/raw",
        database={"pz": {"tau": 27460.5}},
    )
    assert isinstance(dsp_it, Struct)
    assert "geds" in dsp_it and "dsp" in dsp_it["geds"]

    # input table directly
    raw_tb = lh5.read("geds/raw", raw_path)
    dsp_tb = build_dsp(
        raw_tb,
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        lh5_tables="geds/raw",
        database={"pz": {"tau": 27460.5}},
    )
    assert isinstance(dsp_tb, Struct)
    assert "geds" in dsp_tb and "dsp" in dsp_tb["geds"]

    # make sure these all give the same result
    assert isclose(dsp_file, dsp_it)
    assert isclose(dsp_file, dsp_tb)


# test different output types
def test_dsp_out_struct(lgnd_test_data, dsp_test_file_geds):
    dsp_tb = lh5.read("geds/dsp", dsp_test_file_geds)
    assert isinstance(dsp_tb, Table)

    dsp_st = build_dsp(
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        dsp_config=f"{config_dir}/icpc-dsp-config.json",
        database={"pz": {"tau": 27460.5}},
        write_mode="r",
    )
    assert isinstance(dsp_st, Struct)
    assert "geds" in dsp_st and "dsp" in dsp_st["geds"]
    assert len(dsp_tb) == len(dsp_st["geds"]["dsp"])
    assert isclose(dsp_tb, dsp_st["geds"]["dsp"])


# test input field in dsp config
def test_aux_inputs(lgnd_test_data, dsp_test_file_geds):
    raw_in = lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5")

    # This config will find wf_min in an already calculated dsp file
    # and recalculate it as wf_min2 and make sure they are the same
    dsp_config = {
        "inputs": [{"file": dsp_test_file_geds, "group": "geds/dsp", "suffix": "1"}],
        "outputs": ["compare", "tp_max1", "tp_max2"],
        "processors": {
            "tp_min2, tp_max2, wf_min2, wf_max2": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min2", "tp_max2", "wf_min2", "wf_max2"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            },
            "compare": "tp_max1 - tp_max2",
        },
    }

    dsp_out = build_dsp(
        raw_in,
        lh5_tables="geds/raw",
        dsp_config=dsp_config,
    )
    assert np.all(np.isclose(dsp_out["geds"]["dsp"]["compare"], 0))

    # Test using database inputs
    dsp_config = {
        "inputs": [{"file": "db.file", "group": "db.group", "suffix": "1"}],
        "outputs": ["compare", "tp_max1", "tp_max2"],
        "processors": {
            "tp_min2, tp_max2, wf_min2, wf_max2": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min2", "tp_max2", "wf_min2", "wf_max2"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            },
            "compare": "tp_max1 - tp_max2",
        },
    }

    dsp_out = build_dsp(
        raw_in,
        lh5_tables="geds/raw",
        dsp_config=dsp_config,
        database={
            "geds": {
                "file": dsp_test_file_geds,
                "group": "geds/dsp",
            }
        },
    )
    assert np.all(np.isclose(dsp_out["geds"]["dsp"]["compare"], 0))


@pytest.fixture(scope="session")
def dsp_test_file_spm(lgnd_test_data, tmptestdir):
    chan_config = {
        "ch0/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch1/raw": f"{config_dir}/sipm-dsp-config.json",
        "ch2/raw": f"{config_dir}/sipm-dsp-config.json",
    }

    out_file = f"{tmptestdir}/L200-comm-20211130-phy-spms_dsp.lh5"
    build_dsp(
        lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5"),
        out_file,
        {},
        n_entries=5,
        lh5_tables=chan_config.keys(),
        chan_config=chan_config,
        write_mode="r",
    )

    assert os.path.exists(out_file)

    return out_file


def test_build_dsp_spms_channelwise(dsp_test_file_spm):
    assert lh5.ls(dsp_test_file_spm) == ["ch0", "ch1", "ch2"]
    assert lh5.ls(dsp_test_file_spm, "ch0/") == ["ch0/dsp"]
    assert lh5.ls(dsp_test_file_spm, "ch0/dsp/") == [
        "ch0/dsp/energies",
        "ch0/dsp/trigger_pos",
    ]

    lh5_obj = lh5.read("/ch0/dsp/energies", dsp_test_file_spm)
    assert isinstance(lh5_obj, VectorOfVectors)
    assert len(lh5_obj) == 5
