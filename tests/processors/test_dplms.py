from pathlib import Path

import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import dplms


def test_dplms(compare_numba_vs_python):
    with open(Path(__file__).parent / "dplms_noise_mat.dat") as f:
        nmat = [[float(num) for num in line.split(" ")] for line in f]

    kernel = np.zeros(50)
    len_wf = 100
    ref = np.zeros(len_wf)
    ref[int(len_wf / 2 - 1) : int(len_wf / 2)] = 1

    # ensure the DSPFatal is raised for a negative length
    with pytest.raises(DSPFatal):
        dplms(nmat, [], 1, 1, 1, 1, kernel)

    # ensure the DSPFatal is raised for negative coefficients
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, -1, 1, 1, 1, kernel)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, 1, -1, 1, 1, kernel)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, 1, 1, -1, 1, kernel)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, 1, 1, 1, -1, kernel)
    with pytest.raises(DSPFatal):
        dplms(nmat, ref, 1, 1, 1, 2, kernel)

    assert np.all(compare_numba_vs_python(dplms, nmat, ref, 1, 1, 1, 1, kernel))
