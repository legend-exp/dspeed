import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import histogram

def test_histogram_fixed_width(compare_numba_vs_python):
    vals = np.arange(100)*2/3
    with pytest.raises(DSPFatal):
        histogram(vals, np.zeros(10), np.zeros(10))
    
    hist_weights = np.zeros(66)
    hist_edges = np.zeros(67)
    histogram(vals, hist_weights, hist_edges)
    assert(all(hist_edges == np.arange(67)))
    assert all(hist_weights[0::2]==2) and all(hist_weights[1::2]==1)

    vals[5] = np.nan
    histogram(vals, hist_weights, hist_edges)
    assert all(np.isnan(hist_edges))
    assert all(hist_weights == 0)
