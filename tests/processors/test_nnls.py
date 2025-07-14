import numpy as np

from dspeed.processors import optimize_nnls


def test_nnls(compare_numba_vs_python):

    def gumbel_pdf(x, mu, sigma):
        beta = sigma * (np.sqrt(6) / np.pi)
        z = (x - mu) / beta
        w = (1 / beta) * np.exp(-(z + np.exp(-1 * z)))
        return w / np.sum(w)

    # create a Gumbler signal (with bad resolution) with pulses at 40, 80, 82 ns, but known sigma of 2ns
    xd = np.arange(0, 150, 4.8)  # ns
    mean = [40, 40, 80, 82]
    yd = np.zeros(len(xd))
    for m in mean:
        yd += gumbel_pdf(xd, m, 2)

    # generate coefficient matrix with a  Gumbler template (2 ns sigma) at 24x higher sampling rate
    x = np.arange(0, 150, 0.2)  # ns
    a = np.zeros((len(xd), len(x)))
    for i in range(len(x)):
        a[:, i] = gumbel_pdf(xd, x[i], 2)

    kernel = np.zeros(len(x))
    optimize_nnls(a, yd, 1000, 1e-9, False, np.float32(0), kernel)

    # check number of pulses reconstructed (overall 4, with 3 different time positions i.e. 2 at the same spot)
    assert len(kernel[kernel > 1.99]) == 1
    assert len(kernel[kernel > 0.01]) == 3
    assert round(np.sum(kernel[kernel > 0.01])) == 4

    # Check if time reconstruction is correct
    assert np.all(
        np.sort(x[np.where(kernel > 0.01)]).astype(int) == np.sort(list(set(mean)))
    )
    assert np.sort(x[np.where(kernel > 0.01)]).astype(int)[0] == mean[0]

    # compare numba and python results
    kernel = np.zeros(len(x))
    compare_numba_vs_python(
        optimize_nnls, a, yd, 1000, 1e-9, False, np.float32(0), kernel
    )
