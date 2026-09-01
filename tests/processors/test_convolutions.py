import numpy as np
import pytest

from dspeed.processors import convolve_wf, fft_convolve_wf


def test_fft_convolve_wf_valid_mode():
    block = np.tile(np.sin(np.arange(100, dtype="float32") / 10), (4, 1))
    kernel = (np.ones(10) / 10).astype("float32")
    w_out = np.full((4, 100 - 10 + 1), np.nan, dtype="float32")

    fft_convolve_wf(block, kernel, np.int8(ord("v")), w_out)

    expected = np.convolve(block[0], kernel, "valid")
    for i in range(4):
        assert np.allclose(w_out[i], expected, rtol=1e-5, atol=1e-6)


def test_fft_convolve_wf_does_not_mutate_input():
    """A NaN'd waveform must produce NaN output without zeroing the input in
    place: w_in is a view into the shared processing-chain buffer that
    downstream processors also read."""
    block = np.tile(np.sin(np.arange(100, dtype="float32") / 10), (4, 1))
    block[2, 50] = np.nan
    orig = block.copy()
    kernel = (np.ones(10) / 10).astype("float32")
    w_out = np.full((4, 100 - 10 + 1), np.nan, dtype="float32")

    fft_convolve_wf(block, kernel, np.int8(ord("v")), w_out)

    # NaN'd waveform -> all-NaN output row
    assert np.all(np.isnan(w_out[2]))
    # clean waveforms still convolved correctly
    expected = np.convolve(orig[0], kernel, "valid")
    assert np.allclose(w_out[0], expected, rtol=1e-5, atol=1e-6)
    # the input block must be untouched
    assert np.array_equal(block, orig, equal_nan=True)


def test_convolve_wf_modes():
    rng = np.random.default_rng(42)
    w = rng.normal(0, 1, (4, 100)).astype("float32")
    k = rng.normal(0, 1, 10).astype("float32")
    for mode, out_len in [("f", 109), ("v", 91), ("s", 100)]:
        w_out = np.full((4, out_len), np.nan, dtype="float32")
        convolve_wf(w, k, np.int8(ord(mode)), w_out)
        mm = {"f": "full", "v": "valid", "s": "same"}[mode]
        for i in range(4):
            ref = np.convolve(w[i].astype("f8"), k.astype("f8"), mm)
            scale = np.abs(ref).max()
            assert np.max(np.abs(w_out[i] - ref)) < 1e-6 * scale, mode


def test_convolve_wf_nan_and_errors():
    from dspeed.errors import DSPFatal

    rng = np.random.default_rng(1)
    w = rng.normal(0, 1, (2, 50)).astype("float32")
    k = np.ones(5, dtype="float32") / 5
    w_out = np.zeros((2, 46), dtype="float32")

    # NaN input waveform -> NaN output row, others still computed
    w[1, 20] = np.nan
    convolve_wf(w, k, np.int8(ord("v")), w_out)
    assert np.all(np.isnan(w_out[1]))
    assert not np.any(np.isnan(w_out[0]))

    # kernel longer than waveform
    with pytest.raises(DSPFatal):
        convolve_wf(
            np.zeros(4, "float32"), k, np.int8(ord("v")), np.zeros(1, "float32")
        )

    # wrong output length
    with pytest.raises(DSPFatal):
        convolve_wf(w[0], k, np.int8(ord("v")), np.zeros(45, "float32"))

    # invalid mode
    with pytest.raises(DSPFatal):
        convolve_wf(w[0], k, np.int8(ord("x")), np.zeros(46, "float32"))


def test_fft_convolve_wf_1d_input():
    # single 1-D waveform (not a block): NaN masking must handle the
    # 0-d reduction result
    w = np.sin(np.arange(100, dtype="float32") / 10)
    k = (np.ones(10) / 10).astype("float32")
    w_out = np.full(91, np.nan, dtype="float32")
    fft_convolve_wf(w, k, np.int8(ord("v")), w_out)
    assert np.allclose(w_out, np.convolve(w, k, "valid"), rtol=1e-5, atol=1e-6)

    w_nan = w.copy()
    w_nan[50] = np.nan
    orig = w_nan.copy()
    fft_convolve_wf(w_nan, k, np.int8(ord("v")), w_out)
    assert np.all(np.isnan(w_out))
    assert np.array_equal(w_nan, orig, equal_nan=True)
