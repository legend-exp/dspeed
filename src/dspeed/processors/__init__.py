r"""
Contains a list of DSP processors, implemented using Numba's
:func:`numba.guvectorize` to implement NumPy's :class:`numpy.ufunc` interface.
In other words, all of the functions are void functions whose outputs are given
as parameters.  The :class:`~numpy.ufunc` interface provides additional
information about the function signatures that enables broadcasting the arrays
and SIMD processing. Thanks to the :class:`~numpy.ufunc` interface, they can
also be called to return a NumPy array, but if this is done, memory will be
allocated for this array, slowing things down.

The dspeed processors use the :class:`~numpy.ufunc` framework, which is
designed to encourage highly performant python practices. These functions have
several advantages:

1. They work with :class:`numpy.array`. NumPy arrays are arrays of same-typed
   objects that are stored adjacently in memory (equivalent to a dynamically
   allocated C-array or a C++ vector). Compared to standard Python lists, they
   perform computations much faster. They are ideal for representing waveforms
   and transformed waveforms.

2. They perform vectorized operations. Vectorization causes commands to
   perform the same operation on all components of an array. For example, the
   :class:`~numpy.ufunc` ``np.add(a, b, out=c)`` (equivalently ``c=a+b``) is
   equivalent to: ::

       for i in range(len(c)): c[i] = a[i] + b[i]

   Loops are slow in python since it is an interpreted language; vectorized
   commands remove the loop and only call the Python interpreter once.

   Furthermore, :class:`~numpy.ufunc`\ s are capable of `broadcasting
   <https://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting>`_
   their dimensions. This involves a safety check to ensure the dimensions of
   ``a`` and ``b`` are compatible sizes. It also will automatically replicate a
   ``size-1`` dimension over another one, enabling the addition of a scalar to
   a vector quantity. This is useful, as it allows us to process multiple
   waveforms at once.

   One of the biggest advantages of vectorized :class:`~numpy.ufunc`\ s is that
   many of them will take advantage of SIMD (same input-multiple data)
   vectorization on a vector-CPU. Modern CPUs typically have 256- or 512-bit
   wide processing units, which can accommodate multiple 32- or 64-bit numbers.
   Programming with these, however, is quite difficult and requires specialized
   commands to be called.  Luckily for us, many NumPy :class:`~numpy.ufunc`\ s
   will automatically use these for us, speeding up our code!

3. :class:`~numpy.ufunc`\ s are capable of calculating their output in place,
   meaning they can place calculated values in pre-allocated memory rather than
   allocating and returning new values. This is important because memory
   allocation is one of the slowest processes computers can perform, and should
   be avoided. With :class:`~numpy.ufunc`\ s, this can be done using the out
   keyword in arguments (ex ``numpy.add(a, b, out=c)``, or more succinctly,
   ``numpy.add(a, b, c)``).  While this may seem counterintuitive at first, the
   alternative (``c = np.add(a,b)`` or ``c = a+b``) causes an entirely new
   array to be allocated, with c pointing at that. These array allocations can
   add up very quickly: ``e = a*b + c*d``, for example, would allocate 3
   different arrays: one for ``a*b``, one for ``c*d``, and one for the sum of
   those two. As we write :class:`~numpy.ufunc`\ s, it is important that we try
   to use functions that operate in place as much as possible!
"""

from importlib import import_module

# Mapping from function to name of module in which it is defined
# To add a new function to processors, it must be added here!
_modules = {
    "bl_subtract": "bl_subtract",
    "convolve_damped_oscillator": "convolutions",
    "convolve_exp": "convolutions",
    "convolve_wf": "convolutions",
    "fft_convolve_wf": "convolutions",
    "discrete_wavelet_transform": "dwt",
    "cusp_filter": "energy_kernels",
    "dplms": "energy_kernels",
    "zac_filter": "energy_kernels",
    "fixed_time_pickoff": "fixed_time_pickoff",
    "gaussian_filter1d": "gaussian_filter1d",
    "get_multi_local_extrema": "get_multi_local_extrema",
    "get_wf_centroid": "get_wf_centroid",
    "histogram": "histogram",
    "histogram_around_mode": "histogram",
    "histogram_peakstats": "histogram_stats",
    "histogram_stats": "histogram_stats",
    "iir_filter": "iir_filter",
    "notch_filter": "iir_filter",
    "peak_filter": "iir_filter",
    "inject_damped_oscillation": "inject_ringing",
    "inl_correction": "inl_correction",
    "moving_slope": "kernels",
    "step": "kernels",
    "t0_filter": "kernels",
    "linear_slope_fit": "linear_slope_fit",
    "linear_slope_diff": "linear_slope_fit",
    "log_check": "log_check",
    "min_max": "min_max",
    "min_max_norm": "min_max",
    "classification_layer_no_bias": "ml",
    "classification_layer_with_bias": "ml",
    "dense_layer_no_bias": "ml",
    "dense_layer_with_bias": "ml",
    "normalisation_layer": "ml",
    "avg_current": "moving_windows",
    "moving_window_left": "moving_windows",
    "moving_window_multi": "moving_windows",
    "moving_window_right": "moving_windows",
    "multi_a_filter": "multi_a_filter",
    "multi_t_filter": "multi_t_filter",
    "remove_duplicates": "multi_t_filter",
    "optimize_nnls": "nnls",
    "optimize_1pz": "optimize",
    "optimize_2pz": "optimize",
    "param_lookup": "param_lookup",
    "peak_snr_threshold": "peak_snr_threshold",
    "inject_general_logistic": "pmt_pulse_injector",
    "inject_gumbel": "pmt_pulse_injector",
    "pole_zero": "pole_zero",
    "double_pole_zero": "pole_zero",
    "poly_diff": "poly_fit",
    "poly_exp_rms": "poly_fit",
    "poly_fit": "poly_fit",
    "presum": "presum",
    "inject_exp_pulse": "pulse_injector",
    "inject_sig_pulse": "pulse_injector",
    "rc_cr2": "rc_cr2",
    "recursive_filter": "recursive_filter",
    "round_to_nearest": "round_to_nearest",
    "floor_to_nearest": "round_to_nearest",
    "ceil_to_nearest": "round_to_nearest",
    "trunc_to_nearest": "round_to_nearest",
    "saturation": "saturation",
    "soft_pileup_corr": "soft_pileup_corr",
    "soft_pileup_corr_bl": "soft_pileup_corr",
    "svm_predict": "svm",
    "tf_model": "tf_model",
    "time_over_threshold": "time_over_threshold",
    "bi_level_zero_crossing_time_points": "time_point_thresh",
    "interpolated_time_point_thresh": "time_point_thresh",
    "multi_time_point_thresh": "time_point_thresh",
    "time_point_thresh": "time_point_thresh",
    "asym_trap_filter": "trap_filters",
    "trap_filter": "trap_filters",
    "trap_norm": "trap_filters",
    "trap_pickoff": "trap_filters",
    "interpolating_upsampler": "upsampler",
    "upsampler": "upsampler",
    "wf_alignment": "wf_alignment",
    "wf_correction": "wf_correction",
    "where": "where",
    "wiener_filter": "wiener_filter",
    "windower": "windower",
}

__all__ = list(_modules)


# Lazy loader
def __getattr__(name):
    if name in _modules:
        mod_name = _modules[name]
        mod = import_module(f".{mod_name}", __name__)
        funs = {f: getattr(mod, f) for f, m in _modules.items() if m == mod_name}
        globals().update(funs)
        return funs[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__ + list(globals().keys())
