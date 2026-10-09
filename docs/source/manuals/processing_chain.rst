Building a Processing Chain
===========================

A :class:`~.processing_chain.ProcessingChain` is the core object that runs a
sequence of digital-signal-processing (DSP) transforms on waveform data in
``dspeed``.  It stores a set of named variables and a list of processors that
operate on those variables. Processors must be vectorized, preferably using
the `ufunc <https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-basics>`_
or `generalized ufunc <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html#c-api-generalized-ufuncs>`_
interfaces. The best way to produce such processors is to use
:class:`~numba.guvectorize` or `numpy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`_.

Most users build a :class:`~.processing_chain.ProcessingChain` using :func:`~.build_dsp.build_dsp`
to read in a YAML or JSON configuration file or a python dict, and process an input LH5 file,
:class:`~lgdo.Table`, or :class:`~lh5.LH5Iterator`. The :class:`~vis.waveform_browser.WaveformBrowser`
is likewise capable of reading these config files to interactively analyze and visualize
the effects of processors on waveforms. This page explains how configuration
files work.

Configuration overview
----------------------

A processing-chain configuration is a dictionary with two top-level keys:

- ``outputs`` -- a list of parameter names to keep in the output table by
  default. Note that this may be overridden using input arguments!
- ``processors`` -- a dictionary describing how to compute every derived
  quantity.

When a ProcessingChain is constructed, it will only include the minimal set
of processors required to build the output parameters, and will exclude
others (unless they are requested using input args).

A basic entry in ``processors`` has the form:

.. code-block:: yaml

   "output1":
     function: module.processor_name
     args: [input1, input2, output1(unit=u)]

This block defines a processor that uses the gufunc ``processor_name``
from python module ``module`` to process ``input1`` and ``input2`` in order
to produce ``output1``, which has units of ``u``. The name of the block
is the output of the processor.

The full spec is as follows:

.. code-block:: yaml

   "output1[, output2...]":
     function: [module].<processor_name> | <expression>
     [module]: <module>
     args:
     - input1[(shape=<size>|(<shape>), dtype=<dtype>, unit=<u>, vector_len=<variable>, period=<q>, offset=<q>, grid=(<period>, <offset>), is_coord=<bool>)]
     - <value>
     - <expression>
     - db.some.param
     - output1[(shape=<size>|(<shape>), dtype="<dtype>", unit=<u>, vector_len=<variable>, period=<q>, offset=<q>, grid=(<period>, <offset>), is_coord=<bool>)]
     - <kwarg>=
     [kwargs]:
       signature: "(n),()->()"
       types: 'fff'
     [init_args]:
     - ...same as args
     [unit]: [<output1 unit>, <output2 unit>]
     [defaults]:
       db.some.param: 42

The key is a comma-separated list of parameter names produced by the processor.
The body tells ``dspeed`` which function to call and how its arguments map to
chain variables or constants. The fields are as follows:

``function``
  Name of the processor function. A dotted module path can be provided (e.g.
  ``dspeed.processors.trap_norm``), or just the function name can be provided;
  if just a name is used then the ``module`` field must be provided. This can
  also be a short arithmetic expression that will be assigned to the output;
  see below on arithmetic expressions.

``module``
  Python module that contains the function. Not needed if the module is included
  in the ``function`` field.

``args``
  List of arguments passed to the processor. Arguments can take several forms:

  - named variables (e.g. ``input1``). Array and unit construction arguments
    may be provided in parenthesis; if not provided, the ProcessingChain will
    to the best of its ability infer the correct values based on the signature
    of the processor and the values associated with other input variables.

    The construction arguments are:

    - ``shape``: the shape of the variable for a single entry; can be provided
      as a tuple or integer length
    - ``dtype``: the datatype of the variable; any string accepted by :class:`numpy.dtype`
    - ``unit``: the unit of the variable, which will be written to the output file.
      A unit found in the pint unit registry will enable unit conversions
      to be performed on the variable (e.g. convert from ``ns`` to ``us`` by dividing
      by 1000). A string value not in the unit registry will be treated as unitless,
      but still written
    - ``vector_len`` for variable length/ragged arrays, an integer variable containing
      the length of each entry
    - ``period`` for waveforms, the separation (with a unit) between samples
    - ``offset`` for waveforms, the relative offset of the samples
    - ``grid`` tuple of period and offset
    - ``is_coord`` for variables that can be used to index waveforms (e.g. timepoints),
      this will be set to true; unit conversions using the period and offset of the
      waveform will be performed before passing the value into the processor. The automated
      behavior will set this to ``True`` if the unit for an output matches the period of an
      input waveform.

    Note that in expressions, these attributes may be accessed as members using `variable.attr`;
    these may be used as arguments for other variables (e.g. ``output(period=input.period//2)``).

  - Constants: constant values passed to parameters. Units can be provided by multiplying
    onto other values; these will be converted into unitless values based on the units and
    periods of other variables

  - Arithmetic expressions: basic arithmetic expressions (see below) that may include
    other named variables and constants

  - Database parameters: names, potentially with dotted attributes, that are preceded by
    db are looked up in the database dictionary provided when constructing the ProcessingChain

  - ``<kwarg> = <arg>``: arg can be any of the above; this will be passed as a keyword argument
    to the processor

``kwargs``
  Optional keyword arguments forwarded to
  :meth:`~.processing_chain.ProcessingChain.add_processor`:
  - ``signature``: a gufunc-style size signature (e.g. ``"(n),()->()"``)
  - ``types``: a list of type characters used for the type signature (e.g. ``fif``)

``init_args``
  Optional list of arguments used by factory functions used to generate processors

``unit``
  Optional units attached to output variables. Pint units will be used for conversions.
  Note that it is preferred to provide these as construction arguments as above.

``defaults``
  Default values for ``db.*`` arguments (see below).

Expressions
-----------
Simple arithmetic expressions may be used as arguments for processors. Internally,
these expressions will be evaluated on constant values, or will produce one or more
additional processors with output variables that are fed as inputs
to the processor. These are documented in :meth:`processing_chain.ProcessingChain.get_variable`.

Expressions can be constructed from:

- Unary and binary operators :obj:`+`, :obj:`-`, :obj:`*`, :obj:`/`,
  :obj:`//` are available.
- ``varname[slice]``: return the variable with a slice applied. Slice
  values can be ``float``\ s, and will have round applied to them
- ``len(expr)``: return the length of the array found with ``expr``
- ``a if b else c``: see ``where``; return value held in ``a`` if ``b``
  is ``True``, else ``c``

In addition, several built in functions are provided:

- ``astype(expr, dtype)``: cast ``expr`` to ``dtype``
- ``round(expr, to_nearest = 1, [dtype])``: return the value found with
    ``expr`` rounded to the nearest multiple of ``to_nearest``
- ``floor(expr, to_nearest = 1, [dtype])``: return the value found with
    ``expr`` rounded to last multiple of ``to_nearest`` smaller
- ``ceil(expr, to_nearest = 1, [dtype])``: return the value found with
    ``expr`` rounded to first multiple of ``to_nearest`` larger
- ``trunc(expr, to_nearest = 1, [dtype])``: return the value found with
    ``expr`` rounded to first multiple of ``to_nearest`` towards zero
- ``where(condition, a, b, [dtype])``: if ``condition`` is ``True`` return the
    value held in ``a``, else ``b``
- ``isnan(expr)``: return ``True`` if ``expr`` is ``NaN``
- ``isfinite(expr)``: return ``True`` if not ``NaN`` ``inf`` or ``-inf``
- ``loadlh5(file, group)``: load LH5 object held in ``group`` of lh5
    file. Returned object will be treated as a const.

An expression can be provided for the ``function`` in a processor YAML block;
in this case, no arguments should be provided. An additional shorthand can
be used for this:

.. code-block:: yaml

   output: expression

Inputs, outputs, and dependencies
---------------------------------
``dspeed`` figures out which variables to plug in as arguments based on
the keys of the processors table. Dependencies will be automatically resolved,
so processors can be listed in any order, and processors not needed for outputs
will not be included.

Any parameter name that appears in ``args`` but is not itself defined in
``processors`` is treated as an input quantity that must be present in the
input table. Unprovided construction arguments will be set based on the
inputs from the file.

Parameters listed as outputs will be copied into the output buffer. If the
variable has a unit, it will be included as an attribute. The :class:`lgdo.LGDOType`
of the output will be inferred from the properties of the variable; e.g., if a
variable has a coordinate grid defined and is not a coordinate, it will be
written as a :class:`lgdo.WaveformTable`; if it has a ``vector_len`` it will be
written as a :class:`lgdo.VectorOfVectors`.

Units and coordinates
---------------------
Variables (e.g. ``var(unit=ns)``) and constant values (e.g. ``5*ns``) can have units,
provided by pint. Units will be recorded to output files using the ``"units"``
attribute, and will be automatically converted for certain arithmetic operations (e.g.
``5*ns/us = 0.005``). Processors are not capable of working with units; in ordinary
cases, the magnitude will be directly passed to the processor, so if any unit changes
are required they should be made explicit.

Some unitful values represent coordinates, which ``dspeed`` will automatically
convert to unitless values to pass as array indices, based on the coordinate
grids of waveforms. A :class:`CoordinateGrid` consists of a period and an offset.
For example, in the case of:

.. code-block:: yaml

    output
      function: dspeed.processors.fixed_time_pickoff
      args: [waveform, 1*us, output]

``1*us`` will be automatically divided by the period of the waveform (so if it is ``10*ns``
the value ``100`` will be passed to the processor). This will also happen with variables;
in the case of variables, if a coordinate grid has an offset, this will also be subtracted:

.. code-block:: yaml

    wf_windowed
      function: dspeed.processors.windower
      args: [waveform, 1*us, wf_windowed(shape=len(waveform)//4)]
    timepoint
      function: numpy.argmax
      args: [wf_windowed, axis=-1, out=timepoint(unit=ns)]
      kwargs:
        signature: (n),()->()
        types: fff
    amplitude
      function: dspeed.processors.fixed_time_pickoff
      args: [waveform, timepoint, "'i'", amplitude(unit=waveform.unit)]

In this case, when we call argmax, the output will be offset by 100 due to the
waveform windowing; when we pickoff the value from the un-windowed waveform,
100 will be added back into the index to ensure that the correct sample is
selected. This is due to the default behavior of treating values with units
compatible with the coordinate grids of waveforms as coordinates.

To override this default behavior, one can set ``is_coord=False`` when
defining a variable. Likewise, if you want this behavior, but the default
doesn't provide it (e.g. unitless values), then you can pass ``is_coord=True``.
To explicitly round to an integer index of a waveform, ``round(t, wf.grid)``
can be used.

Debugging and profiling
-----------------------
If your processing chain is not behaving as expected, several tools exist to
understand what is happening. First, the :class:`WaveformBrowser` can be used
to interactively process waveforms and visualize outputs.

In addition, turning on debug logging will output in granular detail how
variables are constructed, what automated decisions are made, what implicit
conversions and operations are performed, and how variables and constants are
fed into processors.

.. code-block:: python

    from dspeed import log
    log.setLevel("DEBUG")


If running in jupyter, you may also have to run ``logging.basicConfig(stream=sys.stdout)``.
The output will look like::

    ...
    DEBUG:dspeed.processing_chain:prereqs for bl, bl_sig, slope, intercept are ['waveform']
    DEBUG:dspeed.processing_chain:prereqs for wf_blsub are ['waveform', 'bl']
    DEBUG:dspeed.processing_chain:prereqs for wf_pz are ['wf_blsub']
    DEBUG:dspeed.processing_chain:prereqs for wf_trap are ['wf_pz']
    ...
    DEBUG:dspeed.processing_chain:added variable: wf_blsub(shape: auto, dtype: auto, grid: auto, unit: ADC, is_coord: auto)
    DEBUG:dspeed.processing_chain:updated variable: wf_blsub(shape: (5592,), dtype: float32, grid: (16.0 ns,waveform__t0), unit: ADC, is_coord: False)
    DEBUG:dspeed.processing_chain:added processor: subtract(waveform, bl, wf_blsub)
    DEBUG:dspeed.processing_chain:added variable: wf_pz(shape: auto, dtype: auto, grid: auto, unit: ADC, is_coord: auto)
    DEBUG:dspeed.processing_chain:updated variable: wf_pz(shape: (5592,), dtype: float32, grid: (16.0 ns,waveform__t0), unit: ADC, is_coord: False)
    DEBUG:dspeed.processing_chain:added processor: pole_zero(wf_blsub, 180 µs, wf_pz)
    ...

This includes valuable information to track how dependencies are resolved, how variables are defined (including
what arguments are automated, and how the automated decisions are updated), and what is fed
to processors.

To profile processors, for speed or for memory, standard profiling tools may be used.
Each processor will show up as a separate function in the profiler stats, named for the
function used and the arguments provided.

Database arguments
------------------
Processor arguments can pull values from a runtime database by using the prefix
``db.``.  For example:

.. code-block:: yaml

   args: ["waveform", "db.pz.tau", "wf_pz"]
   defaults:
     db.pz.tau: "27460.5"

When the chain is built, ``db.pz.tau`` is looked up in the ``db_dict`` passed to
:func:`~.processing_chain.build_processing_chain`.  If the key is missing, the
value from ``defaults`` is used.  If neither is available, an error is raised.

Example Processor Blocks
------------------------
In this section, we will attempt to provide a set of illustrative examples
of several common patterns seen in processors beyond the simple one listed
above

**Processor that isn't a ufunc or gufunc**
If you want to use a function that is vectorized, but is not a ufunc
or gufunc, you must use the ``kwargs`` to provide a gufunc-style signature.
Oftentimes, these sorts of functions will also require kwargs (such as
specifying ``axis=-1`` to loop over the last dimension; don't forget that
processors are vectorized and implicitly have an outer dimension equal to
the size of the block of inputs being processed!)

Example: use numpy to get the mean

    .. code-block:: yaml

        bl_mean
          function: numpy.mean
          args:
            - waveform
            - axis=-1
            - out=bl_mean(unit='ADC')
          kwargs:
            signature: "(n),()->()"
            types: "fff"


**Changing the shape of the output**
If the output has a size that is different from any inputs, many of its properties
cannot be inferred, and must be provided explicitly. This is the most common reason
for needing the ``shape``, ``period``, and ``offset`` constructor arguments.

Example: window a waveform 5*us from the start, with 1/4 the length:

    .. code-block:: yaml

        wf_windowed:
          function: dspeed.processors.windower
          args:
            - waveform
            - 5*us
            - wf_windowed(len(waveform)//4, offset=5*us, period=waveform.period)

**Initializing constants with processors**
If a const value cannot be expressed or requires computation at initialization,
a const processor can be added by having all inputs to the processor be
const values. In this case, the processor itself will be run at initialization,
and will not re-processed for each pass. This is particularly useful, e.g., for
convolution kernels.

Example: cusp filter kernel

    .. code-block:: yaml

      cusp_kernel:
        function: dspeed.processors.cusp_filter
        args:
          - 20*us/wf_pz.period
          - round(3*us, wf_pz.period)
          - .inf
          - cusp_kernel(shape=len(wf_pz) - round(4.8*us, wf_pz.period), dtype='f')
      wf_cusp:
        function: dspeed.processors.fft_convolve_wf
        args:
          - wf_pz
          - cusp_kernel
          - "'v'"
          - wf_cusp(shape=round(4.8*us, wf_pz.period)+2, dtype='f')

**Factory functions**
In other situations requiring filter initialization, a factory can be used
which produces the gufunc-compatible object and returns it to be called by
the ProcessingChain. The arguments used by the factory function use the
``init_args`` field, while the arguments for the generated function are
defined using ``args``.

Example: scipy IIR filter; this is a ``dspeed`` factory processor that
builds a wrapper around the scipy IIR filters.

    .. code-block:: yaml

      function: dspeed.processors.iir_filter
      init_args:
        - "15*MHz"
        - 4
        - wf
        - "ftype=butter"
        - "btype=lowpass"
      args:
        - wf
        - "wf_lp(unit=ADC)"
