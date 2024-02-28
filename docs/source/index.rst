dspeed
======

|dspeed| is a package for applying Digital Signal Processing to particle
detector digitized signals. The main contents of this package are:

* :mod:`.processors`: A collection of `Numba <http://numba.pydata.org>`_
  functions that perform individual DSP transforms and reductions on our data.
  Available processors include all :class:`numpy.ufunc`\ s as well.
* :class:`.ProcessingChain`: A class that manages and efficiently runs a list
  of DSP processors.
* :func:`.build_processing_chain`: A function that builds a
  :class:`.ProcessingChain` using LH5-formatted input and output files, and a
  JSON configuration file.
* :func:`.build_dsp`: A function that runs :func:`.build_processing_chain` to
  build a :class:`.ProcessingChain` from a JSON config file and then processes
  an input file and writes into an output file, using the LH5 file format.

Getting started
---------------

|dspeed| is published on the `Python Package Index <2_>`_. Install on local
systems with `pip <3_>`_:

.. tab:: Stable release

    .. code-block:: console

        $ pip install dspeed

.. tab:: Unstable (``main`` branch)

    .. code-block:: console

        $ pip install dspeed@git+https://github.com/legend-exp/dspeed@main

.. tab:: Linux Containers

    Get a LEGEND container with |dspeed| pre-installed on `Docker hub
    <https://hub.docker.com/r/legendexp/legend-software>`_ or follow
    instructions on the `LEGEND wiki
    <https://legend-exp.atlassian.net/l/cp/nF1ww5KH>`_.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   manuals/index
   tutorials
   Package API reference <api/modules>

.. toctree::
   :maxdepth: 1
   :caption: Related projects

   LEGEND Data Objects <https://legend-pydataobj.readthedocs.io>
   Decoding Digitizer Data <https://legend-daq2lh5.readthedocs.io>
   pygama <https://pygama.readthedocs.io>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Source Code <https://github.com/legend-exp/dspeed>
   License <https://github.com/legend-exp/dspeed/blob/main/LICENSE>
   Citation <https://zenodo.org/doi/10.5281/zenodo.10684779>
   Changelog <https://github.com/legend-exp/dspeed/releases>
   developer

.. _2: https://pypi.org/project/dspeed
.. _3: https://pip.pypa.io/en/stable/getting-started
.. |dspeed| replace:: *dspeed*
