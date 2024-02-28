# DSPeed

```
_____  _________________________________________________________________
     ||                  ____  _____  ____                   __          `,_
     ||                 / __ \/ ___/ / __ \ ___   ___   ____/ /           | `-_
 []  ||  [] [] [] []   / / / /\__ \ / /_/ // _ \ / _ \ / __  /  [] [] []  '-----`-,_
 ====||===============/ /_/ /___/ // ____//  __//  __// /_/ /====================== ``--,_
     ||              /_____//____//_/     \___/ \___/ \__,_/                              ``--,
     ||    ________                                                        ________            )
\____||___/.-.  .-.\______________________________________________________/.-.  .-.\______,,--'
==========='-'=='-'========================================================'-'=='-'=============
```

[![PyPI](https://img.shields.io/pypi/v/dspeed?logo=pypi)](https://pypi.org/project/dspeed/)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/legend-exp/dspeed?logo=git)
[![GitHub Workflow Status](https://img.shields.io/github/checks-status/legend-exp/dspeed/main?label=main%20branch&logo=github)](https://github.com/legend-exp/dspeed/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://img.shields.io/codecov/c/github/legend-exp/dspeed?logo=codecov)](https://app.codecov.io/gh/legend-exp/dspeed)
![GitHub issues](https://img.shields.io/github/issues/legend-exp/dspeed?logo=github)
![GitHub pull requests](https://img.shields.io/github/issues-pr/legend-exp/dspeed?logo=github)
![License](https://img.shields.io/github/license/legend-exp/dspeed)
[![Read the Docs](https://img.shields.io/readthedocs/dspeed?logo=readthedocs)](https://dspeed.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10684779.svg)](https://doi.org/10.5281/zenodo.10684779)

DSPeed (pronounced dee-ess-speed) is a python-based package that performs bulk,
high-performance digital signal processing (DSP) of time-series data such as
digitized waveforms. This package is part of the
[pygama](https://github.com/legend-exp/pygama) scientific computing suite.

DSPeed enables the user to define an arbitrary chain of vectorized signal
processing routines that can be applied in bulk to waveforms and other data
provided using the
[LH5-format](https://legend-exp.github.io/legend-data-format-specs). These
routines can include [numpy
ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html), custom functions
accelerated with [numba](https://numba.pydata.org/), or other arbitrary
functions. DSPeed will carefully manage file I/O to optimize memory usage and
performance. Processing chains are defined using highly portable JSON files
that can be applied to data from multiple digitizers.

See the [online documentation](https://dspeed.readthedocs.io/en/stable/) for
more information.

If you are using this software, consider
[citing](https://doi.org/10.5281/zenodo.10684779)!
