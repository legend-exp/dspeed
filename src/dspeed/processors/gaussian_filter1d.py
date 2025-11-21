"""Processors applying one-dimensional Gaussian smoothing filters."""

# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# All this code belongs to the team that coded Scipy, found at this link:
#     https://github.com/scipy/scipy/blob/v1.6.0/scipy/ndimage/filters.py#L210-L260
# The only thing changed was the calculation of the convolution, which
# originally called a function from a C library.  In this code, the convolution is
# performed with NumPy's built in convolution function.
from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32, float32, float32[:])",
        "void(float64, float64, float64[:])",
    ],
    "(),(),(n)",
    **nb_kwargs(
        forceobj=True,
    ),
)
def gaussian_filter1d(sigma: int, truncate: float, weights: np.ndarray) -> None:
    """1-D Gaussian filter.

    Note
    ----
    This processor is composed of a factory function that is called using the
    `init_args` argument. The input and output waveforms are passed using
    `args`.

    Parameters
    ----------
    sigma
        standard deviation for Gaussian kernel
    truncate
        truncate the filter at this many standard deviations.
    """

    # Make the radius of the filter equal to truncate standard deviations
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)

    sigma2 = sigma * sigma
    x = np.arange(-lw, lw + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    weights[:] = np.asarray(phi_x, dtype=np.float64)
