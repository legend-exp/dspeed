from __future__ import annotations

import pickle
from typing import Callable

import numpy as np

from ..utils import GUFuncWrapper


def svm_predict(svm_file: str) -> Callable:
    """
    Apply a Support Vector Machine (SVM) to an input waveform to
    predict a data cleaning label.

    Note
    ----
    This processor is composed of a factory function that is called
    using the ``init_args`` argument. The input waveform and output
    label are passed using ``args``.


    Parameters
    ----------
    svm_file
       The name of the file with the trained SVM ``svm_p*_r*_T***Z.sav``


    YAML Configuration Example
    --------------------------
    .. code-block:: yaml

      svm_label:
        function: svm_predict
        module: dspeed.processors
        args:
          - dwt_norm
          - svm_label
        unit: ""
        prereqs:
          - dwt_norm
        init_args:
          - "'svm_p*_r*_T***Z.sav'"
    """

    if svm_file == 0:
        svm = None
    else:
        with open(svm_file, "rb") as f:
            svm = pickle.load(f)

    def svm_proc(w_in):
        if svm is None or np.isnan(w_in).any():
            return np.nan

        if w_in.ndim == 1:
            return svm.predict(w_in.reshape(1, -1))
        else:
            return svm.predict(w_in)

    return GUFuncWrapper(
        svm_proc,
        name=svm_file if isinstance(svm_file, str) else "svm_null",
        signature="(n)->()",
        types="f->f",
    )
