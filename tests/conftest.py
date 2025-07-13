import copy
import os
import shutil
import uuid
from getpass import getuser
from inspect import unwrap
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pytest
from legendtestdata import LegendTestData
from lgdo.lh5 import read

from dspeed.utils import GUFuncWrapper

config_dir = Path(__file__).parent / "dsp" / "configs"
_tmptestdir = os.path.join(
    gettempdir(), f"dspeed-tests-{getuser()}-{str(uuid.uuid4())}"
)


@pytest.fixture(scope="session")
def tmptestdir():
    os.mkdir(_tmptestdir)
    return _tmptestdir


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        shutil.rmtree(_tmptestdir)


@pytest.fixture(scope="session")
def lgnd_test_data():
    ldata = LegendTestData()
    ldata.checkout("69aaeee")
    return ldata


@pytest.fixture(scope="session")
def geds_raw_tbl(lgnd_test_data):
    obj = read(
        "/geds/raw",
        lgnd_test_data.get_path("lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5"),
        n_rows=10,
    )
    return obj


@pytest.fixture(scope="session")
def spms_raw_tbl(lgnd_test_data):
    obj = read(
        "/ch0/raw",
        lgnd_test_data.get_path("lh5/L200-comm-20211130-phy-spms.lh5"),
        n_rows=10,
    )
    return obj


@pytest.fixture(scope="session")
def compare_numba_vs_python():
    def numba_vs_python(func, *inputs, signature_override=None):
        """
        Function for testing that the numba and python versions of a
        function are equal.

        Parameters
        ----------
        func
            The Numba-wrapped function to be tested
        *inputs
            The various inputs to be passed to the function to be
            tested.
        signature_override
            Function shape signature to use in the special case that
            outputs are sent as inputs (due to need for new output shape)
            but at least one output is a scalar. In this case, scalars
            will be passed as read-only. To remedy this, rewrite the
            signature with a `->`. For example, say we have a function
            with signature `(n),(m),()` with in output and two outputs.
            You would pass `(n)->(m),()` for the signature override.
            I wish there was another way, but this is how it has to be.

        Returns
        -------
        func_output
            The output of the function to be used in a unit test.

        """

        if signature_override:
            sig = signature_override
            copy_out = False
        elif func.signature:
            sig = func.signature
            copy_out = False
        else:
            # function was implemented with @vectorize
            sig = ",".join(["()"] * func.nin) + "->" + ",".join(["()"] * func.nout)
            copy_out = True

        if len(inputs) == func.nargs:
            # outputs passed as inputs; required when size of output
            # differs from size of input. In this case apply in-place
            # and check/return all args

            # numba outputs
            func(*inputs)
            outputs_numba = [copy.deepcopy(arg) for arg in inputs]

            # unwrapped python outputs
            func_unwrapped = GUFuncWrapper(
                unwrap(func),
                sig,
                "".join([np.array(ar).dtype.char for ar in outputs_numba]),
                copy_out=False,
            )
            func_unwrapped(*inputs)
            outputs_python = [copy.deepcopy(arg) for arg in inputs]

        elif len(inputs) == func.nin:
            # outputs not passed as inputs. Copy and return only outputs

            # numba outputs
            outputs_numba = func(*inputs)
            if func.nout == 1:
                outputs_numba = [outputs_numba]

            # unwrapped python outputs
            types = (
                "".join([np.array(ar).dtype.char for ar in inputs])
                + "->"
                + "".join([np.array(ar).dtype.char for ar in outputs_numba])
            )
            func_unwrapped = GUFuncWrapper(unwrap(func), sig, types, copy_out=copy_out)

            # now outputs must be passed as args. Scalars must be
            # converted to rank 1 arrays
            outputs_python = []
            scalars = []
            for i, ar in enumerate(outputs_numba):
                if len(ar.shape) == 0:
                    outputs_python.append(np.zeros_like(ar).reshape(1))
                    scalars.append(i)
                else:
                    outputs_python.append(np.zeros_like(ar))

            func_unwrapped(*inputs, *outputs_python)

            # convert rank 1 arrays back to scalars
            for i in scalars:
                outputs_python[i] = outputs_python[i][0]
        else:
            raise ValueError("Must pass either all inputs or all inputs and outputs")

        # assert that numba and python are the same up to floating point
        # precision, setting nans to be equal
        assert all(
            np.allclose(o_nb, o_py, equal_nan=True)
            for o_nb, o_py in zip(outputs_numba, outputs_python)
        )

        # return value for comparison with expected solution
        return outputs_numba if len(outputs_numba) > 1 else outputs_numba[0]

    return numba_vs_python
