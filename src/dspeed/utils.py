from __future__ import annotations

import os
from abc import ABCMeta
from collections.abc import Callable, Collection, Iterator, MutableMapping
from typing import Any

import numpy as np
from numba.np.ufunc import sigparse


class GUFuncWrapper:
    """
    A wrapper class to create a u-func like object from an arbitrary function.
    This class is callable and is intended for use for processors that require
    setup with persistent state information; these processors are generated
    using the "factory" method and typically utilize "init_args"

    Example 1:
    ----------
    .. highlight::python
    .. code-block:: python

        # set up some object 'obj' that has a function we want to call on w_in
        gufunc = GUFuncWrapper(
            lambda w_in: obj.execute(w_in, args...),
            "(n)->()",
            "ff"
        )

    Example 2:
    ----------
    .. highlight::python
    .. code-block:: python

        # fun is a vectorized python function, but we want to use ufunc interface
        gufunc = GUFuncWrapper(
            lambda w_in, a, w_out: fun(w_in, a, out=w_out, ...more kwargs),
            "(n),()->(n)",
            "fff",
            vectorized=True,
            copy_out=False
        )

    """

    def __init__(
        self,
        fun: Callable,
        signature: str,
        types: str | Collection[str],
        name: str | None = None,
        vectorized: bool = False,
        copy_out: bool = True,
        doc_string: str = None,
    ):
        """
        Parameters
        ----------
        fun
            python function to be wrapped
        signature
            gufunction signature (see https://numpy.org/doc/2.1/reference/c-api/generalized-ufuncs.html)
        types
            string of type chars, e.g. fi->f
        name
            name of function. By default use `fun.__name__` (this can be very
            unhelpful, e.g. "<lambda>")
        vectorized
            if False, use np.vectorize to loop over function. Set to True
            if `fun` is already vectorized
        copy_out
            set to False if function does in-place calculation for outputs.
            Cannot be False if vectorized is also False
        doc_string
            manually set doc string. If None, use docstring of fun if it
            exists. Else use this docstring.
        """
        self.__name__ = name if name else fun.__name__
        self.ufunc = fun
        self.signature = signature
        if "->" in signature:
            # numba signature parser can't handle outputs with new dimensions
            self.in_dims, self.out_dims = (
                np.lib._function_base_impl._parse_gufunc_signature(signature)
            )
        else:
            # numpy signature parser can't handle no outputs
            self.in_dims, self.out_dims = sigparse.parse_signature(signature)

        self.nin = len(self.in_dims)
        self.nout = len(self.out_dims)
        self.nargs = self.nin + self.nout
        self.types = [types] if isinstance(types, str) else types
        self.ntypes = len(self.types)
        self.copy_out = copy_out
        self.vectorized = vectorized
        if doc_string:
            self.__doc__ = doc_string
        elif fun.__doc__:
            self.__doc__ = fun.__doc__

    def __call__(self, *args):
        """Call wrapped function with "in place" outputs"""

        assert len(args) == self.nargs

        if self.vectorized and self.copy_out and self.nout > 0:
            ins = args[: self.nin]
            outs = args[-self.nout :]
            rets = self.ufunc(*ins)
            if self.nout == 1:
                rets = [rets]
            for out, ret in zip(outs, rets):
                out[...] = ret
        elif self.vectorized:
            self.ufunc(*args)
        else:
            # Calculate broadcasted shapes based on signature
            args = [
                arg if isinstance(arg, np.ndarray) else np.array(arg) for arg in args
            ]

            broadcast_shape, dim_sizes = (
                np.lib._function_base_impl._parse_input_dimensions(
                    args, self.in_dims + self.out_dims
                )
            )
            # handle special case, if scalars were passed as length 1 arrays
            if len(broadcast_shape) > 0 and broadcast_shape[0] == 1:
                broadcast_shape = broadcast_shape[1:]
            shapes = np.lib._function_base_impl._calculate_shapes(
                broadcast_shape, dim_sizes, self.in_dims + self.out_dims
            )

            # check that dimensions match; broadcast inputs that don't
            for i, (arg, shape) in enumerate(zip(args, shapes)):
                if arg.shape != shape and arg.shape != (1,) + shape:
                    if i >= self.nin:
                        raise ValueError("Outputs are not the right shape")
                    args[i] = np.broadcast_to(arg, shape, subok=True)

            # loop over outer dimensions and call function
            for index in np.ndindex(*broadcast_shape):
                ins = [arg[index] for arg in args[: self.nin]]
                outs = []
                for arg in args[self.nin :]:
                    if len(arg.shape) == len(index):
                        if len(index) == 1:
                            outs += [arg[index[0] : index[0] + 1]]
                        else:
                            outs += [arg[np.s_[index[:-1], index[-1] : index[-1] + 1]]]
                    else:
                        outs += [arg[index]]

                if self.copy_out and self.nout > 0:
                    rets = self.ufunc(*ins)
                    if self.nout == 1:
                        rets = [rets]
                    for out, ret in zip(outs, rets):
                        out[...] = ret
                else:
                    self.ufunc(*ins, *outs)


def dspeed_guvectorize(*args, **kwargs):
    """
    Decorator to create a callable object implementing the gufunc interface.
    See arguments in GUFuncWrapper initializer
    """
    return lambda fun: GUFuncWrapper(fun, *args, **kwargs)


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get environment value as a boolean, returning True for 1, t and true
    (caps-insensitive), and False for any other value and default if undefined.
    """
    val = os.getenv(name)
    if not val:
        return default
    elif val.lower() in ("1", "t", "true"):
        return True
    else:
        return False


class NumbaDefaults(MutableMapping):
    """Bare-bones class to store some Numba default options. Defaults values
    are set from environment variables

    Examples
    --------
    Set all default option values for a processor at once by expanding the
    provided dictionary:

    >>> from numba import guvectorize
    >>> from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.dsp.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", **nb_defaults(cache=False) # def proc(...): ...

    Override global options at runtime:

    >>> from pygama.dsp.utils import numba_defaults
    >>> from pygama.dsp import build_dsp
    >>> # must set options before explicitly importing pygama.dsp.processors!
    >>> numba_defaults.cache = False
    >>> numba_defaults.boundscheck = True
    >>> build_dsp(...) # if not explicit, processors imports happen here
    """

    def __init__(self) -> None:
        self.cache: bool = getenv_bool("DSPEED_CACHE", default=True)
        self.boundscheck: bool = getenv_bool("DSPEED_BOUNDSCHECK")
        self.target: str = os.getenv("DSPEED_TARGET", default="cpu")

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, val: Any) -> None:
        self.__dict__[item] = val

    def __delitem__(self, item: str) -> None:
        del self.__dict__[item]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __call__(self, **kwargs) -> dict:
        mapping = self.__dict__.copy()
        mapping.update(**kwargs)
        return mapping

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)


numba_defaults = NumbaDefaults()
numba_defaults_kwargs = numba_defaults


class ProcChainVarBase(metaclass=ABCMeta):
    r"""Base class.

    :class:`ProcChainVar` implements this class. This base class is used
    by processors that use ProcChainVar in their constructors.
    """

    pass
