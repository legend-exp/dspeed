"""
This module provides routines for setting up and running signal processing
chains on waveform data.
"""

from __future__ import annotations

import ast
import importlib
import itertools as it
import json
import logging
import re
import time
import traceback
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import Any

import lgdo
import numpy as np
from numba import guvectorize
from pint import Quantity, Unit
from yaml import safe_load

from . import processors
from .errors import DSPFatal, ProcessingChainError
from .units import unit_registry as ureg
from .utils import ProcChainVarBase
from .utils import numba_defaults_kwargs as nb_kwargs

log = logging.getLogger("dspeed")

# Filler value for variables to be automatically deduced later
auto = "auto"

# Map from ast interpreter operations to functions to call and format string
ast_ops_dict = {
    ast.Add: (np.add, "{}+{}"),
    ast.Sub: (np.subtract, "{}-{}"),
    ast.Mult: (np.multiply, "{}*{}"),
    ast.Div: (np.divide, "{}/{}"),
    ast.FloorDiv: (np.floor_divide, "{}//{}"),
    ast.USub: (np.negative, "-{}"),
    ast.Eq: (np.equal, "{}=={}"),
    ast.NotEq: (np.not_equal, "{}!={}"),
    ast.Lt: (np.less, "{}<{}"),
    ast.LtE: (np.less_equal, "{}<={}"),
    ast.Gt: (np.greater, "{}>{}"),
    ast.GtE: (np.greater_equal, "{}>={}"),
}


# helper function to tell if an object is found in the unit registry
def is_in_pint(unit):
    return isinstance(unit, (Unit, Quantity)) or (unit and unit in ureg)


@dataclass
class CoordinateGrid:
    """Helper class that describes a system of units, consisting of a period
    and offset.

    `period` is a unitted :class:`pint.Quantity`, `offset` is a scalar in units
    of `period`, a :class:`pint.Unit` or a :class:`ProcChainVar`. In the last
    case, a :class:`ProcChainVar` variable is used to store a different offset
    for each event.
    """

    period: Quantity | Unit | str
    offset: Quantity | ProcChainVar | Real = 0

    def __post_init__(self) -> None:
        # Copy constructor and conversions
        if isinstance(self.period, CoordinateGrid):
            self.offset = self.period.offset
            self.period = self.period.period
        elif isinstance(self.period, ProcChainVar):
            if self.period.grid in (None, auto):
                raise ProcessingChainError(
                    f"{self.period} does not have an assigned coordinate grid"
                )
            self.offset = self.period.offset
            self.period = self.period.period
        elif isinstance(self.period, Collection) and not isinstance(self.period, str):
            self.period, self.offset = self.period

        if isinstance(self.period, str):
            self.period = Quantity(1.0, self.period)
        elif isinstance(self.period, Unit):
            self.period *= 1  # make Quantity

        if isinstance(self.offset, Real):
            self.offset = self.offset * self.period
        assert isinstance(self.period, Quantity) and isinstance(
            self.offset, (Quantity, ProcChainVar)
        )

    def __eq__(self, other: CoordinateGrid) -> bool:
        """True if values are equal; if offset is a variable, compares reference"""
        return self.period == other.period and (
            self.offset is other.offset
            if isinstance(self.offset, ProcChainVar)
            else self.offset == other.offset
        )

    def unit_str(self) -> str:
        string = format(self.period.u, "~")
        if string == "":
            string = str(self.period.u)
        return string

    def get_period(self, unit: str | Unit) -> float:
        if isinstance(unit, str):
            unit = ureg.Quantity(unit)
        return float(self.period / unit)

    def get_offset(self, unit: str | Unit = None) -> float:
        """Get the offset (convert)ed to unit. If `unit` is ``None`` use period."""
        if unit is None:
            unit = self.period
        elif isinstance(unit, str):
            unit = ureg.Quantity(unit)

        if isinstance(self.offset, ProcChainVar):
            return self.offset.get_buffer(CoordinateGrid(unit))
        else:
            return float(self.offset / unit)

    def __str__(self) -> str:
        offset = (
            self.offset.name
            if isinstance(self.offset, ProcChainVar)
            else str(self.offset)
        )
        return f"({str(self.period)},{offset})"


class ProcChainVar(ProcChainVarBase):
    """Helper data class with buffer and information for internal variables in
    :class:`ProcessingChain`.

    Members can be set to ``auto`` to attempt to deduce these when adding this
    variable to a processor for the first time.
    """

    def __init__(
        self,
        proc_chain: ProcessingChain,
        name: str,
        shape: int | tuple[int, ...] = auto,
        dtype: np.dtype = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
        vector_len: str | ProcChainVar = None,
        is_const: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        proc_chain
            :class:`ProcessingChain` that contains this variable.
        name
            Name of variable used to look it up.
        shape
            Shape of variable, without `buffer_len` dimension.
        dtype
            Data type of variable.
        grid
            Coordinate grid associated with variable. This contains the
            period and offset of the variable. For variables where
            is_coord is True, use this to perform unit conversions.
        unit
            Unit associated with variable during I/O.
        is_coord
            If ``True``, variable represents an array index and can be converted
            into a unitted number using grid.
        vector_len
            For VectorOfVector variables, this points to the variable used
            to represent the length of each vector
        is_const
            If ``True``, variable is a constant. Variable will be set before
            executing, and will not be recomputed. Does not have outer
            dimension of size _block_width
        """
        assert isinstance(proc_chain, ProcessingChain) and isinstance(name, str)
        self.proc_chain = proc_chain
        self.name = name

        # ndarray containing data buffer of size block_width x shape
        # list of ndarrays in different coordinate systems if is_coord is true
        self._buffer: list | np.ndarray = None

        self.shape = shape
        self.dtype = dtype
        self.grid = grid
        self.unit = unit
        self.is_coord = is_coord
        self.vector_len = vector_len
        self.is_const = is_const

        log.debug(f"added variable: {self.description()}")

    # Use this to enforce type constraints and perform conversions
    def __setattr__(self, name: str, value: Any) -> None:
        if value is auto:
            pass

        elif name == "shape":
            if hasattr(value, "__iter__"):
                value = tuple(value)
            else:
                value = (int(value),)
            value = tuple(value)
            assert all(isinstance(d, int) for d in value)

        elif name == "dtype" and not isinstance(value, np.dtype):
            value = np.dtype(value)

        elif (
            name == "grid"
            and not isinstance(value, CoordinateGrid)
            and value is not None
        ):
            if isinstance(value, str):
                value = CoordinateGrid(value, 0)
            elif isinstance(value, Collection):
                value = CoordinateGrid(*value)
            else:
                value = CoordinateGrid(value, 0)

        elif name == "unit" and value is not None:
            value = value

        elif name == "is_coord":
            value = bool(value)

        elif name == "vector_len" and value is not None:
            if not isinstance(value, ProcChainVar):
                value = self.proc_chain.get_variable(value)
            value.update_auto(
                shape=(),
                grid=None,
                unit=None,
                is_coord=False,
            )

        super().__setattr__(name, value)

    def _make_buffer(self) -> np.ndarray:
        if self.is_const:
            shape = (1,) + self.shape
        else:
            shape = (self.proc_chain._block_width,) + self.shape
        len = np.prod(shape)
        # Flattened array, with padding to allow memory alignment
        buf = np.zeros(len + 64 // self.dtype.itemsize, dtype=self.dtype)
        # offset to ensure memory alignment
        offset = (64 - buf.ctypes.data) % 64 // self.dtype.itemsize
        return buf[offset : offset + len].reshape(shape)

    def get_buffer(self, unit: str | Unit = None) -> np.ndarray:
        # If buffer needs to be created, do so now
        if self._buffer is None:
            if self.shape is auto:
                raise ProcessingChainError(f"cannot deduce shape of {self.name}")
            if self.dtype is auto:
                raise ProcessingChainError(f"cannot deduce dtype of {self.name}")
            self._buffer = self._make_buffer()

        # if no unit is given, use the native unit/coordinate grid
        if unit is None:
            unit = self.grid if self.is_coord else self.unit
        if not isinstance(unit, CoordinateGrid) and is_in_pint(unit):
            unit = CoordinateGrid(unit)

        if isinstance(self._buffer, np.ndarray):
            if self.is_coord is True:
                if isinstance(self.grid, CoordinateGrid):
                    pass
                elif unit is not None:
                    self.grid = CoordinateGrid(unit)

            if not (isinstance(unit, CoordinateGrid) or is_in_pint(unit)):
                # buffer cannot be converted so return
                return self._buffer
            else:
                # buffer can be converted, so make it a list of buffers
                self._buffer = [(self._buffer, unit)]

        if not isinstance(unit, CoordinateGrid) and not is_in_pint(unit):
            return self._buffer[0][0]

        # check if coordinate conversion has been done already
        for buff, buf_u in self._buffer:
            if buf_u == unit:
                return buff

        # If we get this far, add conversion processor to ProcChain and add new buffer to _buffer
        conversion_manager = UnitConversionManager(self, unit)
        self._buffer.append((conversion_manager.out_buffer, unit))
        self.proc_chain._proc_managers.append(conversion_manager)
        log.debug(f"added conversion: {conversion_manager}")
        return conversion_manager.out_buffer

    @property
    def buffer(self):
        return self.get_buffer()

    @property
    def period(self):
        return self.grid.period if self.grid else None

    @property
    def offset(self):
        return self.grid.offset if self.grid else None

    def description(self) -> str:
        return (
            f"{self.name}(shape: {self.shape}, "
            f"dtype: {self.dtype}, grid: {self.grid}, "
            f"unit: {self.unit}, is_coord: {self.is_coord})"
        )

    def update_auto(
        self,
        shape: int | tuple[int, ...] = auto,
        dtype: np.dtype = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
        period: period = None,
        offset: offset = 0,
        vector_len: str | ProcChainVar = None,
    ) -> None:
        """Update any variables set to ``auto``; leave the others alone. Emit a
        message only if anything was updated.
        """
        updated = False

        # Construct coordinate grid from period/offset if given
        if grid is auto and period is not None:
            if isinstance(offset, str):
                offset = self.get_variable(offset, expr_only=True)
            grid = CoordinateGrid(period, offset)

        if self.shape is auto and shape is not auto:
            self.shape = shape
            updated = True
        if self.dtype is auto and dtype is not auto:
            self.dtype = dtype
            updated = True
        if self.grid is auto and grid is not auto:
            self.grid = grid
            updated = True
        if self.unit is auto and unit is not auto:
            self.unit = unit
            updated = True
        if self.is_coord is auto and is_coord is not auto:
            self.is_coord = is_coord
            updated = True
        if self.vector_len is None and vector_len is not None:
            self.vector_len = vector_len
        if updated:
            log.debug(f"updated variable: {self.description()}")

    def __str__(self) -> str:
        return self.name


class ProcessingChain:
    """A class to efficiently perform a sequence of digital signal processing (DSP) transforms.

    It contains a list of DSP functions and a set of constant values and named
    variables contained in fixed memory locations. When executing the
    :class:`ProcessingChain`, processors will act on the internal memory
    without allocating new memory in the process. Furthermore, the memory is
    allocated in blocks, enabling vectorized processing of many entries at
    once. To set up a :class:`ProcessingChain`, use the following methods:

    - :meth:`.link_input_buffer` bind a named variable to an external NumPy
      array to read data from
    - :meth:`.add_processor` add a dsp function and bind its inputs to a set of
      named variables and constant values
    - :meth:`.link_output_buffer` bind a named variable to an external NumPy
      array to write data into

    When calling these methods, the :class:`ProcessingChain` class will use
    available information to allocate buffers to the correct sizes and data
    types. For this reason, transforms will ideally implement the
    :class:`numpy.ufunc` class, enabling broadcasting of array dimensions. If
    not enough information is available to correctly allocate memory, it can be
    provided through the named variable strings or by calling add_vector or
    add_scalar.
    """

    def __init__(self, block_width: int = 8, buffer_len: int = None) -> None:
        """
        Parameters
        ----------
        block_width
            number of entries to simultaneously process.
        buffer_len
            length of input and output buffers. Should be a multiple of
            `block_width`.
        """
        # Dictionary from name to scratch data buffers as ProcChainVar
        self._vars_dict = {}
        # list of processors with variables they are called on
        self._proc_managers = []
        # lists of I/O managers that handle copying data to/from external memory buffers
        self._input_managers = []
        self._output_managers = []

        self._block_width = block_width
        self._buffer_len = buffer_len

    def add_variable(
        self,
        name: str,
        dtype: np.dtype | str = auto,
        shape: int | tuple[int, ...] = auto,
        grid: CoordinateGrid = auto,
        unit: str | Unit = auto,
        is_coord: bool = auto,
        period: CoordinateGrid.period = None,
        offset: CoordinateGrid.offset = 0,
        vector_len: str | ProcChainVar = None,
    ) -> ProcChainVar:
        """Add a named variable containing a block of values or arrays.

        Parameters
        ----------
        name
            name of variable.
        dtype
            default is ``None``, meaning `dtype` will be deduced later, if
            possible.
        shape
            length or shape tuple of element. Default is ``None``, meaning length
            will be deduced later, if possible.
        grid
            for variable, containing period and offset.
        unit
            unit of variable.
        period
            unit with period of waveform associated with object. Do not use if
            `grid` is provided.
        offset
            unit with offset of waveform associated with object. Requires a
            `period` to be provided.
        is_coord
            if ``True``, transform value based on `period` and `offset`.
        """
        self._validate_name(name, raise_exception=True)
        if name in self._vars_dict:
            raise ProcessingChainError(name + " is already in variable list")

        # Construct coordinate grid from period/offset if given
        if grid is auto and period is not None:
            if isinstance(offset, str):
                offset = self.get_variable(offset, expr_only=True)
            grid = CoordinateGrid(period, offset)

        var = ProcChainVar(
            self,
            name,
            shape=shape,
            dtype=dtype,
            grid=grid,
            unit=unit,
            is_coord=is_coord,
            vector_len=vector_len,
        )
        self._vars_dict[name] = var
        return var

    def set_constant(
        self,
        varname: str,
        val: np.ndarray | Real | Quantity,
        dtype: str | np.dtype = None,
        unit: str | Unit | Quantity = None,
    ) -> ProcChainVar:
        """Make a variable act as a constant and set it to val.

        Parameters
        ----------
        varname
            name of internal variable to set. If it does not exist, create
            it; otherwise, set existing variable to be constant
        val
            value of constant
        dtype
            dtype of constant
        unit
            unit of constant
        """

        param = self.get_variable(varname)
        assert param.is_const or param._buffer is None
        param.is_const = True

        if isinstance(val, Quantity):
            unit = val.u
            val = val.m

        val = np.array(val, dtype=dtype)

        param.update_auto(shape=val.shape, dtype=val.dtype, unit=unit, is_coord=False)
        np.copyto(param.get_buffer(), val, casting="unsafe")
        log.debug(f"set constant: {param.description()} = {val}")
        return param

    def link_input_buffer(
        self, varname: str, buff: np.ndarray | lgdo.LGDO = None
    ) -> np.ndarray | lgdo.LGDO:
        """Link an input buffer to a variable.

        Parameters
        ----------
        varname
            name of internal variable to copy into buffer at the end
            of processor execution. If variable does not yet exist, it will
            be created with a similar shape to the provided buffer.
        buff
            object to use as input buffer. If ``None``, create a new buffer
            with a similar shape to the variable.

        Returns
        -------
        buffer
            `buff` or newly allocated input buffer.
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname, expr_only=True)
        if var is None:
            var = self.add_variable(varname)

        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(
                "Must link an input buffer to a processing chain variable"
            )

        # Create input buffer that will be linked and returned if none exists
        if buff is None:
            dtype = var.get_buffer().dtype

            if var is None:
                raise ProcessingChainError(
                    f"{varname} does not exist and no buffer was provided"
                )
            elif (
                isinstance(var.grid, CoordinateGrid)
                and len(var.shape) == 1
                and not var.is_coord
            ):
                buff = lgdo.WaveformTable(
                    size=self._buffer_len, wf_len=var.shape[0], dtype=dtype
                )
            elif len(var.shape) == 0:
                buff = lgdo.Array(shape=(self._buffer_len), dtype=dtype)
            elif var.vector_len is not None:
                buff = lgdo.VectorOfVectors(
                    shape_guess=(self._buffer_len, *var.shape), dtype=dtype
                )
            elif len(var.shape) > 0:
                buff = lgdo.ArrayOfEqualSizedArrays(
                    shape=(self._buffer_len, *var.shape), dtype=dtype
                )
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape, dtype)

        # Add the buffer to the input buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, lgdo.ArrayOfEqualSizedArrays):
            out_man = LGDOArrayOfEqualSizedArraysIOManager(buff, var)
        elif isinstance(buff, lgdo.VectorOfVectors):
            out_man = LGDOVectorOfVectorsIOManager(buff, var)
        elif isinstance(buff, lgdo.Array):
            out_man = LGDOArrayIOManager(buff, var)
        elif isinstance(buff, lgdo.WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError(
                "Could not link input buffer of unknown type", str(buff)
            )

        log.debug(f"added input buffer: {out_man}")
        self._input_managers.append(out_man)

        return buff

    def link_output_buffer(
        self, varname: str, buff: np.ndarray | lgdo.LGDO = None
    ) -> np.ndarray | lgdo.LGDO:
        """Link an output buffer to a variable.

        Parameters
        ----------
        varname
            name of internal variable to copy into buffer at the end of
            processor execution. If variable does not yet exist, it will be
            created with a similar shape to the provided buffer.
        buff
            object to use as output buffer. If ``None``, create a new buffer
            with a similar shape to the variable.

        Returns
        -------
        buffer
            `buff` or newly allocated output buffer.
        """
        self._validate_name(varname, raise_exception=True)
        var = self.get_variable(varname, expr_only=True)
        if var is None:
            var = self.add_variable(varname)

        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(
                "must link an output buffer to a processing chain variable"
            )

        # Create output buffer that will be linked and returned if none exists
        if buff is None:
            dtype = var.get_buffer().dtype

            if var is None:
                raise ProcessingChainError(
                    varname + " does not exist and no buffer was provided"
                )
            elif (
                isinstance(var.grid, CoordinateGrid)
                and len(var.shape) == 1
                and not var.is_coord
            ):
                buff = lgdo.WaveformTable(
                    size=self._buffer_len, wf_len=var.shape[0], dtype=dtype
                )
            elif len(var.shape) == 0:
                buff = lgdo.Array(shape=(self._buffer_len), dtype=dtype)
            elif var.vector_len is not None:
                buff = lgdo.VectorOfVectors(
                    shape_guess=(self._buffer_len, *var.shape), dtype=dtype
                )
            elif len(var.shape) > 0:
                buff = lgdo.ArrayOfEqualSizedArrays(
                    shape=(self._buffer_len, *var.shape), dtype=dtype
                )
            else:
                buff = np.ndarray((self._buffer_len,) + var.shape, dtype)

        # Add the buffer to the output buffers list
        if isinstance(buff, np.ndarray):
            out_man = NumpyIOManager(buff, var)
        elif isinstance(buff, lgdo.ArrayOfEqualSizedArrays):
            out_man = LGDOArrayOfEqualSizedArraysIOManager(buff, var)
        elif isinstance(buff, lgdo.VectorOfVectors):
            out_man = LGDOVectorOfVectorsIOManager(buff, var)
        elif isinstance(buff, lgdo.Array):
            out_man = LGDOArrayIOManager(buff, var)
        elif isinstance(buff, lgdo.WaveformTable):
            out_man = LGDOWaveformIOManager(buff, var)
        else:
            raise ProcessingChainError(
                "could not link output buffer of unknown type", str(buff)
            )

        log.debug(f"added output buffer: {out_man}")
        self._output_managers.append(out_man)

        return buff

    def add_processor(
        self,
        func: np.ufunc,
        *args,
        signature: str = None,
        types: list[str] = None,
        coord_grid: tuple | str = None,
    ) -> None:
        """Make a list of parameters from `*args`. Replace any strings in the
        list with NumPy objects from `vars_dict`, where able.
        """
        params = []
        kw_params = {}
        for _, param in enumerate(args):
            if isinstance(param, str):
                param = self.get_variable(param)
            if isinstance(param, MutableMapping):
                kw_params.update(param)
            else:
                params.append(param)

        if coord_grid is not None:
            coord_grid = CoordinateGrid(coord_grid)

        proc_man = ProcessorManager(
            self, func, params, kw_params, signature, types, coord_grid
        )
        self._proc_managers.append(proc_man)
        log.debug(f"added processor: {proc_man}")

    def execute(self, start: int = 0, stop: int = None) -> None:
        """Execute the dsp chain on the entire input/output buffers."""
        if stop is None:
            stop = self._buffer_len
        for i in range(start, stop, self._block_width):
            try:
                self._execute_procs(i, min(i + self._block_width, stop))
            except IndexError:
                break

    def get_variable(
        self, expr: str, get_names_only: bool = False, expr_only: bool = False
    ) -> Any:
        r"""Parse string `expr` into a NumPy array or value, using the following
        syntax:

        - numeric values are parsed into ``int``\ s or ``float``\ s
        - units found in the :mod:`pint` package
        - other strings are parsed into variable names. If `get_names_only` is
          ``False``, fetch the internal buffer (creating it as needed). Else,
          return a string of the name
        - if a string is followed by ``(...)``, try parsing into one of the
          following expressions:

          - ``len(expr)``: return the length of the array found with `expr`
          - ``astype(expr, dtype)``: cast `expr` to `dtype`
          - ``round(expr, to_nearest = 1, [dtype])``: return the value found with
              `expr` rounded to the nearest multiple of `to_nearest`
          - ``floor(expr, to_nearest = 1, [dtype])``: return the value found with
              `expr` rounded to last multiple of `to_nearest` smaller
          - ``ceil(expr, to_nearest = 1, [dtype])``: return the value found with
              `expr` rounded to first multiple of `to_nearest` larger
          - ``trunc(expr, to_nearest = 1, [dtype])``: return the value found with
              `expr` rounded to first multiple of `to_nearest` towards zero
          - ``where(condition, a, b, [dtype])``: if `condition` is `True` return the
              value held in `a`, else `b`
          - ``isnan(expr)``: return `True` if `expr` is `NaN`
          - ``isfinite(expr)``: return `True`` if not `NaN` `inf` or `-inf`
          - ``varname(shape, type)``: allocate a new buffer with the
            specified shape and type, using ``varname``. This is used if
            the automatic type and shape deduction for allocating variables
            fails
          - ``loadlh5(file, group)``: load LH5 object held in `group` of lh5
              file. Returned object will be treated as a const.

        - Unary and binary operators :obj:`+`, :obj:`-`, :obj:`*`, :obj:`/`,
          :obj:`//` are available. If a variable name is included in the
          expression, a processor will be added to the
          :class:`ProcessingChain` and a new buffer allocated to store the
          output
        - ``varname[slice]``: return the variable with a slice applied. Slice
          values can be ``float``\ s, and will have round applied to them
        - ``keyword = expr``: return a ``dict`` with a single element
          pointing from keyword to the parsed `expr`. This is used for
          `kwargs`. If `expr_only` is ``True``, raise an exception if we see
          this.
        - ``a if b else c``: see `where`; return value held in `a` if `b`
          is `True`, else `c`

        If `get_names_only` is set to ``True``, do not fetch or allocate new
        arrays, instead return a list of variable names found in the expression.
        """
        names = []
        try:
            stmt = ast.parse(expr).body[0]
            var = self._parse_expr(stmt.value, expr, get_names_only, names)
        except Exception as e:
            raise ProcessingChainError("Could not parse expression:\n  " + expr) from e

        # Check if this is an arg (i.e. expr) or kwarg (i.e. assign)
        if not get_names_only:
            if isinstance(stmt, ast.Expr):
                return var
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                if expr_only:
                    raise ProcessingChainError(
                        "kwarg assignment is not allowed in this context\n  " + expr
                    )
                return {stmt.targets[0].id: var}
            else:
                raise ProcessingChainError("Could not parse expression:\n  " + expr)
        else:
            return names

    def _parse_expr(
        self, node: Any, expr: str, dry_run: bool, var_name_list: list[str]
    ) -> Any:
        """
        Helper function for :meth:`ProcessingChain.get_variable` that
        recursively evaluates the AST tree. Whenever we encounter a variable
        name, add it to `var_name_list` (which should begin as an empty list).
        Only add new variables and processors to the chain if `dry_run` is
        ``True``. Based on `this Stackoverflow
        answer <https://stackoverflow.com/a/9558001>`_.
        """
        if node is None:
            return None

        elif isinstance(node, ast.List):
            npparr = np.array(
                ast.literal_eval(expr[node.col_offset : node.end_col_offset])
            )
            if len(npparr.shape) == 1:
                return npparr
            else:
                raise ProcessingChainError("only 1D arrays are supported: " + expr)

        elif isinstance(node, ast.Constant):
            return node.value

        # look for name in variable dictionary
        elif isinstance(node, ast.Name):
            # check if it is a unit
            if node.id in ureg:
                return ureg(node.id)

            # check if it is a variable
            var_name_list.append(node.id)
            if dry_run:
                return None

            val = self._vars_dict.get(node.id, None)
            if val is None:
                val = self.add_variable(node.id)
            return val

        # define binary operators (+,-,*,/)
        elif isinstance(node, ast.BinOp):
            lhs = self._parse_expr(node.left, expr, dry_run, var_name_list)
            rhs = self._parse_expr(node.right, expr, dry_run, var_name_list)
            if rhs is None or lhs is None:
                return None
            op, op_form = ast_ops_dict[type(node.op)]

            if not (isinstance(lhs, ProcChainVar) or isinstance(rhs, ProcChainVar)):
                ret = op(lhs, rhs)
                if isinstance(ret, Quantity) and ureg.is_compatible_with(
                    ret.u, ureg.dimensionless
                ):
                    ret = ret.to(ureg.dimensionless).magnitude
                return ret

            name = "(" + op_form.format(str(lhs), str(rhs)) + ")"
            if isinstance(lhs, ProcChainVar) and isinstance(rhs, ProcChainVar):
                if is_in_pint(lhs.unit) and is_in_pint(rhs.unit):
                    unit = op(Quantity(lhs.unit), Quantity(rhs.unit)).u
                    if unit == ureg.dimensionless:
                        unit = None
                elif lhs.unit is not None and rhs.unit is not None:
                    if type(node.op) in (ast.Mult, ast.Div, ast.FloorDiv):
                        unit = op_form.format(str(lhs.unit), str(rhs.unit))
                    else:
                        unit = str(lhs.unit)
                elif lhs.unit is not None:
                    unit = lhs.unit
                else:
                    unit = rhs.unit
                # If both vars are coordinates, this is probably not a coord.
                # If one var is a coord, this is probably a coord
                out = ProcChainVar(
                    self,
                    name,
                    grid=None if lhs.is_coord and rhs.is_coord else auto,
                    is_coord=(
                        False if lhs.is_coord is True and rhs.is_coord is True else auto
                    ),
                    unit=unit,
                )
            elif isinstance(lhs, ProcChainVar):
                out = ProcChainVar(
                    self,
                    name,
                    unit=lhs.unit,
                    is_coord=lhs.is_coord,
                )
            else:
                out = ProcChainVar(
                    self,
                    name,
                    unit=rhs.unit,
                    is_coord=rhs.is_coord,
                )

            proc_man = ProcessorManager(self, op, [lhs, rhs, out])
            self._proc_managers.append(proc_man)
            log.debug(f"added processor: {proc_man}")
            return out

        # define unary operators (-)
        elif isinstance(node, ast.UnaryOp):
            operand = self._parse_expr(node.operand, expr, dry_run, var_name_list)
            if operand is None:
                return None
            op, op_form = ast_ops_dict[type(node.op)]
            name = "(" + op_form.format(str(operand)) + ")"

            if isinstance(operand, ProcChainVar):
                out = ProcChainVar(
                    self,
                    name,
                    operand.shape,
                    operand.dtype,
                    operand.grid,
                    operand.unit,
                    operand.is_coord,
                )
                proc_man = ProcessorManager(self, op, [operand, out])
                self._proc_managers.append(proc_man)
                log.debug(f"added processor: {proc_man}")
            else:
                out = op(operand)

            return out

        # define comparison operators (<, <=, >, >=, ==, !=)
        elif isinstance(node, ast.Compare):
            lhs = self._parse_expr(node.left, expr, dry_run, var_name_list)
            if len(node.comparators) != 1:
                raise ProcessingChainError("Compound comparisons are not supported.")
            rhs = self._parse_expr(node.comparators[0], expr, dry_run, var_name_list)
            if rhs is None or lhs is None:
                return None
            op, op_form = ast_ops_dict[type(node.ops[0])]

            if not (isinstance(lhs, ProcChainVar) or isinstance(rhs, ProcChainVar)):
                ret = op(lhs, rhs)
                if isinstance(ret, Quantity) and ureg.is_compatible_with(
                    ret.u, ureg.dimensionless
                ):
                    ret = ret.to(ureg.dimensionless).magnitude
                return ret

            out = ProcChainVar(
                self,
                "(" + op_form.format(str(lhs), str(rhs)) + ")",
            )

            proc_man = ProcessorManager(self, op, [lhs, rhs, out])
            self._proc_managers.append(proc_man)
            log.debug(f"added processor: {proc_man}")
            return out

        elif isinstance(node, ast.Subscript):
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None:
                return None
            if not isinstance(val, ProcChainVar) or not len(val.shape) > 0:
                raise ProcessingChainError("Cannot apply subscript to", node.value)

            def get_index(slice_value):
                ret = self._parse_expr(slice_value, expr, dry_run, var_name_list)
                if ret is None:
                    return ret
                if isinstance(ret, Quantity):
                    ret = float(ret / val.period)
                if isinstance(ret, Real):
                    round_ret = int(round(ret))
                    if abs(ret - round_ret) > 0.0001:
                        log.warning(
                            f"slice value {slice_value} is non-integer. Rounding to {round_ret}"
                        )
                    return round_ret
                return int(ret)

            if isinstance(node.slice, ast.Constant):
                index = get_index(node.slice)
                out_buf = val.buffer[..., index]
                out_name = f"{str(val)}[{index}]"
                out_grid = val.grid if val.is_coord else None

            elif isinstance(node.slice, ast.Slice):
                sl = slice(
                    get_index(node.slice.lower),
                    get_index(node.slice.upper),
                    get_index(node.slice.step),
                )
                out_buf = val.buffer[..., sl]
                out_name = "{}[{}:{}{}]".format(
                    str(val),
                    "" if sl.start is None else str(sl.start),
                    "" if sl.stop is None else str(sl.stop),
                    "" if sl.step is None else ":" + str(sl.step),
                )

                if val.grid is None:
                    out_grid = None
                else:
                    pd = val.period
                    if sl.step is not None:
                        pd *= sl.step

                    off = val.offset
                    if sl.start is not None and sl.start > 0:
                        start = sl.start * val.period
                        if isinstance(off, ProcChainVar):
                            new_off = ProcChainVar(
                                self, name=f"({str(off)}+{str(start)})", is_coord=True
                            )
                            proc_man = ProcessorManager(
                                self, np.add, [off, start, new_off]
                            )
                            self._proc_managers.append(proc_man)
                            log.debug(f"added processor: {proc_man}")
                            off = new_off
                        else:
                            off += start
                    out_grid = CoordinateGrid(pd, off)

            elif isinstance(node.slice, ast.ExtSlice):
                # TODO: implement this...
                raise ProcessingChainError("ExtSlice still isn't implemented...")

            # Create our return variable and set the buffer to the slice
            out = ProcChainVar(
                self,
                out_name,
                shape=out_buf.shape[1:],
                dtype=out_buf.dtype,
                grid=out_grid,
                unit=val.unit,
                is_coord=val.is_coord,
            )
            out._buffer = [(out_buf, val._buffer[0][1])] if out.is_coord else out_buf
            return out

        # a if condition else b
        elif isinstance(node, ast.IfExp):
            condition = self._parse_expr(node.test, expr, dry_run, var_name_list)
            a = self._parse_expr(node.body, expr, dry_run, var_name_list)
            b = self._parse_expr(node.orelse, expr, dry_run, var_name_list)
            return self._where(condition, a, b)

        # for name.attribute
        elif isinstance(node, ast.Attribute):
            # If we are looking for an attribute of a module (e.g. np.pi)
            module = expr[node.value.col_offset : node.value.end_col_offset]
            if module in self.module_list:
                mod = self.module_list[module]
                attr = getattr(mod, node.attr)
                if not isinstance(attr, Real):
                    raise ProcessingChainError(
                        f"Attribute {node.attr} from {node.value} is not"
                        f"an int or float..."
                    )
                return attr

            # Otherwise this is probably a ProcChainVar
            val = self._parse_expr(node.value, expr, dry_run, var_name_list)
            if val is None:
                return None
            return getattr(val, node.attr)

        # for func(args, kwargs)
        elif isinstance(node, ast.Call):
            func = self.func_list.get(node.func.id, None)
            args = [
                self._parse_expr(arg, expr, dry_run, var_name_list) for arg in node.args
            ]
            kwargs = {
                kwarg.arg: self._parse_expr(kwarg.value, expr, dry_run, var_name_list)
                for kwarg in node.keywords
            }
            if func is not None:
                return func(self, *args, **kwargs) if not dry_run else None
            elif self._validate_name(node.func.id):
                var_name = node.func.id
                var_name_list.append(var_name)
                if var_name in self._vars_dict:
                    var = self._vars_dict[var_name]
                    var.update_auto(*args, **kwargs)
                    return self._vars_dict[var_name]
                elif not dry_run:
                    return self.add_variable(var_name, *args, **kwargs)
                else:
                    return None

            else:
                raise ProcessingChainError(
                    f"do not recognize call to {func} with arguments "
                    f"{[str(arg.__dict__) for arg in node.args]}"
                )

        raise ProcessingChainError(f"cannot parse AST nodes of type {node.__dict__}")

    def _validate_name(self, name: str, raise_exception: bool = False) -> bool:
        """Check that name is alphanumeric, and not an already used keyword"""
        isgood = (
            re.match(r"\A\w+$", name)
            and name not in self.func_list
            and name not in ureg
            and name not in self.module_list
        )
        if raise_exception and not isgood:
            raise ProcessingChainError(f"{name} is not a valid variable name")
        return isgood

    def _execute_procs(self, begin: int, end: int) -> str:
        """Copy from input buffers to variables, call all the processors on
        their paired arg tuples, copy from variables to list of output buffers.
        """
        # Copy input buffers into proc chain buffers
        for in_man in self._input_managers:
            in_man.read(begin, end)

        # Loop through processors and run each one
        for proc_man in self._proc_managers:
            try:
                proc_man.execute()
            except DSPFatal as e:
                e.processor = str(proc_man)
                e.wf_range = (begin, end)
                raise e

        # copy from processing chain buffers into output buffers
        for out_man in self._output_managers:
            out_man.write(begin, end)

    def __str__(self) -> str:
        return (
            "Input variables:\n  "
            + "\n  ".join([str(in_man) for in_man in self._input_managers])
            + "\nProcessors:\n  "
            + "\n  ".join([str(proc_man) for proc_man in self._proc_managers])
            + "\nOutput variables:\n  "
            + "\n  ".join([str(out_man) for out_man in self._output_managers])
        )

    # Define functions that can be parsed by get_variable
    # Get length of ProcChainVar
    def _length(self, var: ProcChainVar) -> int:
        if var is None:
            return None
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(f"cannot call len() on {var}")
        if not len(var.buffer.shape) == 2:
            raise ProcessingChainError(f"{var} has wrong number of dims")
        if var.vector_len is None:
            return var.buffer.shape[1]
        else:
            return var.vector_len

    def get_timing(self) -> dict[str, float]:
        """Get the timing of each processor in the processing chain."""
        return {str(proc): proc.time_total for proc in self._proc_managers}

    # round value
    def _round(
        self,
        var: ProcChainVar | Quantity,
        to_nearest: Real | Unit | Quantity | CoordinateGrid = 1,
        dtype: str = None,
        mode: str = "round",
    ) -> float | Quantity | ProcChainVar:
        """Round a variable or value to nearest multiple of `to_nearest`.
        If var is a ProcChainVar, and to_nearest is a Unit or Quantity, return
        a new ProcChainVar with a period of to_nearest, and the underlying
        values and offset rounded. If var is a ProcChainVar and to_nearest
        is an int or a float, keep the unit and just round the underlying
        value. Can change mode to "floor", "ceil", or "trunc"

        Example usage:
        round(tp_0, wf.grid) - convert tp_0 to nearest array index of wf
        round(5*us, wf.period) - 5 us in wf clock ticks
        """
        if mode == "round":
            fun = processors.round_to_nearest
        elif mode == "floor":
            fun = processors.floor_to_nearest
        elif mode == "ceil":
            fun = processors.ceil_to_nearest
        elif mode == "trunc":
            fun = processors.trunc_to_nearest
        else:
            raise ProcessingChainError("Mode must be round, floor, ceil or trunc")

        if var is None:
            return None
        if not isinstance(var, ProcChainVar):
            if isinstance(var, Quantity):
                return fun(float(var / to_nearest.u), to_nearest.m) * to_nearest.u
            else:
                return fun(var, to_nearest)
        else:
            name = f"{mode}({var}, {to_nearest})"
            dtype = np.dtype(dtype) if dtype is not None else var.dtype
            if var.is_coord:
                if isinstance(to_nearest, Real):
                    grid = CoordinateGrid(var.grid.period * to_nearest, var.grid.offset)
                elif isinstance(to_nearest, (Unit, Quantity)):
                    grid = CoordinateGrid(to_nearest, var.grid.offset)
                else:
                    grid = to_nearest

                out = ProcChainVar(
                    self,
                    name,
                    var.shape,
                    dtype,
                    grid,
                    var.unit,
                    var.is_coord,
                )
                conversion_manager = UnitConversionManager(var, grid, mode=mode)
                out._buffer = conversion_manager.out_buffer
                self._proc_managers.append(conversion_manager)
                log.debug(f"added conversion: {conversion_manager}")
            else:
                out = ProcChainVar(
                    self,
                    name,
                    var.shape,
                    dtype,
                    var.grid,
                    var.unit,
                    var.is_coord,
                )

                self.add_processor(fun, var, to_nearest, out)

            return out

    # type cast variable
    def _astype(self, var: ProcChainVar, dtype: str) -> ProcChainVar:
        dtype = np.dtype(dtype)
        if var is None:
            return None
        if not isinstance(var, ProcChainVar):
            raise ProcessingChainError(f"cannot call astype() on {var}")
        else:
            name = f"{var}.astype(`{dtype.char}`)"
            out = ProcChainVar(
                self,
                name,
                var.shape,
                dtype,
                var.grid,
                var.unit,
                var.is_coord,
            )
            proc_man = ProcessorManager(
                self,
                np.copyto,
                [out, var],
                kw_params={"casting": "'unsafe'"},
                signature="(),(),()",
                types=f"{dtype.char}{var.dtype.char}",
            )
            self._proc_managers.append(proc_man)
            log.debug(f"added processor: {proc_man}")
            return out

    def _isnan(self, var: ProcChainVar | Real | None):
        """Is value NaN"""
        if var is None:
            return None
        elif not isinstance(var, ProcChainVar):
            return np.isnan(var)
        else:
            out = ProcChainVar(
                self,
                f"isnan({var})",
                var.shape,
                "bool",
                var.grid,
                var.unit,
                var.is_coord,
            )
            proc_man = ProcessorManager(self, np.isnan, [var, out])
            self._proc_managers.append(proc_man)
            log.debug(f"added processor: {proc_man}")
            return out

    def _isfinite(self, var: ProcChainVar | Real | None):
        """Is value finite (i.e. not NaN or infinite)"""
        if var is None:
            return None
        elif not isinstance(var, ProcChainVar):
            return np.isfinite(var)
        else:
            out = ProcChainVar(
                self,
                f"isfinite({var})",
                var.shape,
                "bool",
                var.grid,
                var.unit,
                var.is_coord,
            )
            proc_man = ProcessorManager(self, np.isfinite, [var, out])
            self._proc_managers.append(proc_man)
            log.debug(f"added processor: {proc_man}")
            return out

    # choose a or b
    def _where(
        self,
        condition: ProcChainVar,
        a: ProcChainVar | Real | Quantity,
        b: ProcChainVar | Real | Quantity,
        dtype: str = auto,
    ) -> ProcChainVar:
        """Select value from ``a`` or ``b`` depending on if ``condition`` is ``True`` or ``False``. Used
        for the ``where`` function or ``a if b else c`` pattern."""

        if condition is None:
            return None

        if not (isinstance(condition, ProcChainVar) and condition.dtype == "?"):
            raise ProcessingChainError(f"{condition} must be a boolean variable")

        name = f"where({condition}, {a}, {b})"
        if isinstance(a, ProcChainVar) and isinstance(b, ProcChainVar):
            if a.period != b.period:
                raise ProcessingChainError(
                    f"Cannot select between {a} and {b} with different periods"
                )
            if a.is_coord != b.is_coord:
                raise ProcessingChainError(
                    f"Cannot select between {a} and {b} with different is_coord"
                )
            is_coord = a.is_coord

            if a.offset == b.offset:
                grid = a.grid
            else:
                grid = CoordinateGrid(
                    a.period,
                    self._where(condition, a.offset, b.offset),
                )

            unit_a = Unit(a.unit) if is_in_pint(a.unit) else a.unit
            unit_b = Unit(b.unit) if is_in_pint(b.unit) else b.unit
            if unit_a == unit_b or not unit_b:
                unit = unit_a
            elif not unit_a:
                unit = unit_b
            else:
                raise ProcessingChainError(f"{a} and {b} do not have compatible units")

        elif isinstance(a, ProcChainVar) or isinstance(b, ProcChainVar):
            if isinstance(a, ProcChainVar):
                var = a
                const = b
            else:
                var = b
                const = a

            grid = var.grid
            is_coord = var.is_coord

            if not var.unit:
                unit = None
            elif not isinstance(const, Quantity):
                unit = var.unit
            elif is_in_pint(var.unit):
                unit = var.period if is_coord else Quantity(1, var.unit)
                if isinstance(a, ProcChainVar):
                    b = float(const / (1 * unit))
                else:
                    a = float(const / (1 * unit))
            else:
                raise ProcessingChainError(f"{a} and {b} do not have compatible units")

        else:
            grid = None
            is_coord = False
            if isinstance(a, Quantity) and isinstance(b, Quantity):
                unit = a.u
                a = a.m
                b = float(b / (1 * unit))
            elif isinstance(a, Quantity):
                unit = a.u
                a = a.m
            elif isinstance(b, Quantity):
                unit = b.u
                b = b.m
            else:
                unit = None

        out = ProcChainVar(
            self,
            name,
            auto,
            dtype,
            grid,
            unit,
            is_coord,
        )
        proc_man = ProcessorManager(self, processors.where, [condition, a, b, out])
        self._proc_managers.append(proc_man)
        log.debug(f"added processor: {proc_man}")
        return out

    def _loadlh5(self, path_to_file, path_in_file: str) -> np.array:
        """
        Load data from an LH5 file.

        Args:
            path_to_file (str): The path to the LH5 file.
            path_in_file (str): The path to the data within the LH5 file.

        Returns:
            list: The loaded data.
        """

        from lgdo import lh5

        try:
            loaded_data = lh5.read(path_in_file, path_to_file)
            if isinstance(loaded_data, lgdo.types.Scalar):
                loaded_data = loaded_data.value
            else:
                loaded_data = loaded_data.nda
        except ValueError:
            raise ProcessingChainError(f"LH5 file not found: {path_to_file}")

        return loaded_data

    # dict of functions that can be parsed by get_variable
    func_list = {
        "len": _length,
        "isfinite": _isfinite,
        "isnan": _isnan,
        "round": partial(_round, mode="round"),
        "floor": partial(_round, mode="floor"),
        "ceil": partial(_round, mode="ceil"),
        "trunc": partial(_round, mode="trunc"),
        "astype": _astype,
        "where": _where,
        "loadlh5": _loadlh5,
    }
    module_list = {"np": np, "numpy": np}


class ProcessorManager:
    """The class that calls processors and makes sure variables are compatible."""

    @dataclass
    class DimInfo:
        length: int  # length of arrays in this dimension
        grid: CoordinateGrid  # period and offset of arrays in this dimension

    def __init__(
        self,
        proc_chain: ProcessingChain,
        func: np.ufunc,
        params: list[str],
        kw_params: dict = None,
        signature: str = None,
        types: list[str] = None,
        grid: CoordinateGrid = None,
    ) -> None:
        assert (
            isinstance(proc_chain, ProcessingChain)
            and callable(func)
            and isinstance(params, Collection)
        )

        if kw_params is None:
            kw_params = {}

        # reference back to our processing chain
        self.proc_chain = proc_chain
        # callable function used to process data
        self.processor = func
        # list of parameters prior to converting to internal representation
        self.params = params
        # dict of keyword parameters prior to converting to internal rep
        self.kw_params = kw_params
        # list of raw values and buffers from params; we will fill this soon
        self.args = []
        # dict of kws -> raw values and buffers from params; we will fill this soon
        self.kwargs = {}
        # store time taken by processor
        self.time_total = 0

        # Get the signature and list of valid types for the function
        self.signature = func.signature if signature is None else signature
        if self.signature is None:
            self.signature = (
                ",".join(["()"] * func.nin) + "->" + ",".join(["()"] * func.nout)
            )

        # Get list of allowed type signatures
        if types is None:
            types = func.types.copy()
        if types is None:
            raise ProcessingChainError(
                f"""could not find a type signature list
                                           for {func.__name__}. Please supply a
                                           valid list of types."""
            )
        if isinstance(types, str) or not isinstance(types, Collection):
            types = [types]
        found_types = [typestr.replace("->", "") for typestr in types]

        # Make sure arrays obey the broadcasting rules, and make a dictionary
        # of the correct dimensions and unit system
        dims_list = re.findall(r"\((.*?)\)", self.signature)

        if not len(dims_list) == len(params) + len(kw_params):
            raise ProcessingChainError(
                f"expected {len(dims_list)} arguments from signature "
                f"{self.signature}; found "
                f"{len(params)+len(kw_params)}: ({', '.join([str(par) for par in params])})"
            )

        dims_dict = {}  # map from dim name -> DimInfo
        outerdims = []  # list of DimInfo

        for ipar, (dims, param) in enumerate(
            zip(dims_list, it.chain(self.params, self.kw_params.values()))
        ):
            if not isinstance(param, (ProcChainVar, np.ndarray)):
                continue

            # find type signatures that match type of array
            if param.dtype is not auto:
                arr_type = param.dtype.char
                found_types = [
                    type_sig
                    for type_sig in found_types
                    if np.can_cast(arr_type, type_sig[ipar])
                ]

            # fill out dimensions from dim signature and check if it works
            if param.shape is auto:
                continue
            fun_dims = [od for od in outerdims] + [
                d.strip() for d in dims.split(",") if d
            ]
            arr_dims = list(param.shape)
            if (
                isinstance(param, ProcChainVar)
                and param.grid is not auto
                and not param.is_coord
            ):
                arr_grid = param.grid
            else:
                arr_grid = None
            if not grid:
                grid = arr_grid

            # check if arr_dims can be broadcast to match fun_dims
            for i in range(max(len(fun_dims), len(arr_dims))):
                fd = fun_dims[-i - 1] if i < len(fun_dims) else None
                ad = (
                    arr_dims[-i - 1]
                    if i < len(arr_dims)
                    else self.proc_chain._block_width if i == len(arr_dims) else None
                )

                if isinstance(fd, str):
                    if fd in dims_dict:
                        this_dim = dims_dict[fd]
                        if not ad or this_dim.length != ad:
                            raise ProcessingChainError(
                                f"failed to broadcast array dimensions for "
                                f"{func.__name__}. Could not find consistent value "
                                f"for dimension {fd}"
                            )
                        if not this_dim.grid:
                            dims_dict[fd].grid = arr_grid
                        elif arr_grid and arr_grid != this_dim.grid:
                            log.debug(
                                f"arrays of dimension {fd} for "
                                f"{func.__name__} do not have consistent period "
                                f"and offset!"
                            )
                    else:
                        dims_dict[fd] = self.DimInfo(ad, arr_grid)

                elif not fd:
                    # if we ran out of function dimensions, add a new outer dim
                    outerdims.insert(0, self.DimInfo(ad, arr_grid))

                elif not ad:
                    continue

                elif fd.length != ad:
                    # If dimensions disagree, either insert a broadcasted array dimension or raise an exception
                    if len(fun_dims) > len(arr_dims):
                        arr_dims.insert(len(arr_dims) - i, 1)
                    elif len(fun_dims) < len(arr_dims):
                        outerdims.insert(len(fun_dims) - i, self.DimInfo(ad, arr_grid))
                        fun_dims.insert(len(fun_dims) - i, ad)
                    else:
                        raise ProcessingChainError(
                            f"failed to broadcast array dimensions for "
                            f"{func.__name__}. Input arrays do not have "
                            f"consistent outer dimensions.  Require: "
                            f"{tuple(dim.length for dim in outerdims+fun_dims)}; "
                            f"found {tuple(arr_dims)} for {param}"
                        )
                elif not fd.grid:
                    outerdims[len(fun_dims) - 1 - i].grid = arr_grid

                elif arr_grid and fd.grid != arr_grid:
                    log.debug(
                        f"arrays of dimension {fd} for {func.__name__} "
                        f"do not have consistent period and offset!"
                    )

                arr_grid = None  # this is only used for inner most dim

        # Get the type signature we are using
        if not found_types:
            for param in it.chain(self.params, self.kw_params.values()):
                if not isinstance(param, ProcChainVar):
                    continue
            raise ProcessingChainError(
                f"could not find a type signature matching the types of the "
                f"variables given for {self} (types: {types})"
            )
        # Use the first types in the list that all our types can be cast to
        self.types = [np.dtype(t) for t in found_types[0]]

        # If we haven't identified a coordinate grid from WFs, try from coords
        if not grid:
            for param in it.chain(self.params, self.kw_params.values()):
                if isinstance(param, ProcChainVar) and param.is_coord is True:
                    grid = param.grid
                    break

        # Finish setting up of input parameters for function
        # Iterate through args and then kwargs
        # Reshape variable arrays to add broadcast dimensions
        # Allocate new arrays as needed
        # Convert coords to right system of units as needed
        for _, ((arg_name, param), dims, dtype) in enumerate(
            zip(
                it.chain(zip(it.repeat(None), self.params), self.kw_params.items()),
                dims_list,
                self.types,
            )
        ):
            dim_list = outerdims.copy()
            for d in dims.split(","):
                d = d.strip()
                if not d:
                    continue
                if d not in dims_dict:
                    # If it is an array lets get the length
                    if isinstance(param, np.ndarray):
                        dims_dict[d] = self.DimInfo(len(param), None)
                    else:
                        raise ProcessingChainError(
                            f"could not deduce dimension {d} for {param}"
                        )
                dim_list.append(dims_dict[d])
            shape = tuple(d.length for d in dim_list)
            this_grid = dim_list[-1].grid if dim_list else None

            if isinstance(param, ProcChainVar):
                # Deduce any automated descriptions of parameter
                unit = None
                is_coord = False
                if param.is_coord is True and grid is not None:
                    unit = str(grid.period.u)
                    this_grid = grid
                elif (
                    is_in_pint(param.unit)
                    and grid is not None
                    and ureg.is_compatible_with(grid.period, param.unit)
                ):
                    is_coord = True
                    this_grid = grid

                param.update_auto(
                    shape=shape,
                    dtype=np.dtype(dtype),
                    grid=this_grid,
                    unit=unit,
                    is_coord=is_coord,
                )

                # reshape just in case there are some missing dimensions
                arshape = list(param.buffer.shape)
                for idim in range(-1, -1 - len(shape), -1):
                    if len(arshape) < -idim or arshape[idim] != shape[idim]:
                        arshape.insert(len(arshape) + idim + 1, 1)
                param = param.get_buffer(grid if param.is_coord else None).reshape(
                    arshape
                )

            elif isinstance(param, str):
                # Convert string into integer buffer if appropriate
                if np.issubdtype(dtype, np.integer):
                    try:
                        param = np.frombuffer(param.encode("ascii"), dtype).reshape(
                            shape
                        )
                    except ValueError:
                        raise ProcessingChainError(
                            f"could not convert string '{param}' into"
                            f"byte-array of type {dtype} and shape {shape}"
                        )

            elif param is not None:
                # Convert scalar to right type, including units
                if isinstance(param, (Quantity, Unit)):
                    if ureg.is_compatible_with(ureg.dimensionless, param):
                        param = param.to(ureg.dimensionless).magnitude
                    elif not isinstance(grid, CoordinateGrid):
                        raise ProcessingChainError(
                            f"could not find valid conversion for {param}; "
                            f"CoordinateGrid is {grid}"
                        )
                    else:
                        # This lets us convert powers of unit
                        pi = ureg.pi_theorem({0: grid.period, 1: param})
                        if not pi:
                            raise ProcessingChainError(
                                f"could not find valid conversion for {param}; "
                                f"CoordinateGrid is {grid}"
                            )
                        param = param * grid.period ** (pi[0][0] / pi[0][1])
                        param = param.to(ureg.dimensionless).magnitude
                if np.issubdtype(dtype, np.integer):
                    param = dtype.type(np.round(param))
                else:
                    param = dtype.type(param)

            if arg_name is None:
                self.args.append(param)
            else:
                self.kwargs[arg_name] = param

    def execute(self) -> None:
        start = time.time()
        try:
            self.processor(*self.args, **self.kwargs)
        except Exception as e:
            log.error(f"Error processing {str(self)}: {e}")
            traceback.print_exc()
            raise e
        self.time_total += time.time() - start

    def __str__(self) -> str:
        return (
            self.processor.__name__
            + "("
            + ", ".join(
                [str(par) for par in self.params]
                + [f"{k}={str(v)}" for k, v in self.kw_params.items()]
            )
            + ")"
        )


class UnitConversionManager(ProcessorManager):
    """A special processor manager for handling converting variables between unit systems."""

    def __init__(
        self,
        var: ProcChainVar,
        unit: str | Unit | Quantity | CoordinateGrid,
        mode=None,
    ) -> None:
        # reference back to our processing chain
        self.proc_chain = var.proc_chain
        # callable function used to process data
        from .processors.unit_conversion import (
            convert,
            convert_ceil,
            convert_floor,
            convert_int,
            convert_round,
            convert_trunc,
        )

        if mode == "round":
            self.processor = convert_round
        elif mode == "floor":
            self.processor = convert_floor
        elif mode == "ceil":
            self.processor = convert_ceil
        elif mode == "trunc":
            self.processor = convert_trunc
        elif mode is not None:
            raise ProcessingChainError("Mode must be round, floor, ceil or trunc")
        elif issubclass(var.dtype.type, np.floating):
            self.processor = convert
        else:
            self.processor = convert_int

        to_offset = 0
        if isinstance(unit, CoordinateGrid):
            to_offset = unit.get_offset()
            unit = unit.period

        if isinstance(var._buffer, list):
            from_buffer, from_unit = var._buffer[0]
        else:
            from_buffer = var._buffer
            from_unit = var.unit
            if isinstance(from_unit, str) and from_unit in ureg:
                from_unit = ureg.Quantity(from_unit)

        # list of parameters prior to converting to internal representation
        self.params = [var]
        self.kw_params = {"from": from_unit, "to": unit}

        if isinstance(from_unit, CoordinateGrid):
            ratio = from_unit.get_period(unit)
            from_offset = from_unit.get_offset()
        elif isinstance(from_unit, (Unit, Quantity)):
            if isinstance(unit, str):
                unit = ureg.Quantity(unit)
            ratio = float(from_unit / unit)
            from_offset = 0
        else:
            ratio = 1 / unit
            from_offset = 0

        # Make sure broadcasting will work correctly
        if isinstance(from_offset, np.ndarray):
            from_offset = from_offset.reshape(
                from_offset.shape[0],
                *[1] * (len(from_buffer.shape) - len(from_offset.shape)),
            )
        if isinstance(to_offset, np.ndarray):
            to_offset = to_offset.reshape(
                to_offset.shape[0],
                *[1] * (len(from_buffer.shape) - len(to_offset.shape)),
            )

        self.out_buffer = np.zeros_like(from_buffer, dtype=var.dtype)
        self.args = [
            from_buffer,
            from_offset,
            to_offset,
            ratio,
            self.out_buffer,
        ]
        self.kwargs = {}
        self.time_total = 0


class IOManager(metaclass=ABCMeta):
    r"""Base class.

    :class:`IOManager`\ s will be associated with a type of input/output
    buffer, and must define a read and write for each one. ``__init__()``
    methods should update variable with any information from buffer, and check
    that buffer and variable are compatible.
    """

    @abstractmethod
    def read(self, start: int, end: int) -> None:
        pass

    @abstractmethod
    def write(self, start: int, end: int) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


# Ok, this one's not LGDO
class NumpyIOManager(IOManager):
    r""":class:`IOManager` for buffers that are :class:`numpy.ndarray`\ s."""

    def __init__(self, io_buf: np.ndarray, var: ProcChainVar) -> None:
        assert isinstance(io_buf, np.ndarray) and isinstance(var, ProcChainVar)

        var.update_auto(dtype=io_buf.dtype, shape=io_buf.shape[1:])

        if var.shape != io_buf.shape[1:] or var.dtype != io_buf.dtype:
            raise ProcessingChainError(
                f"numpy.array<{self.io_buf.shape}>({{{self.io_buf.dtype}}}@{self.io_buf.data}) "
                "is not compatible with variable {self.var}"
            )

        self.io_buf = io_buf
        self.var = var
        self.raw_var = var.buffer

    def read(self, start: int, end: int) -> None:
        np.copyto(
            self.raw_var[0 : end - start, ...], self.io_buf[start:end, ...], "unsafe"
        )

    def write(self, start: int, end: int) -> None:
        np.copyto(
            self.io_buf[start:end, ...], self.raw_var[0 : end - start, ...], "unsafe"
        )

    def __str__(self) -> str:
        return (
            f"{self.var} linked to numpy.array(shape={self.io_buf.shape}, "
            f"dtype={self.io_buf.dtype})"
        )


class LGDOArrayIOManager(IOManager):
    r"""IO Manager for buffers that are :class:`lgdo.Array`\ s."""

    def __init__(self, io_array: lgdo.Array, var: ProcChainVar) -> None:
        assert isinstance(io_array, lgdo.Array) and isinstance(var, ProcChainVar)

        unit = io_array.attrs.get("units", None)
        var.update_auto(dtype=io_array.dtype, shape=io_array.nda.shape[1:], unit=unit)

        if isinstance(var.unit, (CoordinateGrid, Quantity, Unit)):
            if isinstance(var.unit, CoordinateGrid):
                var_u = var.unit.period.u
            elif isinstance(var.unit, Quantity):
                var_u = var.unit.u
            else:
                var_u = var.unit

            if unit is None:
                unit = var_u
            elif ureg.is_compatible_with(var_u, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError(
                    f"LGDO array and variable {var} have incompatible units "
                    f"({var_u} and {unit})"
                )
        elif isinstance(var.unit, str) and unit is None:
            unit = var.unit

        if "units" not in io_array.attrs and unit is not None:
            io_array.attrs["units"] = str(unit)

        self.io_array = io_array
        self.var = var
        self.raw_var = var.get_buffer(unit)

        if (
            self.var.shape != self.io_array.nda.shape[1:]
            or self.raw_var.dtype != self.io_array.dtype
        ):
            raise ProcessingChainError(
                f"LGDO object "
                f"{self.io_array.form_datatype()} is "
                f"incompatible with {str(self.var)}"
            )

    def read(self, start: int, end: int) -> None:
        if start >= len(self.io_array):
            raise IndexError
        end = min(end, len(self.io_array))
        self.raw_var[0 : end - start, ...] = self.io_array[start:end, ...]

    def write(self, start: int, end: int) -> None:
        self.io_array.resize(end)
        if self.var.is_const:
            self.io_array[start:end, ...] = self.raw_var[...]
        else:
            self.io_array[start:end, ...] = self.raw_var[0 : end - start, ...]

    def __str__(self) -> str:
        return f"{self.var} linked to lgdo.Array(shape={self.io_array.shape}, dtype={self.io_array.dtype}, attrs={self.io_array.attrs})"


class LGDOArrayOfEqualSizedArraysIOManager(IOManager):
    r""":class:`IOManager` for buffers that are :class:`lgdo.ArrayOfEqualSizedArray`\ s."""

    def __init__(self, io_array: np.ArrayOfEqualSizedArrays, var: ProcChainVar) -> None:
        assert isinstance(io_array, lgdo.ArrayOfEqualSizedArrays) and isinstance(
            var, ProcChainVar
        )

        unit = io_array.attrs.get("units", None)
        var.update_auto(dtype=io_array.dtype, shape=io_array.nda.shape[1:], unit=unit)

        if isinstance(var.unit, (CoordinateGrid, Quantity, Unit)):
            if isinstance(var.unit, CoordinateGrid):
                var_u = var.unit.period.u
            elif isinstance(var.unit, Quantity):
                var_u = var.unit.u
            else:
                var_u = var.unit

            if unit is None:
                unit = var_u
            elif ureg.is_compatible_with(var_u, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError(
                    f"LGDO array and variable {var} have incompatible units "
                    f"({var_u} and {unit})"
                )
        elif isinstance(var.unit, str) and unit is None:
            unit = var.unit

        if "units" not in io_array.attrs and unit is not None:
            io_array.attrs["units"] = str(unit)

        self.io_array = io_array
        self.var = var
        self.raw_var = var.get_buffer(unit)

        if (
            self.var.shape != self.io_array.nda.shape[1:]
            or self.raw_var.dtype != self.io_array.dtype
        ):
            raise ProcessingChainError(
                f"LGDO object "
                f"{self.io_buf.form_datatype()} is "
                f"incompatible with {str(self.var)}"
            )

    def read(self, start: int, end: int) -> None:
        if start >= len(self.io_array):
            raise IndexError
        end = min(end, len(self.io_array))
        self.raw_var[0 : end - start, ...] = self.io_array[start:end, ...]

    def write(self, start: int, end: int) -> None:
        self.io_array.resize(end)
        self.io_array[start:end, ...] = self.raw_var[0 : end - start, ...]

    def __str__(self) -> str:
        return f"{self.var} linked to lgdo.ArrayOfEqualSizedArrays(shape={self.io_array.shape}, dtype={self.io_array.dtype}, attrs={self.io_array.attrs})"


class LGDOVectorOfVectorsIOManager(IOManager):
    r""":class:`IOManager` for buffers that are :class:`lgdo.VectorOfVectors`\ s."""

    def __init__(self, io_vov: lgdo.VectorOfVectors, var: ProcChainVar) -> None:
        assert (
            isinstance(io_vov, lgdo.VectorOfVectors)
            and isinstance(var, ProcChainVar)
            and var.vector_len is not None
        )

        if not np.issubdtype(var.vector_len.dtype, np.integer):
            raise ProcessingChainError(
                f"{var.vector_len} must be an integer to act as a vector len"
            )

        unit = io_vov.attrs.get("units", None)
        var.update_auto(dtype=io_vov.dtype, shape=10, unit=unit)

        if isinstance(var.unit, (CoordinateGrid, Quantity, Unit)):
            if isinstance(var.unit, CoordinateGrid):
                var_u = var.unit.period.u
            elif isinstance(var.unit, Quantity):
                var_u = var.unit.u
            else:
                var_u = var.unit

            if unit is None:
                unit = var_u
            elif ureg.is_compatible_with(var_u, unit):
                unit = ureg.Quantity(unit).u
            else:
                raise ProcessingChainError(
                    f"LGDO array and variable {var} have incompatible units "
                    f"({var_u} and {unit})"
                )
        elif isinstance(var.unit, str) and unit is None:
            unit = var.unit

        if "units" not in io_vov.attrs and unit is not None:
            io_vov.attrs["units"] = str(unit)

        self.io_vov = io_vov
        self.io_vov.flattened_data.resize(
            len(self.io_vov) * np.prod(var.buffer.shape[1:]), trim=True
        )
        self.var = var
        self.raw_var = var.get_buffer(unit)
        self.len_var = var.vector_len.get_buffer()

        if self.raw_var.dtype != self.io_vov.dtype:
            raise ProcessingChainError(
                f"LGDO object "
                f"{self.io_vov.flattened_data.form_datatype()} is "
                f"incompatible with {str(self.var)}"
            )

    @guvectorize(
        [
            f"{t}[:],u4[:],u4,u4[:],{t}[:,:]"
            for t in [
                "b1",
                "i1",
                "i2",
                "i4",
                "i8",
                "u1",
                "u2",
                "u4",
                "u8",
                "f4",
                "f8",
                "c8",
                "c16",
            ]
        ],
        "(n),(l),(),(l),(l,m)",
        **nb_kwargs,
    )
    def _vov2nda(flat_arr_in, cl_in, start_idx_in, l_out, aoa_out):  # noqa: N805
        prev_cl = start_idx_in
        for i, cl in enumerate(cl_in):
            l_out[i] = cl - prev_cl
            if l_out[i] > aoa_out.shape[1]:
                raise DSPFatal(
                    "VectorOfVectors entry has length larger than array variable length"
                )
            aoa_out[i, : l_out[i]] = flat_arr_in[prev_cl:cl]
            prev_cl = cl

    def read(self, start: int, end: int) -> None:
        if start >= len(self.io_vov):
            raise IndexError
        end = min(end, len(self.io_vov))
        self.raw_var = 0 if np.issubdtype(self.raw_var.dtype, np.integer) else np.nan
        LGDOVectorOfVectorsIOManager._vov2nda(
            self.io_vov.flattened_data,
            self.io_vov.cumulative_length,
            self.io_vov.cumulative_length[start - 1] if start > 0 else 0,
            self.len_var,
            self.raw_var,
        )

    def write(self, start: int, end: int) -> None:
        self.io_vov.resize(end)
        self.io_vov.flattened_data.resize(
            int(self.io_vov.cumulative_length[start] + np.sum(self.len_var))
        )
        self.io_vov._set_vector_unsafe(
            start, self.raw_var[: end - start], self.len_var[: end - start]
        )

    def __str__(self) -> str:
        return f"{self.var} linked to lgdo.VectorOfVectors(vector_len={self.var.vector_len}, dtype={self.io_vov.flattened_data.dtype}, attrs={self.io_vov.attrs})"


class LGDOWaveformIOManager(IOManager):
    def __init__(self, wf_table: lgdo.WaveformTable, variable: ProcChainVar) -> None:
        assert isinstance(wf_table, lgdo.WaveformTable) and isinstance(
            variable, ProcChainVar
        )

        self.io_wf = wf_table

        dt_units = wf_table.dt_units
        t0_units = wf_table.t0_units
        if dt_units is None:
            dt_units = t0_units
        elif t0_units is None:
            t0_units = dt_units

        # If needed create a new coordinate grid from the IO buffer
        if (
            variable.grid is auto
            and isinstance(dt_units, str)
            and dt_units in ureg
            and isinstance(t0_units, str)
            and t0_units in ureg
        ):
            grid = CoordinateGrid(
                ureg.Quantity(self.io_wf.dt[0], dt_units),
                ProcChainVar(
                    variable.proc_chain,
                    variable.name + "_dt",
                    shape=(),
                    dtype=self.io_wf.t0.dtype,
                    grid=None,
                    unit=dt_units,
                    is_coord=True,
                ),
            )
        else:
            grid = None

        self.var = variable
        self.var.update_auto(
            shape=self.io_wf.values.shape[1:],
            dtype=self.io_wf.values.dtype,
            grid=grid,
            unit=wf_table.values_units,
            is_coord=False,
        )

        if dt_units is None:
            dt_units = self.var.grid.unit_str()
            t0_units = self.var.grid.unit_str()

        self.wf_var = self.var.buffer

        self.t0_var = self.var.grid.get_offset(t0_units)
        self.variable_t0 = isinstance(self.t0_var, np.ndarray)
        if not self.variable_t0:
            self.io_wf.t0[:] = self.t0_var
        self.io_wf.t0_units = t0_units

        self.io_wf.dt[:] = self.var.grid.get_period(dt_units)
        self.io_wf.dt_units = dt_units

    def read(self, start: int, end: int) -> None:
        if start >= len(self.io_wf):
            raise IndexError
        end = min(end, len(self.io_wf))
        self.wf_var[0 : end - start, ...] = self.io_wf.values[start:end, ...]
        self.t0_var[0 : end - start, ...] = self.io_wf.t0[start:end, ...]

    def write(self, start: int, end: int) -> None:
        self.io_wf.resize(end)
        self.io_wf.values[start:end, ...] = self.wf_var[0 : end - start, ...]
        if self.variable_t0:
            self.io_wf.t0[start:end, ...] = self.t0_var[0 : end - start, ...]

    def __str__(self) -> str:
        return (
            f"{self.var} linked to lgdo.WaveformTable("
            f"values(shape={self.io_wf.values.shape}, dtype={self.io_wf.values.dtype}, attrs={self.io_wf.values.attrs}), "
            f"dt(shape={self.io_wf.dt.shape}, dtype={self.io_wf.dt.dtype}, attrs={self.io_wf.dt.attrs}), "
            f"t0(shape={self.io_wf.t0.shape}, dtype={self.io_wf.t0.dtype}, attrs={self.io_wf.t0.attrs}))"
        )


def build_processing_chain(
    processors: dict | str,
    tb_in: lgdo.Table = None,
    db_dict: dict = None,
    outputs: list[str] = None,
    block_width: int = 16,
) -> tuple[ProcessingChain, list[str], lgdo.Table]:
    """Produces a :class:`ProcessingChain` object and an LGDO
    :class:`~lgdo.types.table.Table` for output parameters from an input LGDO
    :class:`~lgdo.types.table.Table` and a JSON or YAML recipe.

    Parameters
    ----------
    processors
        A dictionary or YAML/JSON filename containing the recipes for computing DSP
        parameter from raw parameters. The format is as follows:

        .. code-block:: json
            :force:

            {
                "name1, name2" : {
                    "function" : "func1"
                    "module" : "mod1"
                    "args" : ["arg1", 3, "arg2"]
                    "kwargs" : {"key1": "val1"}
                    "init_args" : ["arg1", 3, "arg2"]
                    "unit" : ["u1", "u2"]
                    "defaults" : {"arg1": "defval1"}
                },
                ...
            }

        - ``name1, name2`` -- dictionary. key contains comma-separated
          names of parameters computed

          - ``function`` -- string, name of function to call.  Function
            should implement the :class:`numpy.gufunc` interface, a factory
            function returning a ``gufunc``, or an arbitrary function that
            can be mapped onto a ``gufunc``
          - ``module`` -- string, name of module containing function
          - ``args``-- list of strings or numerical values. Contains
            list of names of computed and input parameters or
            constant values used as inputs to function. Note that
            outputs should be fed by reference as args! Arguments read
            from the database are prepended with ``db``.
          - ``kwargs`` -- dictionary. Keyword arguments for
            :meth:`ProcesssingChain.add_processor`.
          - ``init_args`` --  list of strings or numerical values. List
            of names of computed and input parameters or constant values
            used to initialize a :class:`numpy.gufunc` via a factory
            function
          - ``unit`` -- list of strings. Units for parameters
          - ``defaults`` -- dictionary. Default value to be used for
            arguments read from the database
        - The dictionary can also be nested in another, keyed as ``processors``

    tb_in
        input table. This table will be linked to use as inputs when
        executing processors. Can be empty (for now), as long as fields
        and attrs are set.
    db_dict
        A nested :class:`dict` pointing to values for database arguments. As
        instance, if a processor uses the argument ``db.trap.risetime``, it
        will look up ``db_dict['trap']['risetime']`` and use the found value.
        If no value is found, use the default defined in `processors`.
    outputs
        List of parameters to put in the output LGDO table.
    block_width
        number of entries to process at once. To optimize performance,
        a multiple of 16 is preferred, but if performance is not an issue
        any value can be used.

    Returns
    -------
    (proc_chain, field_mask, tb_out)
        - `proc_chain` -- :class:`ProcessingChain` object that is executed
        - `field_mask` -- list of names of input fields that will be used.
          This can be used to ensure only needed values are read in.
        - `tb_out` -- output :class:`~lgdo.table.Table` with size 0, with
          fields and attrs set up to contain outputs
    """
    db_parser = re.compile(r"(?![^\w_.])db\.[\w_.]+")

    if isinstance(processors, str):
        with open(processors) as f:
            processors = safe_load(f)
    elif processors is None:
        processors = {}
    elif isinstance(processors, MutableMapping):
        # We don't want to modify the input!
        processors = deepcopy(processors)
    else:
        raise ValueError("processors must be a dict, json/yaml file, or None")

    if outputs is None:
        if "outputs" not in processors:
            raise ValueError("outputs not provided")
        outputs = processors["outputs"]

    if "processors" in processors:
        processors = processors["processors"]

    buffer_len = len(tb_in) if tb_in is not None else 1
    proc_chain = ProcessingChain(block_width, buffer_len)

    # prepare the processor list
    multi_out_procs = {}
    for key, node in processors.items():
        # if we have multiple outputs, add each to the processesors list
        keys = [k for k in re.split(",| ", key) if k != ""]
        if len(keys) > 1:
            for k in keys:
                multi_out_procs[k] = key

        # find DB lookups in args and replace the values
        if isinstance(node, str):
            node = {"function": node}
            processors[key] = node

        if "function" not in node:
            raise ProcessingChainError
        function = node["function"]
        f_parse = ast.parse(function, mode="eval").body

        mod_err_str = f"Module specified twice for parameter {key}"
        args_err_str = (
            f"Cannot specify arguments if function is expr for parameter {key}"
        )
        if isinstance(f_parse, ast.Name):
            pass
        elif isinstance(f_parse, ast.Attribute):
            module = function[f_parse.value.col_offset : f_parse.value.end_col_offset]
            if module in ProcessingChain.module_list and "args" not in node:
                # this is an attribute like np.pi
                node["module"] = None
                node["args"] = [function]
            else:
                node["function"] = f_parse.attr
                if "module" in node:
                    raise ProcessingChainError(mod_err_str)
                node["module"] = module
        elif isinstance(f_parse, ast.Call):
            # this is a function. Parse arguments from here
            if "args" in node:
                raise ProcessingChainError(args_err_str)

            if (
                isinstance(f_parse.func, ast.Name)
                and f_parse.func.id in ProcessingChain.func_list
                and "module" not in node
            ):
                # this should be treated as an inline expression assignment
                node["module"] = None
                node["args"] = [function]
            elif isinstance(f_parse.func, ast.Name):
                node["function"] = f_parse.func.id
                node["args"] = [
                    function[a.col_offset : a.end_col_offset]
                    for a in f_parse.args + f_parse.keywords
                ]
            elif isinstance(f_parse.func, ast.Attribute):
                node["function"] = f_parse.func.attr
                if "module" in node:
                    raise ProcessingChainError(mod_err_str)
                mod = f_parse.func.value
                node["module"] = function[mod.col_offset : mod.end_col_offset]
                node["args"] = [
                    function[a.col_offset : a.end_col_offset]
                    for a in f_parse.args + f_parse.keywords
                ]
        else:
            # this is an expression that ProcessingChain will try to parse
            if "args" in node:
                raise ProcessingChainError(args_err_str)
            if "module" in node:
                raise ProcessingChainError(mod_err_str)
            node["module"] = None
            node["args"] = [function]

        if "module" not in node:
            raise ProcessingChainError(f"Could not find module for parameter {key}")
        if "args" not in node:
            raise ProcessingChainError(f"Could not find args for parameter {key}")

        # substitute database values in arguments
        args = node["args"]
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                continue
            for db_var in db_parser.findall(arg):
                try:
                    db_node = db_dict
                    for db_key in db_var[3:].split("."):
                        db_node = db_node[db_key]
                    log.debug(f"database lookup: found {db_node} for {db_var}")
                except (KeyError, TypeError):
                    try:
                        db_node = node["defaults"][db_var]
                        log.debug(
                            f"Database lookup: using default value of {db_node} for {db_var}"
                        )
                    except (KeyError, TypeError):
                        raise ProcessingChainError(
                            f"""did not find {db_var} in database, and could
                                not find default value."""
                        )
                if arg == db_var:
                    arg = db_node
                else:
                    arg = arg.replace(db_var, str(db_node))
            args[i] = arg
            if "args" not in node:
                node["function"] = arg

        # parse the arguments list for prereqs, if not included explicitly
        if "prereqs" not in node:
            prereqs = []

            for arg in args:
                if not isinstance(arg, str):
                    continue
                for prereq in proc_chain.get_variable(arg, True):
                    if prereq not in prereqs and prereq not in keys:
                        prereqs.append(prereq)
            node["prereqs"] = prereqs

        log.debug(f"prereqs for {key} are {node['prereqs']}")

    processors.update(multi_out_procs)

    def resolve_dependencies(
        par: str, resolved: list[str], leafs: list[str], unresolved: list[str] = None
    ) -> None:
        """
        Recursive function to crawl through the parameters/processors and get a
        sequence of unique parameters such that parameters always appear after
        their dependencies. For parameters that are not produced by the
        :class:`ProcessingChain` (i.e. input/db parameters), add them to the
        list of leafs.

        .. [ref] https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html
        """
        if unresolved is None:
            unresolved = []

        if par in resolved:
            return
        elif par in unresolved:
            raise ProcessingChainError(
                f"Circular references detected for parameter '{par}'"
            )

        # if we don't find a node, this is a leaf
        node = processors.get(par)
        if node is None:
            if par not in leafs:
                leafs.append(par)
            return

        # if it's a string, that means it is part of a processor that returns multiple outputs (see above); in that case, node is a str pointing to the actual node we want
        if isinstance(node, str):
            resolve_dependencies(node, resolved, leafs, unresolved)
            return

        edges = node["prereqs"]
        unresolved.append(par)
        for edge in edges:
            resolve_dependencies(edge, resolved, leafs, unresolved)
        resolved.append(par)
        unresolved.remove(par)

    proc_par_list = []  # calculated from processors
    input_par_list = []  # input from file and used for processors
    copy_par_list = []  # copied from input to output
    out_par_list = []
    for out_par in outputs:
        if out_par not in processors:
            copy_par_list.append(out_par)
        else:
            resolve_dependencies(out_par, proc_par_list, input_par_list)
            out_par_list.append(out_par)

    log.debug(f"processing parameters: {proc_par_list}")
    log.debug(f"required input parameters: {input_par_list}")
    log.debug(f"copied output parameters: {copy_par_list}")
    log.debug(f"processed output parameters: {out_par_list}")

    # Now add all of the input buffers from tb_in
    for input_par in input_par_list:
        if input_par not in tb_in:
            log.warning(f"'{input_par}' not found in input files or dsp config.")
        try:
            proc_chain.link_input_buffer(input_par, tb_in[input_par])
        except Exception as e:
            raise ProcessingChainError(
                f"Exception raised while linking input buffer '{input_par}'."
            ) from e

    # now add the processors
    for proc_par in proc_par_list:
        recipe = processors[proc_par]
        try:
            # if we are invoking a built in expression, have the parser
            # add it to the processing chain, and then add a new variable
            # that shares its buffer
            if recipe["module"] is None:
                assert len(recipe["args"]) == 1
                fun_var = proc_chain.get_variable(recipe["args"][0])
                if isinstance(fun_var, ProcChainVar):
                    new_var = proc_chain.add_variable(
                        name=proc_par,
                        dtype=fun_var.dtype,
                        shape=fun_var.shape,
                        grid=fun_var.grid,
                        unit=fun_var.unit,
                        is_coord=fun_var.is_coord,
                    )
                    new_var._buffer = fun_var._buffer

                else:
                    new_var = proc_chain.set_constant(
                        varname=proc_par,
                        val=fun_var,
                    )
                log.debug(f"setting {new_var} = {fun_var}")
                continue

            module = importlib.import_module(recipe["module"])
            func = getattr(module, recipe["function"])

            args = recipe["args"]
            new_vars = [k for k in re.split(",| ", proc_par) if k != ""]

            # Initialize the new variables, if needed
            if "unit" in recipe:
                for i, name in enumerate(new_vars):
                    unit = recipe.get("unit", auto)
                    if isinstance(unit, list):
                        unit = unit[i]

                    proc_chain.add_variable(name, unit=unit)

            # get this list of kwargs
            kwargs = recipe.get("kwargs", {})  # might also need db lookup here
            kwargs.update(
                {
                    key: recipe[key]
                    for key in ["signature", "types", "coord_grid"]
                    if key in recipe
                }
            )

            # if init_args are defined, parse any strings and then call func
            # as a factory/constructor function
            try:
                init_args_in = recipe["init_args"]
                init_args = []
                init_kwargs = {}
                for arg in init_args_in:
                    if not isinstance(arg, str):
                        pass

                    for db_var in db_parser.findall(arg):
                        try:
                            db_node = db_dict
                            for key in db_var[3:].split("."):
                                db_node = db_node[key]
                            log.debug(f"database lookup: found {db_node} for {db_var}")
                        except (KeyError, TypeError):
                            try:
                                db_node = recipe["defaults"][db_var]
                                log.debug(
                                    "database lookup: using default value of {db_node} for {db_var}"
                                )
                            except (KeyError, TypeError):
                                raise ProcessingChainError(
                                    f"did not find {db_var} in database, and "
                                    f"could not find default value."
                                )

                        if arg == db_var:
                            arg = db_node
                        else:
                            arg = arg.replace(db_var, str(db_node))

                    # see if string can be parsed by proc_chain
                    if isinstance(arg, str):
                        arg = proc_chain.get_variable(arg)
                    if isinstance(arg, MutableMapping):
                        init_kwargs.update(arg)
                    else:
                        init_args.append(arg)

                expr = ", ".join(
                    [f"{a}" for a in init_args]
                    + [f"{k}={v}" for k, v in init_kwargs.items()]
                )
                log.debug(f"building function from init_args: {func.__name__}({expr})")
                func = func(*init_args)
            except KeyError:
                pass

            # Check if new variables should be treated as constants
            params = []
            kw_params = {}
            out_params = []
            is_const = True
            for param in args:
                if isinstance(param, str):
                    param = proc_chain.get_variable(param)
                if isinstance(param, MutableMapping):
                    kw_params.update(param)
                    param = list(param.values())[0]
                elif isinstance(param, str):
                    params.append(f"'{param}'")
                else:
                    params.append(param)

                if isinstance(param, ProcChainVar):
                    if param.name in new_vars:
                        out_params.append(param)
                    elif not param.is_const:
                        is_const = False

            if is_const:
                if out_params:
                    for param in out_params:
                        param.is_const = True
                    proc_man = ProcessorManager(
                        proc_chain,
                        func,
                        params,
                        kw_params,
                        kwargs.get("signature", None),
                        kwargs.get("types", None),
                    )
                    proc_man.execute()
                    for param in out_params:
                        log.debug(
                            f"set constant: {param.description()} = {param.get_buffer()}"
                        )

                else:
                    const_val = func(*params, **kw_params)
                    if len(new_vars) == 1:
                        const_val = [const_val]
                    for var, val in zip(new_vars, const_val):
                        proc_chain.set_constant(var, val)

            else:
                proc_chain.add_processor(func, *params, kw_params, **kwargs)

        except Exception as e:
            raise ProcessingChainError(
                "Exception raised while attempting to add processor:\n"
                + json.dumps(recipe, indent=2)
            ) from e

    # build the output buffers
    tb_out = lgdo.Table(size=buffer_len)

    # add inputs that are directly copied
    for copy_par in copy_par_list:
        if copy_par not in tb_in:
            log.warning(
                f"'{copy_par}' not found in input files or dsp config. Building output without it!"
            )
        else:
            tb_out.add_field(copy_par, tb_in[copy_par])

    # finally, add the output buffers to tb_out and the proc chain
    for out_par in out_par_list:
        try:
            buf_out = proc_chain.link_output_buffer(out_par)
            recipe = processors[out_par]
            if isinstance(recipe, str):
                recipe = processors[recipe]
            buf_out.attrs.update(recipe.get("lh5_attrs", {}))
            buf_out.resize(len(tb_out))
            tb_out.add_field(out_par, buf_out)
        except Exception as e:
            raise ProcessingChainError(
                f"Exception raised while linking output buffer {out_par}."
            ) from e

    field_mask = input_par_list + copy_par_list

    return (proc_chain, field_mask, tb_out)
