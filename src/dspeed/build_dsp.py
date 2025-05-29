"""
This module provides high-level routines for running signal processing chains
on waveform data.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Collection, Mapping
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from functools import partial
from itertools import chain

from lgdo import LGDO, Struct, Table, lh5
from tqdm.auto import tqdm
from yaml import safe_load

from .errors import DSPFatal, ProcessingChainError
from .processing_chain import build_processing_chain

log = logging.getLogger("dspeed")


def build_dsp(
    raw_in: str | LGDO,
    dsp_out: str | None = None,
    dsp_config: str | Mapping = None,
    lh5_tables: Collection[str] | str = None,
    base_group: str = None,
    database: str | Mapping = None,
    outputs: Collection[str] = None,
    write_mode: str = None,
    entry_list: Collection[int] = None,
    entry_mask: Collection[bool] = None,
    i_start: int = 0,
    n_entries: int | None = None,
    buffer_len: int = 3200,
    block_width: int = 16,
    processes: int = None,
    chan_config: str | Mapping[str, str] = None,
) -> None:
    """Convert raw-tier LH5 data into dsp-tier LH5 data by running a sequence
    of processors via the :class:`~.processing_chain.ProcessingChain`.

    Parameters
    ----------
    raw_in
        raw data to process. Can be name of raw-tier LH5 file to read from,
        LH5Iterator, or LGDO Table
    dsp_out
        name of file in which to output data. If None return a :class:`lgdo.Struct` or
        :class:`lgdo.Table`
    dsp_config
        :class:`dict` or name of JSON or YAML file containing the recipe for computing
        DSP parameters. If ``chan_config`` is provided, this is the default configuration
        to use. Can only be ``None`` if ``chan_config`` is provided, in which case we
        skip channels that are not found in ``chan_config`` The format is as follows:

        .. code-block:: json

            {
               "inputs" : [
                 { "file": "fname", "group": "gname", "prefix": "pre_" },
                ]
               "outputs" : [ "par1", "par2" ]
               "processors" : {
                   ...
                }
            }

        - ``inputs`` (optional) -- list of files/lh5 table names to read input data from.
          these will be friended to any input data provided to build_processing_chain.
          - ``file`` -- file path
          - ``group`` -- lh5 table group name.
          - ``prefix`` (optional) -- prefix to disambiguate variable names
          - ``suffix`` (optional) -- suffix to disambiguate variable names
        - ``outputs`` (optional) -- list of output parameters (strings) to compute by
          default.  This will be used if no argument is provided for ``outputs``
        - ``processors`` -- configuration for :class:`~.processing_chain.ProcessingChain`.
          See :func:`~.processing_chain.build_processing_chain` for details.
    lh5_tables
        list of LGDO groups to process in the input file. These table should
        include all input variables for processing or contain a subgroup
        called raw that contains such a table. If ``None``, process
        all valid groups. Note that wildcards are accepted (e.g. "ch*"). Not a
        valid argument if ``raw_in`` is an :class:`lgdo.Table`.
    base_group
        name of group in which to find tables listed in ``lh5_tables``. By default,
        check if there is a base group called ``raw``, otherwise use no base.
    database
        dictionary or name of JSON or YAML file containing a parameter database. See
        :func:`~.processing_chain.build_processing_chain` for details.
    outputs
        list of parameter names to write to the output file. If not provided,
        use list provided under ``"outputs"`` in the DSP configuration file.
    n_max
        number of waveforms to process.
    write_mode
        - ``None`` -- create new output file if it does not exist
        - `'r'` -- delete existing output file with same name before writing
        - `'a'` -- append to end of existing output file
        - `'u'` -- update values in existing output file
    buffer_len
        number of waveforms to read/write from/to disk at a time.
    block_width
        number of waveforms to process at a time.
    chan_config
        an ordered mapping, or a json file containing such a mapping, from
        a channel or wildcard pattern to a DSP config. Loop over channels in
        ``lh5_tables`` and match them to a separate DSP config. If no matching
        channel or pattern is found, use ``dsp_config`` as a default. If channel
        matches several patterns, use the first one found; an ordered mapping
        can be used to override certain patterns. For example:

        .. code-block:: JSON
        {
            "ch1*": "config1.json",
            "ch2000000": "config2.json",
            "ch2*": "config3.json"
        }

        will process all channels beginning with 2, except for 2000000, with config3.
    """
    db_parser = re.compile(r"(?![^\w_.])db\.[\w_.]+")

    if isinstance(lh5_tables, str):
        lh5_tables = [lh5_tables]

    if isinstance(raw_in, (Table, lh5.LH5Iterator)):
        # single table

        # in this case, lh5_tables will just be used for naming output group
        if base_group is None:
            base_group = ""
        if lh5_tables is None:
            lh5_tables = [""]
        elif len(lh5_tables) > 1:
            raise RuntimeError(
                "Cannot have more than one value in lh5_tables for input of type Table or LH5Iterator"
            )

    elif isinstance(raw_in, str):
        # file name
        # default base_group behavior
        if base_group is None:
            if lh5.ls(raw_in, "raw"):
                base_group = "raw"
            else:
                base_group = ""

        # if no group is specified, assume we want to decode every table in the file
        if lh5_tables is None:
            lh5_tables = lh5.ls(raw_in, f"{base_group}/*")
        elif isinstance(lh5_tables, str):
            lh5_tables = lh5.ls(raw_in, f"{base_group}/f{lh5_tables}")
        elif isinstance(lh5_tables, Collection):
            lh5_tables = [
                tab
                for tab_wc in lh5_tables
                for tab in lh5.ls(raw_in, f"{base_group}/{tab_wc}")
            ]

        elif not (
            isinstance(lh5_tables, Collection)
            and all(isinstance(el, str) for el in lh5_tables)
        ):
            raise RuntimeError(
                "lh5_tables must be None, a string, or a collection of strings"
            )

        # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
        for i, tb in enumerate(lh5_tables):
            if lh5.ls(raw_in, f"{tb}/*") == [f"{tb}/raw"]:
                lh5_tables[i] = f"{tb}/raw"
            elif not lh5.ls(raw_in, tb):
                del lh5_tables[i]

        if len(lh5_tables) == 0:
            raise RuntimeError(f"could not find any valid LH5 table in {raw_in}")

    else:
        raise RuntimeError(
            f"raw_in was not a file name, Table, or LH5Iterator: {raw_in}"
        )

    # get the config(s)
    if isinstance(dsp_config, str):
        with open(lh5.utils.expand_path(dsp_config)) as config_file:
            dsp_config = safe_load(config_file)

    if isinstance(chan_config, str):
        with open(lh5.utils.expand_path(chan_config)) as config_file:
            # safe_load is order preserving, but doesn't load into an OrderedDict
            # and so may not be totally robust here...
            chan_config = safe_load(config_file)
    elif chan_config is None:
        chan_config = {}

    for chan, config in chan_config.items():
        if isinstance(config, str):
            with open(lh5.utils.expand_path(config)) as config_file:
                chan_config[chan] = safe_load(config_file)

    # get the database parameters
    if isinstance(database, str):
        with open(lh5.utils.expand_path(database)) as db_file:
            database = safe_load(db_file)

    if database and not isinstance(database, Mapping):
        raise ValueError("input database is not a valid JSON or YAML file or dict")

    # Setup output
    if dsp_out is None:
        # Output to tables
        if lh5_tables == [""]:
            dsp_st = Table()
        else:
            dsp_st = Struct()
    else:
        # Output to file
        if write_mode is None and os.path.isfile(dsp_out):
            raise FileExistsError(
                f"output file {dsp_out} exists. Set the 'write_mode' keyword"
            )

        # clear existing output files
        if write_mode == "r":
            if os.path.isfile(dsp_out):
                os.remove(dsp_out)

        dsp_st = lh5.LH5Store(keep_open=True)

    # Initialize processing
    if processes and isinstance(raw_in, (lh5.LH5Iterator, str)):
        pool = ProcessPoolExecutor(max_workers=processes)
    else:
        pool = None
    dsp_names = []
    async_results = []

    # loop over tables to run DSP on
    for tb in lh5_tables:
        # get the config to use
        this_config = dsp_config
        for pat, config in chan_config.items():
            if fnmatch(tb, pat):
                this_config = config
                break

        # get the DB values
        if tb not in ("", "raw"):
            chan_name = next(k for k in tb.split("/") if k not in ("", "raw"))
            db_dict = database.get(chan_name) if database else None
            if db_dict is not None:
                log.info(f"Found database for {chan_name}")
        else:
            db_dict = database

        # get input as either table or iterator
        if isinstance(raw_in, str):
            # Setup lh5 iterator from input
            lh5_in = lh5.LH5Iterator(
                raw_in,
                tb,
                entry_list=entry_list,
                entry_mask=entry_mask,
                i_start=i_start,
                n_entries=n_entries,
                buffer_len=buffer_len,
            )
        else:
            lh5_in = raw_in

        # Check for aux input files
        inputs = []
        config_inputs = this_config.get("inputs", [])
        if isinstance(config_inputs, Mapping):
            inputs += [
                (
                    config_inputs["file"],
                    config_inputs["group"],
                    config_inputs.get("prefix", ""),
                    config_inputs.get("suffix", ""),
                )
            ]
        elif isinstance(config_inputs, Collection):
            inputs += [
                (ci["file"], ci["group"], ci.get("prefix", ""), ci.get("suffix", ""))
                for ci in config_inputs
            ]

        for file, group, prefix, suffix in inputs:
            # check if file points to a db override
            if db_parser.fullmatch(file):
                try:
                    db_node = db_dict
                    for db_key in file.split(".")[1:]:
                        db_node = db_node[db_key]
                    log.debug(f"database lookup: found {db_node} for {file}")
                except (KeyError, TypeError):
                    raise ProcessingChainError(f"did not find {file} in database.")

            # check if group points to a db override
            if db_parser.fullmatch(group):
                try:
                    db_node = db_dict
                    for db_key in file.split(".")[1:]:
                        db_node = db_node[db_key]
                    log.debug(f"database lookup: found {db_node} for {group}")
                except (KeyError, TypeError):
                    raise ProcessingChainError(f"did not find {group} in database.")

            if isinstance(lh5_in, lh5.LH5Iterator):
                lh5_in.add_friend(
                    lh5.LH5Iterator(
                        file,
                        group,
                        entry_list=entry_list,
                        entry_mask=entry_mask,
                        i_start=i_start,
                        n_entries=n_entries,
                        buffer_len=buffer_len,
                    ),
                    prefix=prefix,
                    suffix=suffix,
                )
            else:
                lh5_in.join(
                    lh5.read(group, file, n_rows=len(lh5_in)),
                    prefix=prefix,
                    suffix=suffix,
                )

        processors = this_config["processors"]

        # Get outputs from config if they weren't provided
        if outputs is None:
            outputs = this_config["outputs"]

        # start processing
        if isinstance(lh5_in, lh5.LH5Iterator):
            if n_entries is not None:
                lh5_in.n_entries = n_entries
            dsp_result = lh5_in.map(
                _build_dsp,
                processes=pool,
                chunks=1,
                aggregate=Table.append,
                begin=partial(
                    _initialize_channel,
                    processors,
                    db_dict,
                    outputs,
                    buffer_len,
                    block_width,
                ),
                terminate=_finish_channel,
            )
        else:
            if n_entries is not None:
                lh5_in.resize(n_entries)
            _initialize_channel(
                processors, db_dict, outputs, buffer_len, block_width, lh5_in
            )
            dsp_result = _build_dsp(lh5_in)
            _finish_channel(lh5_in)

        # if not multiprocessing, record results; otherwise add them to our list of asynchronous results
        dsp_name = tb.replace("raw", "dsp")
        if not pool:
            _record_result(dsp_result, dsp_st, dsp_name, dsp_out, write_mode, i_start)
        else:
            dsp_names.append(dsp_name)
            async_results.append(dsp_result)

    # if multiprocessing, record results now
    if pool:
        pool.shutdown(wait=False)
        for dsp_name, dsp_tb in zip(dsp_names, chain(*async_results)):
            _record_result(dsp_tb, dsp_st, dsp_name, dsp_out, write_mode, i_start)

    if isinstance(dsp_st, Struct):
        return dsp_st


def _initialize_channel(processors, db_dict, outputs, buffer_len, block_width, lh5_it):
    """Initialize a processing chain for the current table"""
    global t0
    t0 = time.time()

    n_rows = len(lh5_it)
    if isinstance(lh5_it, lh5.LH5Iterator):
        tb_name = lh5_it.groups[0]
        tb_in = lh5_it.read(0)
    else:
        tb_name = "raw"
        tb_in = lh5_it
    log.info(f"Processing table {tb_name} with {n_rows} rows")

    # Setup processing chain
    global proc_chain, field_mask, tb_out
    proc_chain, field_mask, tb_out = build_processing_chain(
        processors,
        tb_in,
        db_dict=db_dict,
        outputs=outputs,
        buffer_len=buffer_len,
        block_width=block_width,
    )

    if isinstance(lh5_it, lh5.LH5Iterator):
        lh5_it.reset_field_mask(field_mask)

    global progress_bar
    if log.getEffectiveLevel() >= logging.INFO:
        progress_bar = tqdm(
            desc=f"Processing table {tb_name}",
            total=n_rows,
            delay=2,
            unit=" rows",
        )

    global timing_info
    timing_info = {"Disk read": 0, "build_processing_chain": time.time() - t0}

    global t_iter
    t_iter = time.time()


def _finish_channel(lh5_it):
    global t0  # noqa: F824
    tb_name = lh5_it.groups[0] if isinstance(lh5_it, lh5.LH5Iterator) else "raw"
    log.info(f"Table {tb_name} processed in {time.time() - t0:.2f} s")

    global timing_info, proc_chain  # noqa: F824
    timing_info.update(proc_chain.get_timing())
    for name, t in timing_info.items():
        log.debug(f"- {name}: {t} s")

    global progress_bar  # noqa: F824
    if log.getEffectiveLevel() >= logging.INFO:
        progress_bar.close()


def _build_dsp(tb_in, lh5_it=None):
    global timing_info, t_iter  # noqa: F824
    timing_info["Disk read"] += time.time() - t_iter

    i_entry = lh5_it.current_i_entry if isinstance(lh5_it, lh5.LH5Iterator) else 0
    try:
        proc_chain.execute(0, len(tb_in))
    except DSPFatal as e:
        # Update the wf_range to reflect the file position
        e.wf_range = f"{i_entry}-{i_entry+len(tb_in)}"
        raise e

    if log.getEffectiveLevel() >= logging.INFO:
        progress_bar.update(len(tb_in))

    t_iter = time.time()

    # copy and return result
    tb_out.resize(len(tb_in))
    return tb_out


def _record_result(dsp_tb, dsp_st, dsp_name, dsp_out, write_mode, i_start):
    # combine tables from each processing chunk
    t0 = time.time()

    if dsp_name == "":
        dsp_name = "dsp"
    # write to file/update struct
    if isinstance(dsp_st, lh5.LH5Store):
        # if name has nesting, build a struct
        gp_name = dsp_name.split("/", 1)
        if len(gp_name) == 2:
            dsp_tb = Struct({gp_name[1]: dsp_tb})
        gp_name = gp_name[0]

        dsp_st.write(
            dsp_tb,
            name=gp_name,
            lh5_file=dsp_out,
            wo_mode="o" if write_mode == "u" else "a",
            write_start=i_start,
        )

        log.info(f"Wrote table {dsp_name} (write took {time.time() - t0} s).")
    elif isinstance(dsp_st, Table):
        dsp_st.update(dsp_tb)
    elif isinstance(dsp_st, Struct):
        dsp_st.update({dsp_name: dsp_tb})
