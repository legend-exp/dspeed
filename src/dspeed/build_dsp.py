"""
This module provides high-level routines for running signal processing chains
on waveform data.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Collection, Mapping

import h5py
import numpy as np
from lgdo import lh5
from tqdm.auto import tqdm
from yaml import safe_load

from .errors import DSPFatal
from .processing_chain import build_processing_chain

log = logging.getLogger("dspeed")


def build_dsp(
    f_raw: str,
    f_dsp: str,
    dsp_config: str | Mapping = None,
    lh5_tables: Collection[str] | str = None,
    database: str | Mapping = None,
    outputs: Collection[str] = None,
    n_max: int = np.inf,
    write_mode: str = None,
    buffer_len: int = 3200,
    block_width: int = 16,
    chan_config: Mapping[str, str] = None,
) -> None:
    """Convert raw-tier LH5 data into dsp-tier LH5 data by running a sequence
    of processors via the :class:`~.processing_chain.ProcessingChain`.

    Parameters
    ----------
    f_raw
        name of raw-tier LH5 file to read from.
    f_dsp
        name of dsp-tier LH5 file to write to.
    dsp_config
        :class:`dict` or name of JSON or YAML file containing
        :class:`~.processing_chain.ProcessingChain` config. See
        :func:`~.processing_chain.build_processing_chain` for details.
    lh5_tables
        list of LGDO groups to process in the input file. These table should
        include all input variables for processing or contain a subgroup
        called raw that contains such a table. If ``None``, process
        all valid groups. Note that wildcards are accepted (e.g. "ch*").
    database
        dictionary or name of JSON or YAMLfile containing a parameter database. See
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
        contains JSON or YAML DSP configuration file names for every table in
        `lh5_tables`.
    """

    if chan_config is not None:
        # clear existing output files
        if write_mode == "r":
            if os.path.isfile(f_dsp):
                os.remove(f_dsp)
            write_mode = "a"

        for tb, dsp_config in chan_config.items():
            log.debug(f"processing table: {tb} with DSP config file {dsp_config}")
            try:
                build_dsp(
                    f_raw,
                    f_dsp,
                    dsp_config,
                    [tb],
                    database,
                    outputs,
                    n_max,
                    write_mode,
                    buffer_len,
                    block_width,
                )
            except RuntimeError:
                log.debug(f"table {tb} not found")
        return

    raw_store = lh5.LH5Store()
    lh5_file = raw_store.gimme_file(f_raw, "r")
    if lh5_file is None:
        raise ValueError(f"input file not found: {f_raw}")
        return

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = lh5.ls(f_raw)
    elif isinstance(lh5_tables, str):
        lh5_tables = lh5.ls(f_raw, lh5_tables)
    elif isinstance(lh5_tables, Collection):
        lh5_tables = [tab for tab_wc in lh5_tables for tab in lh5.ls(f_raw, tab_wc)]
    elif not (
        hasattr(lh5_tables, "__iter__")
        and all(isinstance(el, str) for el in lh5_tables)
    ):
        raise RuntimeError("lh5_tables must be None, a string, or a list of strings")

    # check if group points to raw data; sometimes 'raw' is nested, e.g g024/raw
    for i, tb in enumerate(lh5_tables):
        if (
            "raw" not in tb
            and not isinstance(raw_store.gimme_file(lh5_file, "r")[tb], h5py.Dataset)
            and lh5.ls(lh5_file, f"{tb}/raw")
        ):
            lh5_tables[i] = f"{tb}/raw"
        elif not lh5.ls(lh5_file, tb):
            del lh5_tables[i]

    if len(lh5_tables) == 0:
        raise RuntimeError(f"could not find any valid LH5 table in {f_raw}")

    # get the database parameters. For now, this will just be a dict in a
    # file, but eventually we will want to interface with the metadata repo
    if isinstance(database, str):
        with open(lh5.utils.expand_path(database)) as db_file:
            database = safe_load(db_file)

    if database and not isinstance(database, Mapping):
        raise ValueError("input database is not a valid JSON or YAML file or dict")

    if write_mode is None and os.path.isfile(f_dsp):
        raise FileExistsError(
            f"output file {f_dsp} exists. Set the 'write_mode' keyword"
        )

    # clear existing output files
    if write_mode == "r":
        if os.path.isfile(f_dsp):
            os.remove(f_dsp)

    # loop over tables to run DSP on
    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max < tot_n_rows:
            tot_n_rows = n_max

        chan_name = tb.split("/")[0]
        log.info(f"Processing table {tb} with {tot_n_rows} rows")
        start = time.time()
        db_dict = database.get(chan_name) if database else None
        if db_dict is not None:
            log.info(f"Found database for {chan_name}")
        tb_name = tb.replace("/raw", "/dsp")

        write_offset = 0
        raw_store.gimme_file(f_dsp, "a")
        if write_mode == "a" and lh5.ls(f_dsp, tb_name):
            write_offset = raw_store.read_n_rows(tb_name, f_dsp)

        loading_time = 0
        write_time = 0
        start = time.time()
        # Main processing loop
        lh5_it = lh5.LH5Iterator(f_raw, tb, buffer_len=buffer_len, n_entries=tot_n_rows)
        proc_chain = None
        curr = time.time()
        loading_time += curr - start
        processing_time = 0

        for lh5_in in lh5_it:
            loading_time += time.time() - curr
            # Initialize

            if proc_chain is None:
                proc_chain_start = time.time()
                proc_chain, lh5_it.field_mask, tb_out = build_processing_chain(
                    lh5_in, dsp_config, db_dict, outputs, block_width
                )
                if log.getEffectiveLevel() >= logging.INFO:
                    progress_bar = tqdm(
                        desc=f"Processing table {tb}",
                        total=tot_n_rows,
                        delay=2,
                        unit=" rows",
                    )
                log.info(
                    f"Table: {tb} processing chain built in {time.time() - proc_chain_start:.2f} seconds"
                )

            entries = lh5_it.current_global_entries
            processing_time_start = time.time()
            try:
                proc_chain.execute(0, len(lh5_in))
            except DSPFatal as e:
                # Update the wf_range to reflect the file position
                e.wf_range = f"{entries[0]}-{entries[-1]}"
                raise e
            processing_time += time.time() - processing_time_start
            write_start = time.time()
            raw_store.write(
                obj=tb_out,
                name=tb_name,
                lh5_file=f_dsp,
                wo_mode="o" if write_mode == "u" else "a",
                write_start=write_offset + entries[0],
            )
            write_time += time.time() - write_start
            if log.getEffectiveLevel() >= logging.INFO:
                progress_bar.update(len(lh5_in))

            curr = time.time()
        if log.getEffectiveLevel() >= logging.INFO:
            progress_bar.close()

        log.info(f"Table {tb} processed in {time.time() - start:.2f} seconds")
        log.debug(f"Table {tb} loading time: {loading_time:.2f} seconds")
        log.debug(f"Table {tb} write time: {write_time:.2f} seconds")
        log.debug(f"Table {tb} processing time: {processing_time:.2f} seconds")

        if log.getEffectiveLevel() >= logging.DEBUG:
            times = proc_chain.get_timing()
            log.debug("Processor timing info: ")
            for proc, t in dict(
                sorted(times.items(), key=lambda item: item[1], reverse=True)
            ).items():
                log.debug(f"{proc}: {t:.3f} s")
