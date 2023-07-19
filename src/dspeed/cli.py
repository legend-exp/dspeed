"""dspeed's command line interface utilities."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__, build_dsp, logging


def dspeed_cli():
    """dspeed's command line interface.

    Defines the command line interface (CLI) of the package, which exposes some
    of the most used functions to the console.  This function is added to the
    ``entry_points.console_scripts`` list and defines the ``dspeed`` executable
    (see ``setuptools``' documentation). To learn more about the CLI, have a
    look at the help section:

    .. code-block:: console

      $ dspeed --hep
    """

    parser = argparse.ArgumentParser(
        prog="dspeed",
        description="""Process LH5 raw files and produce a
        dsp file using a JSON configuration""",
    )

    # global options
    parser.add_argument(
        "--version", action="store_true", help="""Print dspeed version and exit"""
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="""Increase the program verbosity to maximum""",
    )

    # build_dsp
    parser.add_argument(
        "raw_lh5_file",
        nargs="+",
        help="""Input raw LH5 file. Can be a single file or a list of them""",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help=""""JSON file holding configuration of signal processing
                 routines""",
    )
    parser.add_argument(
        "--hdf5-groups",
        "-g",
        nargs="*",
        default=None,
        help="""Name of group in the LH5 file. By default process all base
                groups. Supports wildcards""",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="""Name of output file, if only one is supplied. By default,
                output to <input-filename>_dsp.lh5""",
    )
    parser.add_argument(
        "--database",
        "-D",
        default=None,
        help="""JSON file to read database parameters from.  Should be nested
                dict with channel at the top level, and parameters below that""",
    )
    parser.add_argument(
        "--output-pars",
        "-p",
        nargs="*",
        default=None,
        help="""List of additional output DSP parameters written to file. By
                default use the "outputs" list defined in in the JSON
                configuration file""",
    )
    parser.add_argument(
        "--max-rows",
        "-n",
        default=None,
        type=int,
        help="""Number of rows to process. By default do the whole file""",
    )
    parser.add_argument(
        "--block",
        "-b",
        default=16,
        type=int,
        help="""Number of waveforms to process simultaneously. Default is
                16""",
    )
    parser.add_argument(
        "--chunk",
        "-k",
        default=3200,
        type=int,
        help="""Number of waveforms to read from disk at a time. Default is
                3200""",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite",
        "-w",
        action="store_const",
        const="r",
        dest="writemode",
        default="r",
        help="""Overwrite file if it already exists. Default option""",
    )
    group.add_argument(
        "--update",
        "-u",
        action="store_const",
        const="u",
        dest="writemode",
        help="""Update existing file with new values. Useful with the --output-pars
                option""",
    )
    group.add_argument(
        "--append",
        "-a",
        action="store_const",
        const="a",
        dest="writemode",
        help="""Append values to existing file""",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.setup(logging.DEBUG)
    elif args.debug:
        logging.setup(logging.DEBUG, logging.root)
    else:
        logging.setup()

    if args.version:
        print(__version__)  # noqa: T201
        sys.exit()

    if len(args.raw_lh5_file) > 1 and args.output is not None:
        raise NotImplementedError("not possible to set multiple output file names yet")

    out_files = []
    if len(args.raw_lh5_file) == 1:
        if args.output is None:
            basename = os.path.splitext(os.path.basename(args.raw_lh5_file[0]))[0]
            basename = basename.removesuffix("_raw")
            out_files.append(f"{basename}_dsp.lh5")
        else:
            out_files.append(args.output)
    else:
        for file in args.raw_lh5_file:
            basename = os.path.splitext(os.path.basename(file))[0]
            basename = basename.removesuffix("_raw")
            out_files.append(f"{basename}_dsp.lh5")

    for i in range(len(args.raw_lh5_file)):
        build_dsp(
            args.raw_lh5_file[i],
            out_files[i],
            args.config,
            lh5_tables=args.hdf5_groups,
            database=args.database,
            outputs=args.output_pars,
            n_max=args.max_rows,
            write_mode=args.writemode,
            buffer_len=args.chunk,
            block_width=args.block,
        )
