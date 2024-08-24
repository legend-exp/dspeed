"""This module implements some helpers for setting up logging."""

import logging

import colorlog

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
FATAL = logging.FATAL
CRITICAL = logging.CRITICAL


def setup(level: int = logging.INFO, logger: logging.Logger = None) -> None:
    """Setup a colorful logging output.

    If `logger` is None, sets up only the ``dpeed`` logger.

    Parameters
    ----------
    level
        logging level (see :mod:`logging` module).
    logger
        if not `None`, setup this logger.

    Examples
    --------
    >>> from dspeed import logging
    >>> logging.setup(level=logging.DEBUG)
    """
    handler = colorlog.StreamHandler()
    colors = colorlog.default_log_colors.copy()
    colors["DEBUG"] = "bold_cyan"
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(name)s [%(levelname)s] %(message)s", log_colors=colors
        )
    )

    if logger is None:
        logger = colorlog.getLogger("dspeed")

    logger.setLevel(level)
    logger.addHandler(handler)
