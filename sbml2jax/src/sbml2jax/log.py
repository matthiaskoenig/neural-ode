"""Opt-in rich logging for scripts and the command lines."""

import logging

from rich.logging import RichHandler


def enable_rich_logging(level: int = logging.INFO) -> None:
    """Log the `sbml2jax` logger with rich at `level`."""
    logger = logging.getLogger("sbml2jax")
    logger.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(show_path=False))
