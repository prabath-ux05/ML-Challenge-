"""Logging setup and formatting utilities."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "business_entity_resolution", level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger with standard stream formatting.

    Args:
        name: Logger name.
        level: Logging level (defaults to logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
