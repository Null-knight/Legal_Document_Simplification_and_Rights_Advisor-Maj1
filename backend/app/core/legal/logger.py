from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "legal_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger for the legal pipeline.

    Uses a StreamHandler by default and avoids adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_child_logger(parent: Optional[logging.Logger], suffix: str) -> logging.Logger:
    if parent is None:
        return get_logger(suffix)
    return get_logger(f"{parent.name}.{suffix}")
