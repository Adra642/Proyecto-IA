"""Logging setup shared by every module (console + rotating file)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_ROOT_NAME = "rag"


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> None:
    """Configure console and ``logs/pipeline.log`` handlers (idempotent).

    Args:
        logs_dir: Directory where ``pipeline.log`` is written.
        level: Logging level for the project logger.
    """
    logger = logging.getLogger(_ROOT_NAME)
    if getattr(logger, "_configured", False):
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        logs_dir / "pipeline.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the project logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger``.
    """
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
