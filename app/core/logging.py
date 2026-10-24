"""Logging setup without import-time side effects."""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure root logging once and return the root logger."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    resolved_path = Path(
        log_file or os.environ.get("LOCAL_LLM_LOG_FILE", "local_isolated_llm.log")
    ).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == resolved_path.resolve()
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            resolved_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(file_handler)

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(console_handler)

    return root_logger
