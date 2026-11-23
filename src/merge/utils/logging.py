"""Project-wide logging helpers."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with stdout and file handlers.

    The logger format is ``[time] [level] [module] message``. Existing handlers
    are removed before adding new ones so repeated script invocations do not
    duplicate log lines in notebooks or resumed runs.

    Args:
        name: Logger name.
        log_file: File path for persistent logs.
        level: Logging level.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(level)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class JsonLinesWriter:
    """JSON Lines writer for training and evaluation records."""

    def __init__(self, path: str):
        """Open a JSONL file for append writes.

        Args:
            path: JSONL output path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        """Append one JSON-serializable record and flush immediately."""
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self.f.close()
