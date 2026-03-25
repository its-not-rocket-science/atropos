"""Logging configuration for Atropos."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Environment variable names
ENV_LOG_LEVEL = "ATROPOS_LOG_LEVEL"
ENV_LOG_FILE = "ATROPOS_LOG_FILE"
ENV_LOG_FORMAT = "ATROPOS_LOG_FORMAT"

# Default values
DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_LOG_FORMAT = "text"
DEFAULT_LOG_FILE = None  # No file logging by default


def get_log_level_from_env() -> LogLevel:
    """Get log level from environment variable."""
    level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        logging.warning(f"Invalid log level {level!r}, using {DEFAULT_LOG_LEVEL}")
        level = DEFAULT_LOG_LEVEL
    return level  # type: ignore[return-value]


def get_log_file_from_env() -> Path | None:
    """Get log file path from environment variable."""
    path_str = os.environ.get(ENV_LOG_FILE)
    if not path_str:
        return None
    path = Path(path_str).expanduser().resolve()
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_log_format_from_env() -> str:
    """Get log format from environment variable."""
    fmt = os.environ.get(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT)
    if fmt not in ("text", "json"):
        logging.warning(f"Invalid log format {fmt!r}, using {DEFAULT_LOG_FORMAT}")
        fmt = DEFAULT_LOG_FORMAT
    return fmt


def setup_logging(
    *,
    level: LogLevel | None = None,
    log_file: Path | str | None = None,
    format: str | None = None,  # noqa: A002
    verbose: bool = False,
    debug: bool = False,
    traceback: bool = False,
) -> None:
    """Configure logging for Atropos.

    Args:
        level: Logging level (overrides environment variable and verbose/debug flags).
        log_file: Path to log file (overrides environment variable).
        format: Log format ("text" or "json").
        verbose: Enable verbose output (sets level to INFO unless overridden).
        debug: Enable debug output (sets level to DEBUG unless overridden).
        traceback: Enable full traceback display for exceptions.
    """
    # Determine log level
    if level is None:
        if debug:
            level = "DEBUG"
        elif verbose:
            level = "INFO"
        else:
            level = get_log_level_from_env()

    # Determine log file
    if log_file is None:
        log_file = get_log_file_from_env()
    elif isinstance(log_file, str):
        log_file = Path(log_file).expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format is None:
        format = get_log_format_from_env()

    # Convert level string to logging constant
    level_num = getattr(logging, level)

    # Configure root logger for atropos
    logger = logging.getLogger("atropos")
    logger.setLevel(level_num)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Create formatter
    formatter: logging.Formatter
    if format == "json":
        # Simple JSON formatter (could be enhanced with proper JSON logging)
        import json
        import time

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": time.time(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        formatter = JsonFormatter()
    else:
        # Human-readable text format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level_num)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level_num)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Capture warnings from warnings module
    logging.captureWarnings(True)

    # Store traceback setting as module attribute for exception handling
    import atropos.logging_config as config_module

    config_module.SHOW_TRACEBACK = traceback


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the given name, or the default 'atropos' logger.

    Args:
        name: Logger name. If None, returns the root 'atropos' logger.

    Returns:
        Configured logger instance.
    """
    if name is None:
        name = "atropos"
    elif not name.startswith("atropos."):
        name = f"atropos.{name}"
    return logging.getLogger(name)


# Module-level configuration
SHOW_TRACEBACK = False
