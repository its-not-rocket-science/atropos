"""Tests for logging configuration and exception handling."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401

from atropos.exceptions import (
    AtroposError,
    CalculationError,
    ConfigurationError,
    ImportErrorException,
    ValidationError,
)
from atropos.logging_config import (
    ENV_LOG_FILE,
    ENV_LOG_FORMAT,
    ENV_LOG_LEVEL,
    get_log_file_from_env,
    get_log_format_from_env,
    get_log_level_from_env,
    get_logger,
    setup_logging,
)


def test_atropos_error_basic() -> None:
    """Test basic AtroposError functionality."""
    error = AtroposError("Test error")
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.context == {}


def test_atropos_error_with_context() -> None:
    """Test AtroposError with context dictionary."""
    error = AtroposError("Validation failed", {"field": "memory_gb", "value": -5.0})
    assert "Validation failed" in str(error)
    assert "field='memory_gb'" in str(error)
    assert "value=-5.0" in str(error)


def test_error_inheritance() -> None:
    """Test that custom exceptions inherit from AtroposError."""
    validation_error = ValidationError("Invalid input")
    config_error = ConfigurationError("Missing config")
    calculation_error = CalculationError("Calculation failed")

    assert isinstance(validation_error, AtroposError)
    assert isinstance(config_error, AtroposError)
    assert isinstance(calculation_error, AtroposError)

    assert str(validation_error) == "Invalid input"
    assert str(config_error) == "Missing config"
    assert str(calculation_error) == "Calculation failed"


def test_import_error_exception() -> None:
    """Test ImportErrorException for backward compatibility."""
    error = ImportErrorException("scikit-optimize not installed")
    assert isinstance(error, AtroposError)
    assert str(error) == "scikit-optimize not installed"


def test_get_log_level_from_env_valid() -> None:
    """Test getting valid log level from environment."""
    with patch.dict(os.environ, {ENV_LOG_LEVEL: "DEBUG"}):
        level = get_log_level_from_env()
        assert level == "DEBUG"

    with patch.dict(os.environ, {ENV_LOG_LEVEL: "INFO"}):
        level = get_log_level_from_env()
        assert level == "INFO"

    with patch.dict(os.environ, {ENV_LOG_LEVEL: "WARNING"}):
        level = get_log_level_from_env()
        assert level == "WARNING"


def test_get_log_level_from_env_invalid() -> None:
    """Test getting invalid log level from environment (should fallback)."""
    with patch.dict(os.environ, {ENV_LOG_LEVEL: "INVALID_LEVEL"}):
        with patch.object(logging, "warning") as mock_warning:
            level = get_log_level_from_env()
            assert level == "WARNING"  # Default
            mock_warning.assert_called_once()


def test_get_log_level_from_env_case_insensitive() -> None:
    """Test that log level is converted to uppercase."""
    with patch.dict(os.environ, {ENV_LOG_LEVEL: "debug"}):
        level = get_log_level_from_env()
        assert level == "DEBUG"


def test_get_log_file_from_env() -> None:
    """Test getting log file path from environment."""
    with tempfile.TemporaryDirectory() as tempdir:
        log_path = Path(tempdir) / "atropos.log"
        with patch.dict(os.environ, {ENV_LOG_FILE: str(log_path)}):
            result = get_log_file_from_env()
            assert result == log_path.expanduser().resolve()
            assert result.parent.exists()  # Directory should be created


def test_get_log_file_from_env_none() -> None:
    """Test getting log file when environment variable not set."""
    with patch.dict(os.environ, clear=True):
        result = get_log_file_from_env()
        assert result is None


def test_get_log_format_from_env_valid() -> None:
    """Test getting valid log format from environment."""
    with patch.dict(os.environ, {ENV_LOG_FORMAT: "json"}):
        fmt = get_log_format_from_env()
        assert fmt == "json"

    with patch.dict(os.environ, {ENV_LOG_FORMAT: "text"}):
        fmt = get_log_format_from_env()
        assert fmt == "text"


def test_get_log_format_from_env_invalid() -> None:
    """Test getting invalid log format from environment (should fallback)."""
    with patch.dict(os.environ, {ENV_LOG_FORMAT: "invalid"}):
        with patch.object(logging, "warning") as mock_warning:
            fmt = get_log_format_from_env()
            assert fmt == "text"  # Default
            mock_warning.assert_called_once()


def test_setup_logging_default() -> None:
    """Test default logging setup (no environment variables)."""
    with patch.dict(os.environ, clear=True):
        # Clear any existing handlers
        logger = logging.getLogger("atropos")
        logger.handlers.clear()

        setup_logging()

        assert logger.level == logging.WARNING  # Default level
        assert len(logger.handlers) == 1  # Only console handler
        assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logging_verbose() -> None:
    """Test logging setup with verbose flag."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    setup_logging(verbose=True)

    assert logger.level == logging.INFO


def test_setup_logging_debug() -> None:
    """Test logging setup with debug flag."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    setup_logging(debug=True)

    assert logger.level == logging.DEBUG


def test_setup_logging_explicit_level() -> None:
    """Test logging setup with explicit level overrides flags."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    setup_logging(level="ERROR", verbose=True, debug=True)

    # Explicit level should override verbose/debug flags
    assert logger.level == logging.ERROR


def test_setup_logging_with_file() -> None:
    """Test logging setup with log file."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    with tempfile.TemporaryDirectory() as tempdir:
        log_file = Path(tempdir) / "test.log"
        setup_logging(log_file=log_file)

        assert len(logger.handlers) == 2  # Console + file handler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_file.resolve())

        # Close file handlers to avoid locking issues on Windows
        for handler in file_handlers:
            handler.close()
        logger.handlers.clear()


def test_setup_logging_json_format() -> None:
    """Test logging setup with JSON format."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    setup_logging(format="json")

    handler = logger.handlers[0]
    formatter = handler.formatter

    # Verify it's a JsonFormatter
    assert formatter.__class__.__name__ == "JsonFormatter"

    # Test that it produces valid JSON
    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(log_record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test message"


def test_setup_logging_text_format() -> None:
    """Test logging setup with text format."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    setup_logging(format="text")

    handler = logger.handlers[0]
    formatter = handler.formatter

    # Should be regular Formatter
    assert isinstance(formatter, logging.Formatter)
    assert "%(asctime)s - %(name)s - %(levelname)s - %(message)s" in formatter._fmt


def test_setup_logging_clears_existing_handlers() -> None:
    """Test that setup_logging clears existing handlers to avoid duplication."""
    logger = logging.getLogger("atropos")
    logger.handlers.clear()

    # Add some dummy handlers
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.StreamHandler())
    assert len(logger.handlers) == 2

    setup_logging()

    # Should be cleared and replaced with configured handlers
    assert len(logger.handlers) == 1  # Only console handler


def test_get_logger_default() -> None:
    """Test get_logger with no name returns atropos logger."""
    logger = get_logger()
    assert logger.name == "atropos"


def test_get_logger_with_name() -> None:
    """Test get_logger with module name."""
    logger = get_logger("calculations")
    assert logger.name == "atropos.calculations"


def test_get_logger_with_full_name() -> None:
    """Test get_logger with already prefixed name."""
    logger = get_logger("atropos.validation")
    assert logger.name == "atropos.validation"


def test_show_traceback_module_variable() -> None:
    """Test that SHOW_TRACEBACK module variable is set."""
    import atropos.logging_config as config_module

    # Reset to default
    config_module.SHOW_TRACEBACK = False

    setup_logging(traceback=True)
    assert config_module.SHOW_TRACEBACK is True

    setup_logging(traceback=False)
    assert config_module.SHOW_TRACEBACK is False


def test_capture_warnings() -> None:
    """Test that warnings are captured by logging."""
    import warnings

    # Clear any existing warning capture
    logging.captureWarnings(False)

    # Create a simple handler that captures records
    captured_records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.WARNING)

    # Attach to root logger to capture warnings
    # (warnings go to py.warnings logger and propagate to root)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)

    # Also attach to py.warnings logger directly as fallback
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = True  # Ensure propagation

    try:
        # Enable warning capture
        logging.captureWarnings(True)

        # Generate a warning
        warnings.warn("Test warning", UserWarning, stacklevel=2)

        # Should have captured the warning
        # Give a moment for logging to propagate (though it's synchronous)
        import sys

        sys.stderr.flush()

        assert len(captured_records) > 0, (
            f"No records captured. captured_records: {captured_records}"
        )
        assert any("Test warning" in str(r.msg) for r in captured_records), (
            f"Records: {[str(r.msg) for r in captured_records]}"
        )
    finally:
        # Clean up
        root_logger.removeHandler(handler)
        warnings_logger.removeHandler(handler)
        logging.captureWarnings(False)
