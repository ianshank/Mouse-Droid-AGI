from __future__ import annotations

from mousedroid.config.schema import LoggingConfig
from mousedroid.logging.setup import _level_to_int, configure_logging, get_logger


def test_configure_logging_json():
    cfg = LoggingConfig(level="INFO", format="json")
    configure_logging(cfg)
    # Should not raise; structlog is now configured with JSON renderer


def test_configure_logging_console():
    cfg = LoggingConfig(level="DEBUG", format="console")
    configure_logging(cfg)
    # Should not raise; structlog is now configured with console renderer


def test_get_logger_returns_bound_logger():
    logger = get_logger("test_module")
    assert logger is not None


def test_level_to_int_debug():
    assert _level_to_int("DEBUG") == 10


def test_level_to_int_info():
    assert _level_to_int("INFO") == 20


def test_level_to_int_warning():
    assert _level_to_int("WARNING") == 30


def test_level_to_int_error():
    assert _level_to_int("ERROR") == 40


def test_level_to_int_critical():
    assert _level_to_int("CRITICAL") == 50


def test_level_to_int_unknown_returns_20():
    assert _level_to_int("BOGUS") == 20


def test_level_to_int_case_insensitive():
    assert _level_to_int("debug") == 10
    assert _level_to_int("Info") == 20


def test_configure_logging_with_log_buffer():
    """Cover line 43: processors.append(log_buffer) when log_buffer is not None."""
    cfg = LoggingConfig(level="INFO", format="json")
    # Use a callable as a dummy processor (structlog processors are callables)
    dummy_buffer = lambda logger, method, event_dict: event_dict  # noqa: E731
    configure_logging(cfg, log_buffer=dummy_buffer)
