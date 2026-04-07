"""Tests for mousedroid.logging.setup — structured logging configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

from mousedroid.logging.setup import _level_to_int, configure_logging, get_logger


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_bound_logger(self) -> None:
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_has_name(self) -> None:
        logger = get_logger("my.module")
        assert logger is not None


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_configure_json_format(self) -> None:
        cfg = MagicMock()
        cfg.format = "json"
        cfg.level = "INFO"
        configure_logging(cfg)

    def test_configure_console_format(self) -> None:
        cfg = MagicMock()
        cfg.format = "console"
        cfg.level = "DEBUG"
        configure_logging(cfg)

    def test_configure_with_log_buffer(self) -> None:
        cfg = MagicMock()
        cfg.format = "json"
        cfg.level = "WARNING"
        buffer = MagicMock()
        configure_logging(cfg, log_buffer=buffer)


class TestLevelToInt:
    """Tests for _level_to_int."""

    def test_debug(self) -> None:
        assert _level_to_int("DEBUG") == 10

    def test_info(self) -> None:
        assert _level_to_int("INFO") == 20

    def test_warning(self) -> None:
        assert _level_to_int("WARNING") == 30

    def test_error(self) -> None:
        assert _level_to_int("ERROR") == 40

    def test_critical(self) -> None:
        assert _level_to_int("CRITICAL") == 50

    def test_case_insensitive(self) -> None:
        assert _level_to_int("info") == 20
        assert _level_to_int("Info") == 20

    def test_unknown_defaults_to_info(self) -> None:
        assert _level_to_int("UNKNOWN") == 20
