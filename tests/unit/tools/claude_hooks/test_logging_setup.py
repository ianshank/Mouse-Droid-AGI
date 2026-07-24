# tests/unit/tools/claude_hooks/test_logging_setup.py
"""Unit tests for hook logging.

The load-bearing contract: hook logs never touch stdout, because Claude Code
parses a hook's stdout as its decision payload. A stray log line there would
corrupt the decision.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
from tools.claude_hooks import logging_setup
from tools.claude_hooks.logging_setup import DEBUG_ENV, debug_enabled, get_logger


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "debug"])
def test_debug_enabled_for_truthy_values(value: str) -> None:
    assert debug_enabled({DEBUG_ENV: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "   "])
def test_debug_disabled_for_other_values(value: str) -> None:
    assert debug_enabled({DEBUG_ENV: value}) is False


def test_debug_disabled_when_unset() -> None:
    assert debug_enabled({}) is False


def test_fallback_logger_emits_json_to_given_stream() -> None:
    stream = io.StringIO()
    logger = get_logger("test.logger", stream=stream)
    logger.info("something_happened", key="value", count=3)
    record = json.loads(stream.getvalue())
    assert record["event"] == "something_happened"
    assert record["level"] == "info"
    assert record["logger"] == "test.logger"
    assert record["key"] == "value"
    assert record["count"] == 3


def test_fallback_logger_suppresses_debug_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEBUG_ENV, raising=False)
    stream = io.StringIO()
    get_logger("t", stream=stream).debug("noisy_detail")
    assert stream.getvalue() == ""


def test_fallback_logger_emits_debug_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEBUG_ENV, "1")
    stream = io.StringIO()
    get_logger("t", stream=stream).debug("noisy_detail")
    assert "noisy_detail" in stream.getvalue()


@pytest.mark.parametrize("level", ["info", "warning", "error"])
def test_fallback_logger_levels_are_recorded(level: str) -> None:
    stream = io.StringIO()
    getattr(get_logger("t", stream=stream), level)("evt")
    assert json.loads(stream.getvalue())["level"] == level


def test_fallback_logger_handles_unserialisable_fields() -> None:
    stream = io.StringIO()
    get_logger("t", stream=stream).info("evt", obj=object())
    # default=str keeps the record renderable rather than raising.
    assert "evt" in stream.getvalue()


def test_default_logger_never_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # Whichever backend is active, stdout must stay pristine.
    get_logger("stdout.guard").warning("hook_event", detail="x")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_structlog_backend_is_used_when_available() -> None:
    logger = get_logger("structlog.probe")
    # structlog is a hard dependency of this repo, so the real backend should
    # be selected rather than the fallback.
    assert not isinstance(logger, logging_setup._FallbackLogger)


def test_fallback_used_when_structlog_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "structlog", None)
    assert logging_setup._build_structlog_logger("x") is None


def test_structlog_configuration_is_not_mutated_globally() -> None:
    import structlog

    before = structlog.get_config()["logger_factory"]
    get_logger("no.global.mutation").info("evt")
    assert structlog.get_config()["logger_factory"] is before
