"""Honest tick-level tests for CloudLoggingSink (F-039).

``tick()`` emits ``_log.debug("tick_complete")``. Default
``LoggingConfig.level`` and ``GCPLoggingConfig.min_level`` are both INFO, so
structlog drops DEBUG before processors and the sink never sees per-tick
events. Claiming "one overlay away from stalling 30 Hz" is false: the
overlay must also drop **two** log levels to DEBUG.

These tests pin that split. The stall assertion fires only on the both-DEBUG
path, and even then it asserts ``__call__`` itself does not block on a slow
``log_struct`` (the queue owns that).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from mousedroid.cloud.logging_sink import CloudLoggingSink
from mousedroid.config.schema import GCPLoggingConfig, LoggingConfig
from mousedroid.logging.setup import configure_logging, get_logger
from tests.unit.cloud.conftest import _make_gcp_cfg


def _arm_sink(sink: CloudLoggingSink) -> MagicMock:
    mock_logger = MagicMock()
    sink._started = True
    sink._cloud_logger = mock_logger
    return mock_logger


def test_default_info_levels_drop_tick_complete_before_sink() -> None:
    """Default INFO + sink min_level INFO: debug tick never reaches Cloud Logging."""
    sink = CloudLoggingSink(_make_gcp_cfg())
    mock_logger = _arm_sink(sink)
    configure_logging(
        LoggingConfig(level="INFO", format="json"),
        cloud_logging_sink=sink,
    )
    get_logger("mousedroid.orchestrator.orchestrator").debug(
        "tick_complete",
        loop_time_ms=5.0,
        emergency=False,
    )
    sink.flush()
    mock_logger.log_struct.assert_not_called()


def test_both_debug_levels_forward_tick_complete() -> None:
    """The footgun overlay: both levels DEBUG, debug tick does reach the sink."""
    sink = CloudLoggingSink(
        _make_gcp_cfg(logging=GCPLoggingConfig(min_level="DEBUG")),
    )
    mock_logger = _arm_sink(sink)
    configure_logging(
        LoggingConfig(level="DEBUG", format="json"),
        cloud_logging_sink=sink,
    )
    get_logger("mousedroid.orchestrator.orchestrator").debug(
        "tick_complete",
        loop_time_ms=5.0,
        emergency=False,
    )
    mock_logger.log_struct.assert_not_called()
    sink.flush()
    mock_logger.log_struct.assert_called_once()
    entry = mock_logger.log_struct.call_args[0][0]
    assert entry["message"] == "tick_complete"
    assert entry["loop_time_ms"] == 5.0
    assert entry["emergency"] is False


def test_call_does_not_block_when_log_struct_is_slow() -> None:
    """Even on the both-DEBUG overlay, ``__call__`` must not wait on the SDK."""
    sink = CloudLoggingSink(
        _make_gcp_cfg(logging=GCPLoggingConfig(min_level="DEBUG")),
    )
    mock_logger = MagicMock()
    mock_logger.log_struct.side_effect = lambda *_a, **_k: time.sleep(0.3)
    sink._started = True
    sink._cloud_logger = mock_logger

    started = time.perf_counter()
    sink(None, "debug", {"event": "tick_complete", "loop_time_ms": 1.0, "emergency": False})
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05, elapsed
    mock_logger.log_struct.assert_not_called()
    sink.flush()
    mock_logger.log_struct.assert_called_once()
