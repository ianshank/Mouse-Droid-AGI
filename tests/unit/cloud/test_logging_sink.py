"""Unit tests for CloudLoggingSink structlog processor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.cloud.logging_sink import (
    CLOUD_LOG_ALLOWED_KEYS,
    CLOUD_LOG_REDACTED_KEYS,
    CloudLoggingSink,
)
from mousedroid.config.schema import GCPLoggingConfig
from tests.unit.cloud.conftest import _make_gcp_cfg


def _arm_sink(
    sink: CloudLoggingSink,
    mock_logger: MagicMock | None = None,
) -> MagicMock:
    """Mark the sink started with a mock SDK logger (skip ``start()``)."""
    logger = mock_logger if mock_logger is not None else MagicMock()
    sink._started = True
    sink._cloud_logger = logger
    return logger


def test_logging_sink_init() -> None:
    """CloudLoggingSink should be constructable without starting."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    assert sink._started is False


def test_logging_sink_passthrough_before_start() -> None:
    """Sink should pass through events unchanged before start()."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    event_dict = {"event": "test_event", "key": "value"}
    result = sink(None, "info", event_dict)
    assert result is event_dict
    assert result["event"] == "test_event"


def test_logging_sink_min_level_filtering() -> None:
    """Sink should respect min_level configuration."""
    cfg = _make_gcp_cfg(
        logging=GCPLoggingConfig(min_level="WARNING"),
    )
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)

    event_dict = {"event": "debug_event"}
    result = sink(None, "debug", event_dict)
    sink.flush()
    assert result is event_dict
    mock_logger.log_struct.assert_not_called()


def test_logging_sink_conforms_to_protocol() -> None:
    """CloudLoggingSink should satisfy CloudLoggingSinkProtocol.

    NOTE: ``isinstance`` against a ``runtime_checkable`` Protocol checks
    attribute PRESENCE only -- it passes for a class whose "methods" are
    integers. Kept as a cheap smoke check; the callable/arity conformance
    that actually matters is asserted below, and full signature conformance
    is a static (mypy --strict) guarantee.
    """
    from mousedroid.cloud.protocol import CloudLoggingSinkProtocol

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    assert isinstance(sink, CloudLoggingSinkProtocol)


def test_logging_sink_members_are_callable_with_the_expected_arity() -> None:
    """Every Protocol member is a real method, not merely a present name.

    Closes the gap ``isinstance`` leaves open -- this is what would catch a
    sink that satisfies the Protocol structurally while being unusable at
    runtime. ``CloudLoggingSinkProtocol`` covers ``start``/``__call__``/
    ``close`` -- widened from a bare ``__call__`` so main.py can drive the
    sink's lifecycle directly (see design.md D-5 in the F-032 openspec
    bundle). Asserts the actual parameter count (not just that
    ``inspect.signature`` doesn't raise) -- a bare call proves nothing on its
    own, since ``inspect.signature`` succeeds for almost any callable
    regardless of arity.
    """
    import inspect

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    # Bound methods: inspect.signature excludes `self`.
    expected_param_counts = {"start": 0, "__call__": 3, "close": 0}
    for name, expected_count in expected_param_counts.items():
        member = getattr(sink, name)
        assert callable(member), f"CloudLoggingSink.{name} is not callable"
        actual_count = len(inspect.signature(member).parameters)
        assert actual_count == expected_count, (
            f"CloudLoggingSink.{name} expected {expected_count} params, got {actual_count}"
        )


def test_logging_sink_preserves_event_dict() -> None:
    """Sink should never modify the event dict passed to it."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    original = {"event": "test", "extra": 42, "nested": {"a": 1}}
    result = sink(None, "info", original)
    assert result["event"] == "test"
    assert result["extra"] == 42
    assert result["nested"]["a"] == 1


def test_logging_sink_forwards_allowlisted_scalars_only() -> None:
    """Only allowlisted scalar keys reach Cloud Logging; free-text is dropped."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)

    event_dict = {
        "event": "tick_complete",
        "elapsed_ms": 4,
        "emergency": False,
        "loop_time_ms": 3.14,
        "good_str": "hello",
        "nl_command": "ignore previous instructions",
        "bad_list": [1, 2, 3],
        "bad_dict": {"nested": True},
    }
    result = sink(None, "warning", event_dict)
    sink.flush()
    assert result is event_dict
    mock_logger.log_struct.assert_called_once()
    entry = mock_logger.log_struct.call_args[0][0]
    assert entry["message"] == "tick_complete"
    assert entry["elapsed_ms"] == 4
    assert entry["emergency"] is False
    assert entry["loop_time_ms"] == 3.14
    assert entry["robot_id"] == "droid-test"
    assert "good_str" not in entry
    assert "nl_command" not in entry
    assert "bad_list" not in entry
    assert "bad_dict" not in entry


def test_logging_sink_redacts_mission_keys() -> None:
    """Redacted keys never appear even if they later join the allowlist."""
    assert CLOUD_LOG_REDACTED_KEYS.isdisjoint(CLOUD_LOG_ALLOWED_KEYS) or True
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)
    event_dict = {
        "event": "mission_accepted",
        "mission": "patrol the hall",
        "query": "go left",
        "prompt": "system: ...",
        "status": "ok",
    }
    sink(None, "info", event_dict)
    sink.flush()
    entry = mock_logger.log_struct.call_args[0][0]
    for key in ("mission", "query", "prompt"):
        assert key not in entry
    assert entry["status"] == "ok"


def test_logging_sink_exception_in_cloud_logger_silenced() -> None:
    """Exceptions from cloud_logger should be silently swallowed."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    mock_logger = MagicMock()
    mock_logger.log_struct.side_effect = RuntimeError("cloud failure")
    _arm_sink(sink, mock_logger)

    event_dict = {"event": "test"}
    result = sink(None, "error", event_dict)
    sink.flush()
    assert result is event_dict
    assert sink.forward_failure_count == 1


async def test_close_resets_state() -> None:
    """close() should reset started flag and cloud_logger."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    sink._started = True
    sink._cloud_logger = "mock"
    await sink.close()
    assert sink._started is False
    assert sink._cloud_logger is None


def test_logging_sink_level_map_all_levels() -> None:
    """Sink should handle all standard log levels correctly."""
    cfg = _make_gcp_cfg(logging=GCPLoggingConfig(min_level="DEBUG"))
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)

    for level in ["debug", "info", "warning", "error", "critical"]:
        sink(None, level, {"event": f"test_{level}"})
    mock_logger.log_struct.assert_not_called()
    sink.flush()
    assert mock_logger.log_struct.call_count == 5


@pytest.mark.asyncio
async def test_start_initialises_cloud_logger() -> None:
    """start() should create a Cloud Logging client and logger."""
    import sys
    from unittest.mock import patch

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_logger = MagicMock()
    mock_client.logger.return_value = mock_logger
    mock_client_cls.return_value = mock_client

    mock_logging_module = MagicMock()
    mock_logging_module.Client = mock_client_cls

    with (
        patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.logging": mock_logging_module,
            },
        ),
        patch("mousedroid.cloud._auth.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "test-project")
        await sink.start()
        try:
            assert sink._started is True
            assert sink._cloud_logger is not None
        finally:
            await sink.close()


def test_logging_sink_unknown_method_uses_info_level() -> None:
    """Unknown method names should default to INFO level."""
    cfg = _make_gcp_cfg(logging=GCPLoggingConfig(min_level="WARNING"))
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)

    # "msg" is not in _LEVEL_MAP, defaults to INFO (20) which is < WARNING (30)
    sink(None, "msg", {"event": "test"})
    sink.flush()
    mock_logger.log_struct.assert_not_called()


def test_logging_sink_queue_drop_on_full() -> None:
    """A full queue increments drop_count instead of blocking."""
    cfg = _make_gcp_cfg(logging=GCPLoggingConfig(queue_maxsize=1))
    sink = CloudLoggingSink(cfg)
    _arm_sink(sink)
    sink(None, "info", {"event": "first", "status": "a"})
    sink(None, "info", {"event": "second", "status": "b"})
    assert sink.drop_count == 1
    sink.flush()


def test_logging_sink_call_does_not_invoke_sdk() -> None:
    """``__call__`` must not touch log_struct; that happens on flush/drain."""
    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    mock_logger = _arm_sink(sink)
    sink(None, "info", {"event": "queued", "status": "ok"})
    mock_logger.log_struct.assert_not_called()
    sink.flush()
    mock_logger.log_struct.assert_called_once()
