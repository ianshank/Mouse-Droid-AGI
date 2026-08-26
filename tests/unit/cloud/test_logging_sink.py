"""Unit tests for CloudLoggingSink structlog processor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import GCPLoggingConfig
from tests.unit.cloud.conftest import _make_gcp_cfg


def test_logging_sink_init() -> None:
    """CloudLoggingSink should be constructable without starting."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    assert sink._started is False


def test_logging_sink_passthrough_before_start() -> None:
    """Sink should pass through events unchanged before start()."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    event_dict = {"event": "test_event", "key": "value"}
    result = sink(None, "info", event_dict)
    assert result is event_dict
    assert result["event"] == "test_event"


def test_logging_sink_min_level_filtering() -> None:
    """Sink should respect min_level configuration."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg(
        logging=GCPLoggingConfig(min_level="WARNING"),
    )
    sink = CloudLoggingSink(cfg)

    # Debug events should be below min_level threshold
    event_dict = {"event": "debug_event"}
    result = sink(None, "debug", event_dict)
    assert result is event_dict


def test_logging_sink_conforms_to_protocol() -> None:
    """CloudLoggingSink should satisfy CloudLoggingSinkProtocol.

    NOTE: ``isinstance`` against a ``runtime_checkable`` Protocol checks
    attribute PRESENCE only -- it passes for a class whose "methods" are
    integers. Kept as a cheap smoke check; the callable/arity conformance
    that actually matters is asserted below, and full signature conformance
    is a static (mypy --strict) guarantee.
    """
    from mousedroid.cloud.logging_sink import CloudLoggingSink
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
    bundle).
    """
    import inspect

    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    for name in ("start", "__call__", "close"):
        member = getattr(sink, name)
        assert callable(member), f"CloudLoggingSink.{name} is not callable"
        inspect.signature(member)


def test_logging_sink_preserves_event_dict() -> None:
    """Sink should never modify the event dict passed to it."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    original = {"event": "test", "extra": 42, "nested": {"a": 1}}
    result = sink(None, "info", original)
    assert result["event"] == "test"
    assert result["extra"] == 42
    assert result["nested"]["a"] == 1


def test_logging_sink_filters_non_serializable_values() -> None:
    """Sink should only forward str/int/float/bool/None values to cloud."""
    from unittest.mock import MagicMock

    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    sink._started = True
    mock_logger = MagicMock()
    sink._cloud_logger = mock_logger

    event_dict = {
        "event": "test",
        "good_str": "hello",
        "good_int": 42,
        "good_float": 3.14,
        "good_bool": True,
        "good_none": None,
        "bad_list": [1, 2, 3],
        "bad_dict": {"nested": True},
    }
    result = sink(None, "warning", event_dict)
    assert result is event_dict
    mock_logger.log_struct.assert_called_once()
    entry = mock_logger.log_struct.call_args[0][0]
    assert "good_str" in entry
    assert "good_int" in entry
    assert "bad_list" not in entry
    assert "bad_dict" not in entry


def test_logging_sink_exception_in_cloud_logger_silenced() -> None:
    """Exceptions from cloud_logger should be silently swallowed."""
    from unittest.mock import MagicMock

    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    sink._started = True
    mock_logger = MagicMock()
    mock_logger.log_struct.side_effect = RuntimeError("cloud failure")
    sink._cloud_logger = mock_logger

    event_dict = {"event": "test"}
    result = sink(None, "error", event_dict)
    assert result is event_dict


async def test_close_resets_state() -> None:
    """close() should reset started flag and cloud_logger."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    sink._started = True
    sink._cloud_logger = "mock"
    await sink.close()
    assert sink._started is False
    assert sink._cloud_logger is None


def test_logging_sink_level_map_all_levels() -> None:
    """Sink should handle all standard log levels correctly."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg(logging=GCPLoggingConfig(min_level="DEBUG"))
    sink = CloudLoggingSink(cfg)
    sink._started = True
    sink._cloud_logger = MagicMock()

    for level in ["debug", "info", "warning", "error", "critical"]:
        sink(None, level, {"event": f"test_{level}"})

    assert sink._cloud_logger.log_struct.call_count == 5


@pytest.mark.asyncio
async def test_start_initialises_cloud_logger() -> None:
    """start() should create a Cloud Logging client and logger."""
    import sys
    from unittest.mock import patch

    from mousedroid.cloud.logging_sink import CloudLoggingSink

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

    assert sink._started is True
    assert sink._cloud_logger is not None


def test_logging_sink_unknown_method_uses_info_level() -> None:
    """Unknown method names should default to INFO level."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg(logging=GCPLoggingConfig(min_level="WARNING"))
    sink = CloudLoggingSink(cfg)
    sink._started = True
    sink._cloud_logger = MagicMock()

    # "msg" is not in _LEVEL_MAP, defaults to INFO (20) which is < WARNING (30)
    sink(None, "msg", {"event": "test"})
    sink._cloud_logger.log_struct.assert_not_called()
