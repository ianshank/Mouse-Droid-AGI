"""F-009 regression: ``build_tensorrt_compiler`` observability.

The smoke-stability sprint surfaced that the factory used two separate
structured-log events (``tensorrt_compiler_built`` for real,
``tensorrt_compiler_mock_built`` for the disabled branch). Operator dashboards
that want to monitor "what TensorRT backend is in use" had to merge the two
event types and infer mock/real from the event name. The fix consolidates
both branches under one event name with an explicit ``backend`` label.

Additionally, the real branch now emits ``torch2trt_available`` at build
time. The previous code only logged the torch2trt-vs-jit-trace fallback
on the **first compile call** (``torch2trt_not_available_falling_back_to_jit_trace``
in efficiency/tensorrt.py:231), which means a Jetson that never hits a
compile path silently runs without a real TRT engine and nobody notices.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog

from mousedroid.config.schema import Settings
from mousedroid.efficiency.tensorrt import JetsonTensorRTCompiler, MockTensorRTCompiler
from mousedroid.factory import build_tensorrt_compiler


@pytest.fixture
def captured_logs() -> list[dict[str, Any]]:
    """Capture structlog events into a list for assertion."""
    captured: list[dict[str, Any]] = []

    def _capture(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(event_dict)
        return event_dict

    # Insert the capture processor at the head of the chain so we see every event.
    structlog.configure(
        processors=[_capture, structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(0),  # accept all levels
        cache_logger_on_first_use=False,
    )
    yield captured
    structlog.reset_defaults()


def _find_tensorrt_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pluck the single ``tensorrt_compiler_built`` event out of the captured list."""
    matches = [e for e in events if e.get("event") == "tensorrt_compiler_built"]
    assert (
        len(matches) == 1
    ), f"expected exactly one tensorrt_compiler_built event, got {len(matches)}: {events!r}"
    return matches[0]


def test_build_returns_real_compiler_when_tensorrt_enabled(
    captured_logs: list[dict[str, Any]],
) -> None:
    """``tensorrt_enabled=True`` returns ``JetsonTensorRTCompiler`` + logs backend=real."""
    cfg = Settings(mock_hardware=True)
    cfg.jetson.tensorrt_enabled = True

    compiler = build_tensorrt_compiler(cfg)

    assert isinstance(compiler, JetsonTensorRTCompiler)
    event = _find_tensorrt_event(captured_logs)
    assert event["backend"] == "real"
    assert event["reason"] == "tensorrt_enabled=true"
    # torch2trt availability should appear in the event regardless of value
    # (False on dev hosts, True on Jetson production image).
    assert "torch2trt_available" in event


def test_build_returns_mock_compiler_when_tensorrt_disabled(
    captured_logs: list[dict[str, Any]],
) -> None:
    """``tensorrt_enabled=False`` returns ``MockTensorRTCompiler`` + logs backend=mock."""
    cfg = Settings(mock_hardware=True)
    cfg.jetson.tensorrt_enabled = False

    compiler = build_tensorrt_compiler(cfg)

    assert isinstance(compiler, MockTensorRTCompiler)
    event = _find_tensorrt_event(captured_logs)
    assert event["backend"] == "mock"
    assert event["reason"] == "tensorrt_enabled=false"
    assert event["torch2trt_available"] is False


def test_single_log_event_name_across_both_branches(
    captured_logs: list[dict[str, Any]],
) -> None:
    """Both branches use the SAME event name with different labels (F-009 ask)."""
    cfg_real = Settings(mock_hardware=True)
    cfg_real.jetson.tensorrt_enabled = True
    cfg_mock = Settings(mock_hardware=True)
    cfg_mock.jetson.tensorrt_enabled = False

    build_tensorrt_compiler(cfg_real)
    build_tensorrt_compiler(cfg_mock)

    backends = [e["backend"] for e in captured_logs if e.get("event") == "tensorrt_compiler_built"]
    assert backends == [
        "real",
        "mock",
    ], f"expected same event name across both branches; got backends={backends!r}"
    # The old ``tensorrt_compiler_mock_built`` event name MUST be gone (operators
    # may have dashboards alerting on it; the rename is a one-time consolidation).
    legacy = [e for e in captured_logs if e.get("event") == "tensorrt_compiler_mock_built"]
    assert legacy == [], "legacy tensorrt_compiler_mock_built event must not be emitted"
