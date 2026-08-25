"""Tests for main.py's CloudLoggingSink threading through cli_entry/_run/_health_check.

Genuinely new test territory: ``main.py`` is ``# pragma: no cover`` and
listed in ``pyproject.toml``'s coverage ``omit``, and nothing else in the
tree imports it. ``_run``/``_health_check`` are exercised directly as async
tests, since ``cli_entry()`` calls ``asyncio.run()`` internally and would
raise if invoked from an already-running event loop -- so its own coverage
(below) uses a plain synchronous test function instead, per the plan's own
instruction to pick one approach explicitly rather than leave it ambiguous.

``MouseDroidOrchestrator`` and ``build_orchestrator`` are both patched to a
minimal fake so these tests isolate ``cloud_logging_sink`` wiring from
whether the real orchestrator itself functions -- an unrelated concern
already covered by the orchestrator's own test suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mousedroid.config.schema import Settings
from mousedroid.main import _health_check, _run, cli_entry


class _FakeOrchestrator:
    """Stand-in patched in for MouseDroidOrchestrator's isinstance check."""

    def __init__(self, health_result: dict[str, str] | None = None) -> None:
        self.start = AsyncMock()
        self.run = AsyncMock()
        self.stop = AsyncMock()
        self.health_check = AsyncMock(return_value=health_result or {"status": "ok"})


def _settings() -> Settings:
    return Settings(mock_hardware=True)


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


async def test_run_default_path_touches_nothing_when_sink_is_none() -> None:
    fake = _FakeOrchestrator()
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _run(_settings(), cloud_logging_sink=None)

    fake.start.assert_called_once()
    fake.run.assert_called_once()
    fake.stop.assert_called_once()


async def test_run_starts_and_closes_sink_when_wired() -> None:
    fake = _FakeOrchestrator()
    sink = AsyncMock()
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _run(_settings(), cloud_logging_sink=sink)

    sink.start.assert_called_once()
    sink.close.assert_called_once()
    fake.run.assert_called_once()


async def test_run_survives_sink_start_failure() -> None:
    """An unreachable Cloud Logging backend must never block the 30 Hz loop starting."""
    fake = _FakeOrchestrator()
    sink = AsyncMock()
    sink.start.side_effect = RuntimeError("cloud logging unreachable")
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _run(_settings(), cloud_logging_sink=sink)  # must not raise

    fake.start.assert_called_once()
    fake.run.assert_called_once()
    fake.stop.assert_called_once()
    sink.close.assert_called_once()


async def test_run_survives_sink_close_failure() -> None:
    fake = _FakeOrchestrator()
    sink = AsyncMock()
    sink.close.side_effect = RuntimeError("cloud logging unreachable")
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _run(_settings(), cloud_logging_sink=sink)  # must not raise

    fake.stop.assert_called_once()
    sink.close.assert_called_once()


async def test_run_closes_sink_when_build_orchestrator_raises_after_start_succeeded() -> None:
    """cloud_logging_sink.close() must run even when construction fails downstream.

    Regression case for a real gap: build_orchestrator()/the isinstance check/
    orch_obj.start() all used to execute outside any try/finally, so a failure
    there left an already-started sink never closed even though start()
    succeeded.
    """
    sink = AsyncMock()
    with (
        patch("mousedroid.factory.build_orchestrator", side_effect=RuntimeError("bad config")),
        pytest.raises(RuntimeError, match="bad config"),
    ):
        await _run(_settings(), cloud_logging_sink=sink)

    sink.start.assert_called_once()
    sink.close.assert_called_once()


# ---------------------------------------------------------------------------
# _health_check
# ---------------------------------------------------------------------------


async def test_health_check_default_path_touches_nothing_when_sink_is_none() -> None:
    fake = _FakeOrchestrator(health_result={"status": "ok"})
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _health_check(_settings(), cloud_logging_sink=None)

    fake.health_check.assert_called_once()


async def test_health_check_starts_and_closes_sink_when_wired_and_passing() -> None:
    fake = _FakeOrchestrator(health_result={"status": "ok"})
    sink = AsyncMock()
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _health_check(_settings(), cloud_logging_sink=sink)

    sink.start.assert_called_once()
    sink.close.assert_called_once()


async def test_health_check_closes_sink_even_on_sys_exit() -> None:
    """close() must run on the sys.exit(1) failure branch, not just the happy path."""
    fake = _FakeOrchestrator(health_result={"status": "degraded"})
    sink = AsyncMock()
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
        pytest.raises(SystemExit) as exc_info,
    ):
        await _health_check(_settings(), cloud_logging_sink=sink)

    assert exc_info.value.code == 1
    sink.start.assert_called_once()
    sink.close.assert_called_once()


async def test_health_check_survives_sink_start_failure() -> None:
    fake = _FakeOrchestrator(health_result={"status": "ok"})
    sink = AsyncMock()
    sink.start.side_effect = RuntimeError("cloud logging unreachable")
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _health_check(_settings(), cloud_logging_sink=sink)  # must not raise

    fake.health_check.assert_called_once()
    sink.close.assert_called_once()


async def test_health_check_survives_sink_close_failure() -> None:
    fake = _FakeOrchestrator(health_result={"status": "ok"})
    sink = AsyncMock()
    sink.close.side_effect = RuntimeError("cloud logging unreachable")
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _health_check(_settings(), cloud_logging_sink=sink)  # must not raise

    fake.health_check.assert_called_once()
    sink.close.assert_called_once()


# ---------------------------------------------------------------------------
# cli_entry
# ---------------------------------------------------------------------------


class _FakeLoggingSink:
    """Protocol-conforming fake -- records every event it forwards.

    Used (rather than the real ``CloudLoggingSink``, which reaches for
    ``google.cloud.logging.Client``) to prove two things through
    ``cli_entry()``'s own real, unmocked code path: the *same* instance
    reaches both ``configure_logging()`` and ``_run()``/``_health_check()``,
    and a live log event actually forwards through it -- not just object
    identity.
    """

    def __init__(self) -> None:
        self.start = AsyncMock()
        self.close = AsyncMock()
        self.events: list[dict[str, Any]] = []

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        self.events.append(dict(event_dict))
        return event_dict


def test_cli_entry_threads_one_sink_instance_into_configure_logging_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cli_entry() builds one sink, threads it into configure_logging() AND _run().

    A synchronous test (not ``async def``) since ``cli_entry()`` calls
    ``asyncio.run()`` internally, which raises if invoked from within an
    already-running event loop.
    """
    monkeypatch.setattr("sys.argv", ["mousedroid"])
    fake_sink = _FakeLoggingSink()
    fake_orch = _FakeOrchestrator()

    with (
        patch("mousedroid.main.load_settings", return_value=_settings()),
        patch("mousedroid.factory.build_cloud_logging_sink", return_value=fake_sink),
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake_orch),
    ):
        cli_entry()

    # Reached configure_logging() and forwarded a real, live log event --
    # not just object identity (configure_logging's own event-forwarding
    # correctness is separately covered by tests/unit/logging/test_setup.py;
    # this proves cli_entry()'s wiring reaches it end-to-end).
    events = [e["event"] for e in fake_sink.events]
    assert "mousedroid_starting" in events

    # Reached _run() -- the same instance's lifecycle was driven.
    fake_sink.start.assert_called_once()
    fake_sink.close.assert_called_once()
    fake_orch.run.assert_called_once()
