"""Unit tests for the MSE-6 greeting startup seam (Issue #109).

The orchestrator fires the greeting ONCE during ``start()`` — before
entering the 30 Hz loop — iff ``cfg.greeting`` is enabled with
``fire_on_startup=True`` AND a greeter was wired. A greeting failure is
logged (``greeting_startup_failed``) and swallowed so it never blocks
startup; a success emits ``greeting_startup_complete``.

These tests drive ``start()`` with ``AsyncMock`` subsystem stand-ins and
a fake greeter so no real audio / hardware is touched, mirroring the
construction discipline of ``test_orchestrator_face.py``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing

from mousedroid.config.schema import GreetingConfig, Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mousedroid.voice.greeting import GreeterProtocol


class _FakeGreeter:
    """Records ``greet`` calls without driving audio."""

    def __init__(self, *, raises: bool = False) -> None:
        self.calls: list[Sequence[str] | None] = []
        self._raises = raises

    async def greet(self, names: Sequence[str] | None = None) -> None:
        self.calls.append(names)
        if self._raises:
            msg = "synthesis exploded"
            raise RuntimeError(msg)


def _make_orch(
    *,
    greeting: GreetingConfig | None,
    greeter: GreeterProtocol | None,
) -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True, greeting=greeting)
    return MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        greeter=greeter,
    )


@pytest.mark.asyncio
async def test_start_fires_greeting_when_enabled_and_flag_set() -> None:
    greeter = _FakeGreeter()
    cfg = GreetingConfig(enabled=True, names=["John"], fire_on_startup=True)
    orch = _make_orch(greeting=cfg, greeter=greeter)
    with structlog.testing.capture_logs() as logs:
        await orch.start()
    assert greeter.calls == [None], "greeting must fire exactly once with no override"
    events = [e["event"] for e in logs]
    assert "greeting_startup_complete" in events
    assert orch._running is True


@pytest.mark.asyncio
async def test_start_does_not_fire_when_flag_false() -> None:
    greeter = _FakeGreeter()
    cfg = GreetingConfig(enabled=True, names=["John"], fire_on_startup=False)
    orch = _make_orch(greeting=cfg, greeter=greeter)
    await orch.start()
    assert greeter.calls == [], "greeting must not fire when fire_on_startup=False"


@pytest.mark.asyncio
async def test_start_does_not_fire_when_greeting_disabled() -> None:
    greeter = _FakeGreeter()
    # enabled=False; fire_on_startup is irrelevant.
    cfg = GreetingConfig(enabled=False, fire_on_startup=True)
    orch = _make_orch(greeting=cfg, greeter=greeter)
    await orch.start()
    assert greeter.calls == []


@pytest.mark.asyncio
async def test_start_does_not_fire_when_no_greeter_wired() -> None:
    cfg = GreetingConfig(enabled=True, names=["John"], fire_on_startup=True)
    orch = _make_orch(greeting=cfg, greeter=None)
    # Must not raise even though the flag is on but no greeter is present.
    await orch.start()
    assert orch._running is True


@pytest.mark.asyncio
async def test_start_does_not_fire_when_greeting_config_none() -> None:
    greeter = _FakeGreeter()
    orch = _make_orch(greeting=None, greeter=greeter)
    await orch.start()
    assert greeter.calls == []


@pytest.mark.asyncio
async def test_greeting_failure_never_blocks_startup() -> None:
    greeter = _FakeGreeter(raises=True)
    cfg = GreetingConfig(enabled=True, names=["John"], fire_on_startup=True)
    orch = _make_orch(greeting=cfg, greeter=greeter)
    with structlog.testing.capture_logs() as logs:
        await orch.start()  # must NOT raise
    assert greeter.calls == [None]
    events = [e["event"] for e in logs]
    assert "greeting_startup_failed" in events
    # Startup completed despite the greeting failure.
    assert orch._running is True


@pytest.mark.asyncio
async def test_greeting_hang_is_bounded_and_never_blocks_startup() -> None:
    """A hung TTS/ALSA greeting is abandoned at ``startup_timeout_s``.

    Pins the Issue-#109 review fix: ``greet()`` is wrapped in
    ``asyncio.wait_for`` so an indefinitely-blocking greeting can never wedge
    the orchestrator before its 30 Hz loop starts. The dual-catch
    ``except (TimeoutError, asyncio.TimeoutError)`` makes the precise
    ``greeting_startup_timeout`` event fire on EVERY supported interpreter —
    including py3.10, where ``asyncio.TimeoutError`` is a distinct class
    (CodeRabbit #3409673687).
    """

    class _HangingGreeter:
        async def greet(self, names: Sequence[str] | None = None) -> None:
            await asyncio.sleep(3600)  # far longer than the configured timeout

    cfg = GreetingConfig(
        enabled=True,
        names=["John"],
        fire_on_startup=True,
        startup_timeout_s=0.01,  # config-driven bound (not a hardcoded literal)
    )
    orch = _make_orch(greeting=cfg, greeter=_HangingGreeter())
    with structlog.testing.capture_logs() as logs:
        await orch.start()  # must return promptly, not hang
    events = [e["event"] for e in logs]
    assert "greeting_startup_timeout" in events
    assert "greeting_startup_failed" not in events
    assert "greeting_startup_complete" not in events
    assert orch._running is True


@pytest.mark.asyncio
async def test_startup_greeting_propagates_cancellation() -> None:
    """Cooperative cancellation during the greeting is never swallowed.

    ``asyncio.CancelledError`` subclasses ``BaseException`` (not
    ``Exception``) on every supported interpreter, and the greeting path
    keeps an explicit defensive re-raise so cancellation always propagates
    (Copilot #3409950174) — otherwise a caller can't cancel bring-up. The
    greeting MUST NOT be reported as a mere ``greeting_startup_failed``.
    """

    class _CancellingGreeter:
        async def greet(self, names: Sequence[str] | None = None) -> None:
            raise asyncio.CancelledError

    cfg = GreetingConfig(enabled=True, names=["John"], fire_on_startup=True)
    orch = _make_orch(greeting=cfg, greeter=_CancellingGreeter())
    with structlog.testing.capture_logs() as logs, pytest.raises(asyncio.CancelledError):
        await orch.start()
    events = [e["event"] for e in logs]
    assert "greeting_startup_failed" not in events
    assert "greeting_startup_complete" not in events


@pytest.mark.asyncio
async def test_greeter_defaults_to_none() -> None:
    """The greeter kwarg is keyword-only with a None default (legacy parity)."""
    cfg = Settings(mock_hardware=True)
    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
    )
    assert orch._greeter is None
