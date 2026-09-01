"""Integration tests for the PR #104 ``esp32.enabled=False`` factory branch.

These tests exercise the wiring path end-to-end through the
:func:`mousedroid.factory.build_esp32_driver` factory — not just the unit-level
``isinstance`` check covered by ``tests/unit/test_factory.py``. The goal is to
prove that the schema-driven dev-escape-hatch from PR #104 actually short-
circuits ``connect()`` / ``send_velocity()`` / ``emergency_stop()`` so the
orchestrator can boot on a Jetson without an ESP32 plugged in.

Design notes:

* No hardcoded magic numbers — everything is derived from the same
  ``Settings`` instance the factory uses.
* No real serial / WiFi I/O — ``cfg.esp32.enabled=False`` is the entire point;
  the resolved inner driver is an in-memory ``MockESP32Driver`` regardless of
  the configured protocol.
* The tests bracket the contract surface PR #104 cares about: the resolved
  driver responds to the ESP32CommProtocol methods without raising, even when
  ``mock_hardware=False`` (real-hardware mode minus a missing ESP32).
"""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.comms.protocol import ESP32CommProtocol as _ESP32Proto  # noqa: F401
from mousedroid.config.schema import Settings
from mousedroid.factory import build_esp32_driver
from mousedroid.resilience.resilient_driver import ResilientESP32Driver


def _settings_with_esp32_disabled(*, mock_hardware: bool, protocol: str = "serial") -> Settings:
    """Build Settings exercising the ``esp32.enabled=False`` PR-104 escape hatch.

    Wraps :class:`Settings` construction so each test states its scenario
    in one line. ``mock_hardware`` is varied to prove that the
    ``esp32.enabled`` toggle takes effect EVEN WHEN the global
    ``mock_hardware`` switch is False — that was the explicit gap PR #104
    closed (running with real cameras + LiDAR + Hailo but no ESP32).
    """
    return Settings.model_validate(
        {
            # When ``mock_hardware=False`` the root-level Settings validator
            # also requires at least one distance sensor wired in — we declare
            # the ultrasonic so the validator's invariant is satisfied without
            # affecting any code path under test (the factory only consults
            # ``cfg.esp32.enabled`` here). UltrasonicConfig has no ``enabled``
            # field of its own — presence of ``cfg.ultrasonic`` (non-None) is
            # what "wires it in"; ``trigger_pin``/``echo_pin`` are its only
            # required fields.
            "mock_hardware": mock_hardware,
            "ultrasonic": {"trigger_pin": 23, "echo_pin": 24},
            "esp32": {
                "enabled": False,
                "protocol": protocol,
                "serial_port": "/dev/ttyDOES_NOT_EXIST",
                "wifi_host": "10.255.255.1",
            },
        }
    )


def test_build_esp32_returns_resilient_wrapper_when_disabled() -> None:
    """Even with ``enabled=False``, the factory still wraps in ResilientESP32Driver.

    This is the wrapper-preservation invariant: PR #104 only changes which
    *inner* driver is selected — the resilience layer (retry + circuit-
    breaker) must still be in place so the orchestrator's existing error-
    handling assumptions hold.
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=False)
    driver = build_esp32_driver(cfg)
    assert isinstance(driver, ResilientESP32Driver)


def test_disabled_with_real_hardware_mode_uses_mock_inner_driver() -> None:
    """``esp32.enabled=False`` + ``mock_hardware=False`` → still MockESP32Driver.

    The whole point of the PR-104 toggle: an operator running the live
    Jetson dashboard with no ESP32 plugged in shouldn't have to flip the
    global ``mock_hardware`` switch (which would also stub out the camera +
    LiDAR + Hailo). This test pins that behaviour.
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=False)
    driver = build_esp32_driver(cfg)
    # ResilientESP32Driver wraps an `_inner` driver — verify it's the mock.
    inner = getattr(driver, "_inner", None) or getattr(driver, "_driver", None)
    assert isinstance(inner, MockESP32Driver), (
        f"Expected MockESP32Driver when esp32.enabled=False, got {type(inner).__name__}"
    )


def test_disabled_with_mock_hardware_still_uses_mock_inner_driver() -> None:
    """``esp32.enabled=False`` + ``mock_hardware=True`` → still MockESP32Driver.

    Belt-and-suspenders: confirms the two toggles compose without
    surprises. Either alone produces a mock; both together also produce a
    mock (rather than e.g. some union type or a logic-OR bug).
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=True)
    driver = build_esp32_driver(cfg)
    inner = getattr(driver, "_inner", None) or getattr(driver, "_driver", None)
    assert isinstance(inner, MockESP32Driver)


@pytest.mark.parametrize("protocol", ["serial", "wifi"])
def test_disabled_overrides_protocol_choice(protocol: str) -> None:
    """``esp32.enabled=False`` short-circuits BEFORE the protocol branch.

    Confirms the factory's ``if cfg.mock_hardware or not cfg.esp32.enabled``
    check runs in front of the serial-vs-wifi dispatch — even an explicit
    ``protocol: wifi`` config can't override it. This guards against a
    future refactor that accidentally reorders the conditionals.
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=False, protocol=protocol)
    driver = build_esp32_driver(cfg)
    inner = getattr(driver, "_inner", None) or getattr(driver, "_driver", None)
    assert isinstance(inner, MockESP32Driver)


@pytest.mark.asyncio
async def test_disabled_driver_connect_and_send_dont_raise() -> None:
    """The mock-when-disabled inner driver satisfies the ESP32CommProtocol contract.

    The orchestrator's start sequence calls ``connect()`` then issues
    velocity setpoints + occasional emergency stops. Every one of those
    must succeed without raising — that's the entire reason PR #104
    exists. We exercise each surface here in one shot.
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=False)
    driver = build_esp32_driver(cfg)

    # The methods we exercise are part of the ResilientESP32Driver public
    # surface. None should raise when wrapping the mock.
    await driver.connect()
    await driver.send_velocity(0.0, 0.0, 0.0)
    # Even a non-zero command (would be a real motor pulse on real HW) must
    # be safe through the mock.
    await driver.send_velocity(cfg.esp32.smoke_test_velocity_mps, 0.0, 0.0)
    await driver.emergency_stop()
    await driver.disconnect()


def test_default_settings_keep_esp32_enabled_true() -> None:
    """Default Settings (no overrides) still has ``esp32.enabled=True``.

    Backwards-compat guard: a YAML / env config that DOESN'T mention
    ``esp32.enabled`` must inherit the legacy "yes the ESP32 is real" default
    — otherwise rover deployments after a `git pull` would silently switch
    to the mock driver and stop moving.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.enabled is True


def test_concurrent_send_velocity_through_disabled_driver() -> None:
    """Multiple concurrent ``send_velocity`` calls through the disabled driver succeed.

    Stress the mock path with a small fan-out — the orchestrator's
    pipelined sense-plan-act loop issues velocity commands from multiple
    coroutines (planner + safety monitor). The mock driver must handle
    that without state corruption or RuntimeError.
    """
    cfg = _settings_with_esp32_disabled(mock_hardware=False)
    driver = build_esp32_driver(cfg)

    async def _drive() -> None:
        await driver.connect()
        # Use a per-test fan-out of N=8 — small enough to stay fast, large
        # enough to flush out any "first caller wins" bugs in the mock.
        n = 8
        await asyncio.gather(*(driver.send_velocity(float(i) * 0.01, 0.0, 0.0) for i in range(n)))
        await driver.emergency_stop()
        await driver.disconnect()

    asyncio.run(_drive())
