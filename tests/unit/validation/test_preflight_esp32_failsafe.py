"""Unit tests for the ``esp32_failsafe`` preflight check.

``ESP32Config.heartbeat_enabled`` defaults ``True`` and ``heartbeat_window_ms``
resolves under every command set, so config reads as "failsafe on, 3 s window"
even when the resolved codec arms nothing. Only ``waveshare_stock`` emits
``CMD_HEART_BEAT_SET``. These pin that the check tells the truth in each state,
because this is the only mechanism that halts the *wheels* when the host wedges.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.preflight import PreflightStatus, _check_esp32_failsafe

# Settings rejects mock_hardware=false without a distance sensor.
_REAL_HARDWARE: dict[str, object] = {"mock_hardware": False, "lidar": {"enabled": True}}


def _cfg(**esp32: object) -> Settings:
    data: dict[str, object] = dict(_REAL_HARDWARE)
    if esp32:
        data["esp32"] = esp32
    return Settings.model_validate(data)


@pytest.mark.asyncio
async def test_legacy_command_set_warns_that_the_failsafe_is_not_armed() -> None:
    """The shipped default: config says on, nothing is armed."""
    result = await _check_esp32_failsafe(_cfg())
    assert result.status is PreflightStatus.WARN
    assert "NOT armed" in result.detail


@pytest.mark.asyncio
async def test_warning_names_the_exact_remedy() -> None:
    """An operator must not have to go read the codec to learn the fix."""
    result = await _check_esp32_failsafe(_cfg())
    assert "MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock" in result.detail


@pytest.mark.asyncio
async def test_stock_command_set_reports_armed() -> None:
    """``waveshare_stock`` sends ``CMD_HEART_BEAT_SET`` at connect."""
    result = await _check_esp32_failsafe(_cfg(command_set="waveshare_stock"))
    assert result.status is PreflightStatus.OK
    assert "armed at connect" in result.detail


@pytest.mark.asyncio
async def test_explicitly_disabled_failsafe_still_warns() -> None:
    """A deliberate opt-out is still 'nothing stops the wheels'.

    Distinguished from the dormant case by wording so an operator can tell a
    choice from an accident.
    """
    result = await _check_esp32_failsafe(
        _cfg(command_set="waveshare_stock", heartbeat_enabled=False)
    )
    assert result.status is PreflightStatus.WARN
    assert "deliberately disabled" in result.detail


@pytest.mark.asyncio
async def test_mock_hardware_short_circuits_but_still_reports_state() -> None:
    """No chassis attached means the question does not apply.

    Matches the device checks' mock convention, but the detail still carries
    the resolved config so a mock run is not silently uninformative.
    """
    result = await _check_esp32_failsafe(Settings(mock_hardware=True))
    assert result.status is PreflightStatus.OK
    assert "mock_hardware=true" in result.detail
    assert "command_set=" in result.detail


@pytest.mark.asyncio
async def test_check_never_opens_the_serial_port() -> None:
    """A config question, not a device probe — it must not build a driver.

    ``_check_esp32`` already constructs a driver; if this check did too it
    would double the port-open risk on a live rover for no added signal.
    """
    import mousedroid.factory as factory_mod

    calls: list[object] = []
    original = factory_mod.build_esp32_driver

    def _spy(cfg: Settings) -> object:  # pragma: no cover - must not run
        calls.append(cfg)
        return original(cfg)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory_mod, "build_esp32_driver", _spy)
        await _check_esp32_failsafe(_cfg())
    assert calls == []


@pytest.mark.asyncio
async def test_result_is_named_for_the_dispatch_key() -> None:
    """The report entry must be addressable by ``--checks esp32_failsafe``."""
    result = await _check_esp32_failsafe(_cfg())
    assert result.name == "esp32_failsafe"
    assert result.elapsed_s >= 0.0
