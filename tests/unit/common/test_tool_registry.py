"""Tests for ToolRegistry — full coverage including builtins."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec, create_default_registry
from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import GCPConfig, Settings
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.validation.runtime import Esp32ProbeResult

#: ``Settings`` refuses ``mock_hardware=false`` without a distance sensor. The
#: ``esp32_diagnostics`` tool never reads it; it only has to make the config valid.
_ULTRASONIC_CFG = {"trigger_pin": 17, "echo_pin": 27}

#: The ``collect_esp32_probe`` seam the tool resolves lazily, at call time.
_PROBE = "mousedroid.validation.runtime.collect_esp32_probe"


async def _dummy_handler() -> str:
    return "ok"


def _real_cfg(**esp32: object) -> Settings:
    """Real-hardware settings on the stock command set; ``esp32`` overrides win."""
    return Settings.model_validate(
        {
            "mock_hardware": False,
            "ultrasonic": _ULTRASONIC_CFG,
            "esp32": {"command_set": "waveshare_stock", **esp32},
        }
    )


def _live_driver(reading: EncoderReading, voltage: float = 11.9) -> AsyncMock:
    """``AsyncMock`` driver with scripted reads; every other method is an unscripted spy."""
    driver = AsyncMock()
    driver.read_encoders.return_value = reading
    driver.get_battery_voltage.return_value = voltage
    return driver


class _ProbeSpy:
    """Stand-in for ``collect_esp32_probe``: records its calls, then returns or raises."""

    def __init__(self, outcome: Esp32ProbeResult | BaseException) -> None:
        self.outcome = outcome
        self.calls: list[Settings] = []

    async def __call__(self, cfg: Settings) -> Esp32ProbeResult:
        self.calls.append(cfg)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _described(reg: ToolRegistry) -> list[tuple[str, str]]:
    """``(name, description)`` for every registered tool, in registration order."""
    return [(name, spec.description) for name in reg.names if (spec := reg.get(name))]


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_constructor(registry: ToolRegistry) -> None:
    assert len(registry) == 0


def test_register_adds_tool(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="test", description="a test tool", handler=_dummy_handler)
    registry.register(spec)
    assert len(registry) == 1
    assert registry.get("test") is not None


def test_register_multiple(registry: ToolRegistry) -> None:
    for i in range(3):
        registry.register(ToolSpec(name=f"tool_{i}", description="", handler=_dummy_handler))
    assert len(registry) == 3


@pytest.mark.asyncio
async def test_dispatch_calls_handler(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="ping", description="ping", handler=_dummy_handler)
    registry.register(spec)
    result = await registry.dispatch("ping")
    assert result == "ok"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="Tool not registered"):
        await registry.dispatch("nonexistent")


def test_create_default_registry_has_11_tools() -> None:
    reg = create_default_registry()
    assert len(reg) == 11


def test_create_default_registry_threads_settings_and_driver_keyword_only() -> None:
    """``None`` is legacy construction: the new seams are keyword-only and default to ``None``.

    The three pre-existing parameters keep their positional slots, so every existing
    caller (``create_default_registry()``, ``create_default_registry(gateway, metrics)``)
    binds exactly as before.
    """
    params = inspect.signature(create_default_registry).parameters
    legacy = ("llm_gateway", "metrics_registry", "gcp_cfg")
    assert tuple(params)[:3] == legacy
    for name in legacy:
        assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[name].default is None
    for name in ("settings", "esp32_driver"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


def test_create_default_registry_registers_the_same_tools_when_wired() -> None:
    """Supplying settings and a driver changes the tool's answers, not the registered surface."""
    wired = create_default_registry(settings=_real_cfg(), esp32_driver=AsyncMock())
    assert _described(wired) == _described(create_default_registry())
    assert len(wired) == 11


def test_names_property(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(name="a", description="", handler=_dummy_handler))
    registry.register(ToolSpec(name="b", description="", handler=_dummy_handler))
    assert set(registry.names) == {"a", "b"}


def test_get_returns_none_for_missing(registry: ToolRegistry) -> None:
    assert registry.get("missing") is None


def test_register_duplicate_warns(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="dup", description="", handler=_dummy_handler)
    registry.register(spec)
    registry.register(spec)  # Should warn but not raise
    assert len(registry) == 1


# -- Built-in tool handler dispatch tests --


@pytest.mark.asyncio
async def test_dispatch_health_check() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("health_check")
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_dispatch_calibrate_ultrasonic() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("calibrate_ultrasonic")
    assert result["status"] == "calibration_complete"


@pytest.mark.asyncio
async def test_dispatch_esp32_diagnostics() -> None:
    """REVERSAL (F-059 Task 4.3): this pin used to assert ``{"status": "esp32_ok"}``.

    The stub answered ``esp32_ok`` from every registry, whether or not a board was
    wired, so a mocked or dead ESP32 passed for healthy. With no ``Settings`` there is
    nothing to ask, and the tool now says exactly that instead of reporting health.
    """
    reg = create_default_registry()
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {"status": "unconfigured"}
    assert result["status"] != "esp32_ok"


@pytest.mark.asyncio
async def test_esp32_diagnostics_without_settings_never_touches_a_supplied_driver() -> None:
    """No ``Settings`` means mock-versus-real is unknowable, so the driver is left alone."""
    driver = _live_driver(EncoderReading(imu_valid=True))
    reg = create_default_registry(esp32_driver=driver)
    assert await reg.dispatch("esp32_diagnostics") == {"status": "unconfigured"}
    driver.read_encoders.assert_not_awaited()
    driver.get_battery_voltage.assert_not_awaited()


@pytest.mark.asyncio
async def test_esp32_diagnostics_reports_mock_hardware_instead_of_a_silent_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mock-backed config reads as a dead board if interrogated: the mock never sets imu_valid.

    The default command set is ``legacy``, so this also pins that ``mock`` is decided before
    the command-set refusal does.
    """
    spy = _ProbeSpy(Esp32ProbeResult(True, 1001, 11.9, 0.1, "unused"))
    monkeypatch.setattr(_PROBE, spy)
    driver = _live_driver(EncoderReading())
    cfg = Settings.model_validate({"mock_hardware": True})
    reg = create_default_registry(settings=cfg, esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {"status": "mock", "enabled": True, "mock_hardware": True}
    driver.read_encoders.assert_not_awaited()
    driver.get_battery_voltage.assert_not_awaited()
    assert spy.calls == []


@pytest.mark.asyncio
async def test_esp32_diagnostics_reports_mock_when_the_board_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``esp32.enabled=false`` (the shipped production overlay) makes the factory build a mock."""
    spy = _ProbeSpy(Esp32ProbeResult(True, 1001, 11.9, 0.1, "unused"))
    monkeypatch.setattr(_PROBE, spy)
    reg = create_default_registry(settings=_real_cfg(enabled=False))
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {"status": "mock", "enabled": False, "mock_hardware": False}
    assert spy.calls == []


@pytest.mark.asyncio
async def test_esp32_diagnostics_reads_the_live_driver_and_never_actuates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator holds the exclusive port: read through its driver, never reopen it."""
    spy = _ProbeSpy(Esp32ProbeResult(True, 1001, 11.9, 0.1, "unused"))
    monkeypatch.setattr(_PROBE, spy)
    driver = _live_driver(EncoderReading(imu_valid=True), voltage=11.9)
    reg = create_default_registry(settings=_real_cfg(), esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {
        "status": "ok",
        "source": "live_driver",
        "imu_valid": True,
        "battery_v": 11.9,
    }
    driver.read_encoders.assert_awaited_once_with()
    driver.get_battery_voltage.assert_awaited_once_with()
    driver.connect.assert_not_awaited()
    driver.disconnect.assert_not_awaited()
    driver.send_velocity.assert_not_awaited()
    driver.emergency_stop.assert_not_awaited()
    assert spy.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("voltage", "battery_v"), [(11.9, 11.9), (0.0, None)])
async def test_esp32_diagnostics_live_driver_reports_the_driver_voltage_sentinel_as_none(
    voltage: float, battery_v: float | None
) -> None:
    """The driver's 0.0 V "no reading" sentinel is ``None`` in the result, not a flat pack."""
    driver = _live_driver(EncoderReading(imu_valid=True), voltage=voltage)
    reg = create_default_registry(settings=_real_cfg(), esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result["status"] == "ok"
    assert result["battery_v"] == battery_v


def _frame_on_poll(poll: int) -> list[EncoderReading]:
    """Driver reads that carry a ``FEEDBACK_BASE_INFO`` frame only on the ``poll``-th read."""
    return [EncoderReading()] * (poll - 1) + [EncoderReading(imu_valid=True)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame_on_poll", "status", "reads"),
    [(1, "ok", 1), (3, "ok", 3), (4, "no_frame", 3)],
)
async def test_esp32_diagnostics_live_driver_polls_exactly_within_the_budget(
    frame_on_poll: int, status: str, reads: int
) -> None:
    """Stock shares one stream across frame types: a wrong-typed line is not silence.

    With a budget of 3, a frame on poll 3 is a response and a frame on poll 4 is not, so
    ``no_frame`` means the whole ``esp32.max_consecutive_timeouts`` budget went by silent.
    """
    driver = _live_driver(EncoderReading())
    driver.read_encoders.side_effect = _frame_on_poll(frame_on_poll)
    cfg = _real_cfg(max_consecutive_timeouts=3)
    reg = create_default_registry(settings=cfg, esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result["status"] == status
    assert result["source"] == "live_driver"
    assert result["imu_valid"] is (status == "ok")
    assert driver.read_encoders.await_count == reads
    driver.send_velocity.assert_not_awaited()
    driver.emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_esp32_diagnostics_live_driver_silence_reads_no_battery() -> None:
    """With no frame there is nothing to read the bus voltage from, and no value is invented."""
    driver = _live_driver(EncoderReading(), voltage=12.3)
    reg = create_default_registry(
        settings=_real_cfg(max_consecutive_timeouts=2), esp32_driver=driver
    )
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {
        "status": "no_frame",
        "source": "live_driver",
        "imu_valid": False,
        "battery_v": None,
    }
    driver.get_battery_voltage.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_read", ["read_encoders", "get_battery_voltage"])
async def test_esp32_diagnostics_live_driver_failure_is_an_error_result(
    failing_read: str,
) -> None:
    """A diagnostics tool reports a driver failure; it never raises it at the MCP client."""
    driver = _live_driver(EncoderReading(imu_valid=True))
    getattr(driver, failing_read).side_effect = OSError("port vanished")
    reg = create_default_registry(settings=_real_cfg(), esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {"status": "error", "detail": "OSError: port vanished"}
    driver.send_velocity.assert_not_awaited()
    driver.emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_esp32_diagnostics_live_driver_cancellation_propagates() -> None:
    """Cooperative cancellation (the MCP request timeout) must not be swallowed as an error."""
    driver = _live_driver(EncoderReading(imu_valid=True))
    driver.read_encoders.side_effect = asyncio.CancelledError
    reg = create_default_registry(settings=_real_cfg(), esp32_driver=driver)
    with pytest.raises(asyncio.CancelledError):
        await reg.dispatch("esp32_diagnostics")


@pytest.mark.asyncio
async def test_esp32_diagnostics_live_driver_refuses_the_legacy_command_set() -> None:
    """Legacy never sets ``imu_valid``, so ``no_frame`` would call a living board dead.

    That is the F-008 misdiagnosis in miniature (a stock board probed in the wrong dialect
    looks silent), so the tool refuses with the remedy and leaves the driver alone.
    """
    driver = _live_driver(EncoderReading())
    reg = create_default_registry(settings=_real_cfg(command_set="legacy"), esp32_driver=driver)
    result = await reg.dispatch("esp32_diagnostics")
    assert result["status"] == "error"
    detail = str(result["detail"])
    assert detail.startswith("ValueError: ")
    assert "legacy" in detail
    assert "waveshare_stock" in detail
    driver.read_encoders.assert_not_awaited()
    driver.get_battery_voltage.assert_not_awaited()


@pytest.mark.asyncio
async def test_esp32_diagnostics_probes_the_board_when_no_live_driver_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = "FEEDBACK_BASE_INFO after 1 poll(s) on /dev/ttyUSB0; battery_v=11.90"
    spy = _ProbeSpy(Esp32ProbeResult(True, 1001, 11.9, 0.25, detail))
    monkeypatch.setattr(_PROBE, spy)
    cfg = _real_cfg()
    reg = create_default_registry(settings=cfg)
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {
        "status": "ok",
        "source": "probe",
        "frame_type": 1001,
        "battery_v": 11.9,
        "elapsed_s": 0.25,
        "detail": detail,
    }
    assert spy.calls == [cfg]


@pytest.mark.asyncio
async def test_esp32_diagnostics_reports_a_silent_board_as_no_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = "no FEEDBACK_BASE_INFO frame in 5 polls on /dev/ttyUSB0 at 115200 baud"
    monkeypatch.setattr(_PROBE, _ProbeSpy(Esp32ProbeResult(False, None, None, 2.5, detail)))
    reg = create_default_registry(settings=_real_cfg())
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {
        "status": "no_response",
        "source": "probe",
        "frame_type": None,
        "battery_v": None,
        "elapsed_s": 2.5,
        "detail": detail,
    }


@pytest.mark.asyncio
async def test_esp32_diagnostics_probe_failure_is_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect error from the probe becomes ``error`` with ``<Type>: <message>``."""
    monkeypatch.setattr(_PROBE, _ProbeSpy(OSError("no such port")))
    reg = create_default_registry(settings=_real_cfg())
    result = await reg.dispatch("esp32_diagnostics")
    assert result == {"status": "error", "detail": "OSError: no such port"}


@pytest.mark.asyncio
async def test_esp32_diagnostics_probe_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_PROBE, _ProbeSpy(asyncio.CancelledError()))
    reg = create_default_registry(settings=_real_cfg())
    with pytest.raises(asyncio.CancelledError):
        await reg.dispatch("esp32_diagnostics")


@pytest.mark.asyncio
async def test_esp32_diagnostics_refuses_the_legacy_command_set_without_opening_a_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real probe raises ``ValueError`` before it reaches the factory; the tool reports it."""
    built: list[Settings] = []
    monkeypatch.setattr("mousedroid.factory.build_esp32_driver", built.append)
    reg = create_default_registry(settings=_real_cfg(command_set="legacy"))
    result = await reg.dispatch("esp32_diagnostics")
    assert result["status"] == "error"
    assert str(result["detail"]).startswith("ValueError: ")
    assert "waveshare_stock" in str(result["detail"])
    assert built == []


@pytest.mark.asyncio
async def test_dispatch_tensorrt_compile() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("tensorrt_compile")
    assert result["status"] == "compilation_pending"


@pytest.mark.asyncio
async def test_dispatch_benchmark_latency() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("benchmark_latency")
    assert result["status"] == "benchmark_pending"


@pytest.mark.asyncio
async def test_dispatch_export_experience() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("export_experience", path="/tmp/test_export")
    assert result["status"] == "exported"
    assert result["path"] == "/tmp/test_export"


@pytest.mark.asyncio
async def test_dispatch_diagnose_cloud_disabled_without_config() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("diagnose_cloud")
    assert result["status"] == "cloud_disabled"


@pytest.mark.asyncio
async def test_dispatch_diagnose_cloud_missing_dependency() -> None:
    from unittest.mock import patch

    reg = create_default_registry(
        gcp_cfg=GCPConfig(project_id="proj", robot_id="bot"),
    )
    with patch(
        "mousedroid.cloud._auth.resolve_credentials",
        side_effect=RuntimeError("no ADC"),
    ):
        result = await reg.dispatch("diagnose_cloud")
    assert result["status"] in {"missing_dependency", "credentials_error"}


@pytest.mark.asyncio
async def test_dispatch_diagnose_cloud_success() -> None:
    import sys
    from unittest.mock import patch

    cfg = GCPConfig(project_id="proj", robot_id="bot")
    reg = create_default_registry(gcp_cfg=cfg)

    future = MagicMock()
    future.result.return_value = "message-id"

    publisher = MagicMock()
    publisher.topic_path.return_value = "projects/proj/topics/telemetry"
    publisher.publish.return_value = future

    publisher_cls = MagicMock(return_value=publisher)
    pubsub_module = MagicMock(PublisherClient=publisher_cls)
    storage_module = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.pubsub_v1": pubsub_module,
                "google.cloud.storage": storage_module,
            },
        ),
        patch("mousedroid.cloud._auth.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "proj")
        result = await reg.dispatch("diagnose_cloud")

    assert result["status"] == "ok"
    assert result["project_id"] == "proj"
    assert result["topic"] == "projects/proj/topics/telemetry"
    publisher.publish.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_translate_nl_mission() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("translate_nl_mission", mission="go left")
    assert result["status"] == "llm_unavailable"
    assert result["mission"] == "go left"


@pytest.mark.asyncio
async def test_dispatch_translate_nl_mission_with_gateway() -> None:
    gateway = AsyncMock()
    gateway.is_ready = True
    gateway.translate_mission.return_value = GoalVector(0.4, -0.2, 0.1)
    metrics = MagicMock()

    reg = create_default_registry(llm_gateway=gateway, metrics_registry=metrics)
    result = await reg.dispatch("translate_nl_mission", mission="go left")

    assert result["status"] == "translated"
    assert result["goal_vector"] == {
        "vx_target": 0.4,
        "vy_target": -0.2,
        "omega_target": 0.1,
    }
    metrics.inc_llm_translation.assert_called_once_with("translated")
    metrics.observe_llm_translation_latency_ms.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_translate_nl_mission_empty_request() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("translate_nl_mission", mission="   ")
    assert result["status"] == "invalid_request"


@pytest.mark.asyncio
async def test_dispatch_system_info() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("system_info")
    assert "platform" in result
    assert "python" in result
    assert "machine" in result


@pytest.mark.asyncio
async def test_dispatch_mic_diagnostics() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("mic_diagnostics")
    assert "status" in result


# -- Deprecated shim tests --


def test_deprecated_tools_init_imports() -> None:
    """Test that mousedroid.tools re-exports canonical types."""
    import mousedroid.tools as tools_mod

    assert tools_mod.ToolRegistry is ToolRegistry
    assert tools_mod.ToolSpec is ToolSpec


def test_canonical_tools_registry_importable() -> None:
    """Test that canonical mousedroid.common.tools.registry is importable."""
    from mousedroid.common.tools.registry import ToolRegistry as CanonicalRegistry

    assert CanonicalRegistry is ToolRegistry


def test_default_registry_has_lidar_diagnostics() -> None:
    """Verify 'lidar_diagnostics' is registered in the default registry."""
    reg = create_default_registry()
    assert "lidar_diagnostics" in reg.names


@pytest.mark.asyncio
async def test_lidar_diagnostics_reports_ok_when_pyserial_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_lidar_diagnostics`` returns the driver status when pyserial imports."""
    import sys
    import types

    from mousedroid.common.tools import registry as registry_mod

    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    result = await registry_mod._lidar_diagnostics()
    assert result == {"status": "ok", "driver": "FHL-LD19"}


@pytest.mark.asyncio
async def test_lidar_diagnostics_reports_missing_when_pyserial_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_lidar_diagnostics`` reports a clean status when pyserial is unavailable."""
    from mousedroid.common.tools import registry as registry_mod

    monkeypatch.setattr(registry_mod, "module_available", lambda name: False)
    result = await registry_mod._lidar_diagnostics()
    assert result == {"status": "pyserial_not_installed"}
