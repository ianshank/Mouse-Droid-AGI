"""Unit tests for the rover motor MCP tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mousedroid.common.tools.motor_tools import MotorToolDeps, register_motor_tools
from mousedroid.common.tools.registry import ToolRegistry
from mousedroid.config.schema import Settings


def _encoder_reading(**overrides: float) -> Any:
    """Build a stub EncoderReading-shaped object."""
    defaults = {
        "left_velocity_mps": 0.1,
        "right_velocity_mps": 0.1,
        "odometry_x_m": 0.0,
        "odometry_y_m": 0.0,
        "heading_rad": 0.0,
        "timestamp": 1.0,
    }
    defaults.update(overrides)
    return type("Enc", (), defaults)()


@pytest.fixture
def deps() -> MotorToolDeps:
    cfg = Settings()
    esp32 = AsyncMock()
    esp32.read_encoders.return_value = _encoder_reading()
    return MotorToolDeps(esp32=esp32, cfg=cfg)


class TestSetVelocity:
    """``set_velocity`` clamps and dispatches via the ESP32 driver."""

    async def test_clamps_to_max_velocity(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("set_velocity")
        assert spec is not None
        max_v = deps.cfg.esp32.max_velocity_mps
        result = await spec.handler(vx=max_v * 10, vy=0.0, omega=0.0)
        deps.esp32.send_velocity.assert_awaited_with(max_v, 0.0, 0.0)
        assert result == {
            "status": "ok",
            "vx": max_v,
            "vy": 0.0,
            "omega": 0.0,
        }

    async def test_clamps_to_negative_max(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("set_velocity")
        assert spec is not None
        max_v = deps.cfg.esp32.max_velocity_mps
        await spec.handler(vx=-max_v * 10, vy=0.0, omega=0.0)
        deps.esp32.send_velocity.assert_awaited_with(-max_v, 0.0, 0.0)

    async def test_clamps_omega_to_config_bound(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("set_velocity")
        assert spec is not None
        max_omega = deps.cfg.esp32.max_omega_rads
        result = await spec.handler(vx=0.0, vy=0.0, omega=max_omega * 5)
        deps.esp32.send_velocity.assert_awaited_with(0.0, 0.0, max_omega)
        assert result["omega"] == max_omega

    async def test_in_range_values_pass_through(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("set_velocity")
        assert spec is not None
        max_v = deps.cfg.esp32.max_velocity_mps
        target = max_v * 0.25
        await spec.handler(vx=target, vy=0.0, omega=0.0)
        deps.esp32.send_velocity.assert_awaited_with(target, 0.0, 0.0)


class TestEmergencyStop:
    """``emergency_stop`` always issues an e-stop."""

    async def test_dispatches_emergency_stop(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("emergency_stop")
        assert spec is not None
        result = await spec.handler()
        assert result == {"status": "ok"}
        deps.esp32.emergency_stop.assert_awaited_once()


class TestReadEncoders:
    """``read_encoders`` returns a JSON-serialisable dict."""

    async def test_returns_serialisable_payload(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        spec = registry.get("read_encoders")
        assert spec is not None
        result = await spec.handler()
        assert set(result.keys()) >= {
            "left_velocity_mps",
            "right_velocity_mps",
            "odometry_x_m",
            "odometry_y_m",
            "heading_rad",
            "timestamp",
        }
        assert all(isinstance(v, float) for v in result.values())


class TestRegistration:
    """``register_motor_tools`` registers exactly the three expected tools."""

    def test_registers_three_tools(self, deps: MotorToolDeps) -> None:
        registry = ToolRegistry()
        register_motor_tools(registry, deps)
        assert {"set_velocity", "read_encoders", "emergency_stop"} <= set(registry.names)

    def test_custom_tool_names_are_honoured(self) -> None:
        registry = ToolRegistry()
        cfg = Settings()
        esp32 = AsyncMock()
        esp32.read_encoders.return_value = _encoder_reading()
        custom = MotorToolDeps(
            esp32=esp32,
            cfg=cfg,
            name_set_velocity="rover_set_velocity",
            name_read_encoders="rover_read_encoders",
            name_emergency_stop="rover_estop",
        )
        register_motor_tools(registry, custom)
        assert "rover_set_velocity" in registry.names
        assert "rover_read_encoders" in registry.names
        assert "rover_estop" in registry.names


class TestActuationAllowlistDefault:
    """``MCPConfig.actuation_tools`` default expansion stays
    backwards-compatible with existing YAML overrides."""

    def test_default_includes_set_velocity(self) -> None:
        from mousedroid.config.schema import MCPConfig

        cfg = MCPConfig()
        assert "set_velocity" in cfg.actuation_tools

    def test_default_excludes_emergency_stop(self) -> None:
        """`emergency_stop` MUST stay callable during emergencies."""
        from mousedroid.config.schema import MCPConfig

        cfg = MCPConfig()
        assert "emergency_stop" not in cfg.actuation_tools

    def test_default_excludes_read_encoders(self) -> None:
        """`read_encoders` is read-only — never an actuation."""
        from mousedroid.config.schema import MCPConfig

        cfg = MCPConfig()
        assert "read_encoders" not in cfg.actuation_tools

    def test_yaml_override_wins_over_default_expansion(self) -> None:
        """YAMLs that explicitly set actuation_tools keep their values."""
        from mousedroid.config.schema import MCPConfig

        cfg = MCPConfig.model_validate({"actuation_tools": ["only_this_one"]})
        assert cfg.actuation_tools == ["only_this_one"]
        assert "set_velocity" not in cfg.actuation_tools


class TestFactoryWiring:
    """``build_orchestrator`` registers motor tools only for the rover platform."""

    def test_motor_tools_registered_for_mouse_droid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
        from mousedroid.config.schema import PlatformType, Settings
        from mousedroid.factory import build_orchestrator

        cfg = Settings(platform=PlatformType.MOUSE_DROID, mock_hardware=True)
        orch = build_orchestrator(cfg)
        registry_names = set(orch._tool_registry.names)  # type: ignore[attr-defined]
        assert {"set_velocity", "read_encoders", "emergency_stop"} <= registry_names

    def test_motor_tools_skipped_for_robot_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
        from mousedroid.config.schema import PlatformType, Settings
        from mousedroid.factory import build_orchestrator

        cfg = Settings(platform=PlatformType.ROBOT_ARM, mock_hardware=True)
        orch = build_orchestrator(cfg)
        registry_names = set(orch._tool_registry.names)  # type: ignore[attr-defined]
        assert "set_velocity" not in registry_names
        assert "emergency_stop" not in registry_names
