from __future__ import annotations

import yaml

from mousedroid.config.schema import Settings


def test_settings_no_args_works() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.platform.value == "mouse_droid"


def test_settings_defaults_populated() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.loop is not None
    assert s.model is not None
    assert s.mcts is not None
    assert s.safety is not None
    assert s.esp32 is not None
    assert s.camera is not None


def test_old_style_minimal_yaml() -> None:
    minimal_yaml = """
    mock_hardware: true
    platform: mouse_droid
    """
    data = yaml.safe_load(minimal_yaml)
    s = Settings.model_validate(data)
    assert s.mock_hardware is True
    assert s.platform.value == "mouse_droid"


def test_legacy_runtime_validation_fields_get_defaults() -> None:
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    camera:
      backend: auto
    lidar:
      enabled: true
    """
    data = yaml.safe_load(legacy_yaml)
    s = Settings.model_validate(data)
    assert s.camera.device_path == "/dev/video0"
    assert s.lidar is not None
    assert s.lidar.scan_acquisition_timeout_s == 1.0
    assert s.lidar.min_scan_coverage_deg == 270.0
    assert s.lidar.scan_timeout_multiplier == 2.0


def test_new_fields_have_defaults() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.memory is not None
    assert s.learning is not None
    assert s.reward is not None
    assert s.curiosity is not None
    assert s.circuit_breaker is not None
    assert s.metrics is not None
    assert s.health is not None
    assert s.retry is not None


def test_debug_default_false() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.debug is False


def test_legacy_loop_tick_timeout_ms_is_migrated() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "loop": {"tick_timeout_ms": 750},
        }
    )
    assert s.loop.tick_timeout_s == 0.75


def test_legacy_safety_max_loop_time_s_is_migrated() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "safety": {"max_loop_time_s": 0.15},
        }
    )
    assert s.safety.max_loop_time_ms == 150.0


def test_canonical_value_wins_over_legacy_alias() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "loop": {
                "tick_timeout_s": 2.0,
                "tick_timeout_ms": 750,
            },
        }
    )
    assert s.loop.tick_timeout_s == 2.0


def test_legacy_camera_width_height_are_migrated() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "camera": {
                "width": 800,
                "height": 600,
            },
        }
    )
    assert s.camera.resolution_width == 800
    assert s.camera.resolution_height == 600


def test_legacy_telemetry_publish_rate_hz_is_migrated() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "telemetry": {"publish_rate_hz": 15.0},
        }
    )
    assert s.telemetry.publish_hz == 15.0


def test_legacy_telemetry_aliases_are_migrated() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "telemetry": {
                "ws_endpoint": "/legacy/ws",
                "api_base_path": "/legacy/api",
                "telemetry_rate_hz": 12.0,
                "max_ws_clients": 9,
                "websocket_queue_size": 111,
                "bind_host": "127.0.0.1",
                "bind_port": 9010,
            },
        }
    )
    assert s.telemetry.ws_path == "/legacy/ws"
    assert s.telemetry.api_prefix == "/legacy/api"
    assert s.telemetry.publish_hz == 12.0
    assert s.telemetry.max_clients == 9
    assert s.telemetry.queue_size == 111
    assert s.telemetry.host == "127.0.0.1"
    assert s.telemetry.port == 9010


def test_legacy_telemetry_publish_interval_s_is_converted_to_hz() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "telemetry": {
                "publish_interval_s": 0.25,
            },
        }
    )
    assert s.telemetry.publish_hz == 4.0


def test_canonical_telemetry_values_win_over_legacy_aliases() -> None:
    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "telemetry": {
                "ws_path": "/canonical/ws",
                "ws_endpoint": "/legacy/ws",
                "publish_hz": 9.0,
                "publish_interval_s": 0.25,
            },
        }
    )
    assert s.telemetry.ws_path == "/canonical/ws"
    assert s.telemetry.publish_hz == 9.0


def test_settings_without_mcp_block_remains_valid() -> None:
    """Existing YAML without an `mcp:` block must continue to load."""
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    telemetry:
      enabled: false
    """
    data = yaml.safe_load(legacy_yaml)
    s = Settings.model_validate(data)
    assert s.mcp is None


def test_mcp_block_loads_when_present() -> None:
    """A complete `mcp:` block must populate MCPConfig with defaults filled."""
    config_yaml = """
    mock_hardware: true
    platform: mouse_droid
    mcp:
      enabled: false
      transport: stdio
      host: 127.0.0.1
      port: 8765
    """
    data = yaml.safe_load(config_yaml)
    s = Settings.model_validate(data)
    assert s.mcp is not None
    assert s.mcp.enabled is False
    assert s.mcp.transport == "stdio"
    assert s.mcp.host == "127.0.0.1"
    assert s.mcp.port == 8765
    # Defaults still populated
    assert s.mcp.resources.recent_frames_max == 64
    assert s.mcp.resources.log_tail_max == 200
