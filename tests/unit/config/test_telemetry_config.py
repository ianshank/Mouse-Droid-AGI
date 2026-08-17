"""Tests for TelemetryConfig — defaults, validation, YAML loading."""

from __future__ import annotations

import os

from mousedroid.config.schema import Settings, TelemetryConfig


def test_telemetry_config_defaults():
    cfg = TelemetryConfig()
    assert cfg.enabled is False
    assert cfg.host == "0.0.0.0"  # noqa: S104
    assert cfg.port == 8080
    assert cfg.ws_path == "/ws"
    assert cfg.api_prefix == "/api/v1"
    assert cfg.publish_hz == 10.0
    assert cfg.max_clients == 10
    assert cfg.queue_size == 64
    assert cfg.serialization == "json"
    assert cfg.api_key is None
    assert cfg.mdns_enabled is True
    assert cfg.mdns_service_name == "MouseDroid Telemetry"
    assert cfg.cors_origins == ["*"]
    assert cfg.log_stream_buffer == 200
    assert cfg.preferred_interface is None


def test_telemetry_config_custom_values():
    cfg = TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=9090,
        publish_hz=5.0,
        max_clients=20,
        serialization="msgpack",
        api_key="secret123",
        mdns_enabled=False,
        preferred_interface="eth0",
    )
    assert cfg.enabled is True
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9090
    assert cfg.publish_hz == 5.0
    assert cfg.serialization == "msgpack"
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "secret123"
    assert cfg.preferred_interface == "eth0"


def test_telemetry_config_api_key_masked_in_repr():
    """SecretStr must mask the key wherever the config is logged/dumped.

    This is the entire reason api_key is SecretStr rather than str — an
    unredacted repr() would defeat it (e.g. via the settings-dump MLflow
    artifact in training/pipeline_orchestrator.py). Mirrors the equivalent
    LLMConfig.api_key masking test.
    """
    cfg = TelemetryConfig(api_key="secret123")
    assert "secret123" not in repr(cfg)


def test_telemetry_config_port_validation():
    import pytest

    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(port=0)
    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(port=70000)


def test_telemetry_config_publish_hz_validation():
    import pytest

    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(publish_hz=0)
    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(publish_hz=100)


def test_settings_includes_telemetry():
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    settings = Settings(mock_hardware=True)
    assert hasattr(settings, "telemetry")
    assert isinstance(settings.telemetry, TelemetryConfig)
    assert settings.telemetry.enabled is False


def test_settings_backwards_compat_without_telemetry():
    """Settings load fine without telemetry key — defaults kick in."""
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    settings = Settings(mock_hardware=True)
    assert settings.telemetry.port == 8080


def test_telemetry_env_var_override(monkeypatch):
    monkeypatch.setenv("MOUSEDROID_TELEMETRY__ENABLED", "true")
    monkeypatch.setenv("MOUSEDROID_TELEMETRY__PORT", "9999")
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    settings = Settings()
    assert settings.telemetry.enabled is True
    assert settings.telemetry.port == 9999


def test_telemetry_config_cors_origins():
    cfg = TelemetryConfig(cors_origins=["http://localhost:3000", "http://192.168.1.1"])
    assert len(cfg.cors_origins) == 2


def test_telemetry_config_queue_size_validation():
    import pytest

    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(queue_size=0)


def test_telemetry_config_log_stream_buffer_validation():
    import pytest

    with pytest.raises(ValueError, match=r"validation error"):
        TelemetryConfig(log_stream_buffer=0)
