"""Tests for telemetry factory functions."""

from __future__ import annotations

import os

from mousedroid.config.schema import Settings
from mousedroid.factory import build_telemetry_publisher, build_telemetry_server
from mousedroid.health.monitor import HealthMonitor
from mousedroid.telemetry.mock_server import MockTelemetryServer
from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
from mousedroid.telemetry.server import TelemetryServer


def _make_settings(**overrides) -> Settings:
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    defaults = {"mock_hardware": True}
    defaults.update(overrides)
    return Settings(**defaults)


def test_build_publisher_disabled():
    cfg = _make_settings()
    assert cfg.telemetry.enabled is False
    pub = build_telemetry_publisher(cfg)
    assert pub is None


def test_build_publisher_enabled():
    cfg = _make_settings()
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    assert isinstance(pub, TelemetryPublisherProtocol)


def test_build_server_disabled():
    cfg = _make_settings()
    health = HealthMonitor(cfg.health, cfg.jetson)
    server = build_telemetry_server(cfg, None, health)
    assert server is None


def test_build_server_enabled_mock_hardware():
    cfg = _make_settings()
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    pub = build_telemetry_publisher(cfg)
    health = HealthMonitor(cfg.health, cfg.jetson)
    server = build_telemetry_server(cfg, pub, health)
    assert server is not None
    assert isinstance(server, MockTelemetryServer)
    assert isinstance(server, TelemetryServerProtocol)


def test_build_server_returns_none_without_publisher():
    cfg = _make_settings()
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    health = HealthMonitor(cfg.health, cfg.jetson)
    server = build_telemetry_server(cfg, None, health)
    assert server is None


def test_build_publisher_stats():
    cfg = _make_settings()
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    stats = pub.stats
    assert stats["frames_published"] == 0
    assert stats["frames_dropped"] == 0


def test_build_publisher_queue():
    cfg = _make_settings()
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    q = pub.get_queue()
    assert q.maxsize == cfg.telemetry.queue_size


def test_build_server_enabled_real_server_uses_metrics_config_path():
    cfg = _make_settings()
    cfg = cfg.model_copy(
        update={
            "mock_hardware": False,
            "telemetry": cfg.telemetry.model_copy(update={"enabled": True}),
            "metrics": cfg.metrics.model_copy(update={"enabled": True, "path": "/prom-metrics"}),
        }
    )
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    health = HealthMonitor(cfg.health, cfg.jetson)

    server = build_telemetry_server(cfg, pub, health)

    assert isinstance(server, TelemetryServer)
    assert server._metrics is not None
    assert server._metrics_path == "/prom-metrics"
    assert server._publisher is pub


def test_build_server_uses_legacy_telemetry_metrics_path_when_metrics_path_default():
    cfg = _make_settings()
    cfg = cfg.model_copy(
        update={
            "mock_hardware": False,
            "telemetry": cfg.telemetry.model_copy(
                update={"enabled": True, "metrics_path": "/legacy-metrics"}
            ),
            "metrics": cfg.metrics.model_copy(update={"enabled": True, "path": "/metrics"}),
        }
    )
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    health = HealthMonitor(cfg.health, cfg.jetson)

    server = build_telemetry_server(cfg, pub, health)

    assert isinstance(server, TelemetryServer)
    assert server._metrics_path == "/legacy-metrics"


def test_build_server_disables_metrics_registry_when_metrics_disabled():
    cfg = _make_settings()
    cfg = cfg.model_copy(
        update={
            "mock_hardware": False,
            "telemetry": cfg.telemetry.model_copy(update={"enabled": True}),
            "metrics": cfg.metrics.model_copy(update={"enabled": False}),
        }
    )
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    health = HealthMonitor(cfg.health, cfg.jetson)

    server = build_telemetry_server(cfg, pub, health)

    assert isinstance(server, TelemetryServer)
    assert server._metrics is None


def test_build_server_reuses_provided_metrics_registry():
    from mousedroid.telemetry.metrics import MetricsRegistry

    cfg = _make_settings()
    cfg = cfg.model_copy(
        update={
            "mock_hardware": False,
            "telemetry": cfg.telemetry.model_copy(update={"enabled": True}),
            "metrics": cfg.metrics.model_copy(update={"enabled": True}),
        }
    )
    pub = build_telemetry_publisher(cfg)
    assert pub is not None
    health = HealthMonitor(cfg.health, cfg.jetson)
    metrics_registry = MetricsRegistry(cfg.metrics)

    server = build_telemetry_server(cfg, pub, health, metrics_registry=metrics_registry)

    assert isinstance(server, TelemetryServer)
    assert server._metrics is metrics_registry
