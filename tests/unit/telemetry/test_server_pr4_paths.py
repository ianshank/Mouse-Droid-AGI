"""Tests for PR #4 server paths: raw lidar broadcast, mDNS readiness, factory mock source."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from mousedroid.config.schema import (
    HealthConfig,
    JetsonConfig,
    MetricsConfig,
    Settings,
    TelemetryConfig,
)
from mousedroid.factory import build_mock_telemetry_source, build_telemetry_server
from mousedroid.health.monitor import HealthMonitor
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import LidarRawScan
from mousedroid.telemetry.publisher import TelemetryPublisher
from mousedroid.telemetry.server import TelemetryServer


def _cfg() -> TelemetryConfig:
    return TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=1,
        port_discovery_strategy="kernel_assigned",
        mdns_enabled=False,
        publish_hz=30.0,
        lidar_raw_publish_hz=30.0,
    )


def _make_server(*, with_metrics: bool = True) -> tuple[TelemetryServer, TelemetryPublisher]:
    cfg = _cfg()
    publisher = TelemetryPublisher(cfg)
    metrics: MetricsRegistry | None = MetricsRegistry(MetricsConfig()) if with_metrics else None
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        metrics_registry=metrics,
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )
    return server, publisher


@pytest.mark.asyncio
async def test_raw_broadcast_loop_drains_queue_and_increments_metric() -> None:
    """The broadcast loop pulls scans and increments the publish counter."""
    server, publisher = _make_server(with_metrics=True)
    server._running = True

    publisher._lidar_raw_last_publish = 0.0  # type: ignore[attr-defined]
    await publisher.publish_lidar_raw(
        LidarRawScan(
            timestamp=0.0,
            angles_rad=[0.0, 1.0],
            distances_m=[1.0, 2.0],
            n_points=2,
            scan_duration_s=0.1,
        )
    )

    task = asyncio.create_task(server._lidar_raw_broadcast_loop())
    # Drain the queue then stop the loop.
    for _ in range(20):
        if publisher.get_lidar_raw_queue().empty():
            break
        await asyncio.sleep(0.02)
    server._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert server._latest_lidar_raw is not None
    snapshot = server._metrics._lidar_raw_published.value  # type: ignore[attr-defined]
    assert snapshot >= 1


@pytest.mark.asyncio
async def test_raw_broadcast_loop_returns_when_queue_missing() -> None:
    """Without a wired queue the loop is a no-op."""
    cfg = _cfg()
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        publisher=publisher,
        lidar_raw_queue=None,
    )
    # ``return`` immediately when no queue is wired.
    await server._lidar_raw_broadcast_loop()


@pytest.mark.asyncio
async def test_mdns_register_sets_event_on_failure() -> None:
    """``_register_mdns`` always resolves the readiness event."""
    cfg = _cfg().model_copy(update={"mdns_enabled": True, "mdns_register_timeout_s": 0.2})
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )
    # No zeroconf install required: the function must still complete
    # cleanly and set the readiness event regardless of which branch
    # (ImportError or generic Exception) fires.
    await server._register_mdns()
    assert server._mdns_registered_event.is_set()


@pytest.mark.asyncio
async def test_mdns_register_records_failure_on_runtime_error() -> None:
    """Generic exceptions route through FailureRecorder + Prometheus gauge."""
    try:
        import zeroconf  # noqa: F401
    except ImportError:
        pytest.skip("zeroconf not installed; covered by ImportError branch elsewhere")

    cfg = _cfg().model_copy(update={"mdns_enabled": True, "mdns_register_timeout_s": 0.2})
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        metrics_registry=MetricsRegistry(MetricsConfig()),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )

    import mousedroid.telemetry.server as server_mod

    original = server_mod.get_default_ip

    def _raise() -> str:
        raise RuntimeError("no ip")

    server_mod.get_default_ip = _raise  # type: ignore[assignment]

    calls: list[tuple[str, str]] = []

    def _spy(subsystem: str, reason: str, **_kw: Any) -> None:
        calls.append((subsystem, reason))

    server._failure_recorder.record = _spy  # type: ignore[method-assign]

    try:
        await server._register_mdns()
    finally:
        server_mod.get_default_ip = original  # type: ignore[assignment]

    assert server._mdns_registered_event.is_set()
    assert server._mdns_ok is False
    assert calls
    assert calls[0] == ("telemetry", "mdns_register_failed")


@pytest.mark.asyncio
async def test_negotiation_after_close_handled_gracefully() -> None:
    """If the WS closes during ack send, no exception escapes."""
    from aiohttp import WSMsgType

    server, _ = _make_server(with_metrics=False)

    class _ClosingWs:
        def __init__(self) -> None:
            self.received = False
            self.closed = False

        async def receive(self) -> Any:
            return type("Msg", (), {"type": WSMsgType.TEXT, "data": '{"hello":{}}'})()

        async def send_json(self, _: Any) -> None:
            raise ConnectionResetError("client gone")

        async def close(self, *, code: int = 1000, message: bytes = b"") -> None:
            # ``_negotiate_ws`` calls close() on hard failures; the
            # ConnectionResetError surrogate above means the ack send
            # failed, but the negotiation result is still ok=False
            # (invalid_hello) so close() is still invoked.
            self.closed = True

    chosen = await server._negotiate_ws(_ClosingWs())  # type: ignore[arg-type]
    assert chosen == "json"  # server default after invalid_hello rejection


def test_build_mock_telemetry_source_disabled_when_telemetry_off() -> None:
    """``build_mock_telemetry_source`` returns None when telemetry is disabled."""
    cfg = Settings(mock_hardware=True)
    # telemetry.enabled defaults to False — confirm short-circuit.
    publisher = TelemetryPublisher(cfg.telemetry)
    assert build_mock_telemetry_source(cfg, publisher) is None


def test_build_mock_telemetry_source_disabled_without_publisher() -> None:
    """The factory short-circuits when no publisher is wired."""
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    assert build_mock_telemetry_source(cfg, None) is None


def test_build_mock_telemetry_source_disabled_via_flag() -> None:
    """``mock_telemetry_source_enabled=False`` suppresses the source."""
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(
        update={
            "telemetry": cfg.telemetry.model_copy(
                update={"enabled": True, "mock_telemetry_source_enabled": False}
            )
        }
    )
    publisher = TelemetryPublisher(cfg.telemetry)
    assert build_mock_telemetry_source(cfg, publisher) is None


def test_build_mock_telemetry_source_returns_instance() -> None:
    """The factory wires the source when both gates pass."""
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    publisher = TelemetryPublisher(cfg.telemetry)
    src = build_mock_telemetry_source(cfg, publisher)
    assert src is not None
    assert hasattr(src, "start")
    assert hasattr(src, "stop")


@pytest.mark.asyncio
async def test_broadcast_loop_propagates_sensor_liveness_to_metrics() -> None:
    """A frame with sensor_liveness sets the gauge via set_sensor_liveness."""
    from mousedroid.telemetry.protocol import TelemetryFrame

    server, publisher = _make_server(with_metrics=True)
    server._running = True

    frame = TelemetryFrame(
        timestamp=0.0,
        sensor_liveness={
            "lidar": {"state": "live", "age_s": 0.1},
            "vision": {"state": "stale", "age_s": 5.0},
        },
    )
    publisher.get_queue().put_nowait(frame)

    task = asyncio.create_task(server._broadcast_loop())
    for _ in range(20):
        if publisher.get_queue().empty() and server._latest_frame is not None:
            break
        await asyncio.sleep(0.02)
    server._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    snap = server._metrics._sensor_liveness.snapshot()  # type: ignore[attr-defined]
    assert snap[("lidar", "live")] == 1.0
    assert snap[("vision", "stale")] == 1.0


@pytest.mark.asyncio
async def test_start_with_mdns_enabled_completes() -> None:
    """``start()`` with mdns_enabled=True awaits the readiness event."""
    cfg = _cfg().model_copy(update={"mdns_enabled": True, "mdns_register_timeout_s": 0.5})
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        metrics_registry=MetricsRegistry(MetricsConfig()),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )
    try:
        await server.start()
        # Either succeeded or timed out; either way start() returned.
        assert server.is_running
        assert server._bound_port > 0
    finally:
        await server.stop()


def test_build_telemetry_server_wires_lidar_raw_queue() -> None:
    """The factory passes the publisher's raw queue into the server."""
    cfg = Settings(mock_hardware=True)
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"enabled": True})})
    publisher = TelemetryPublisher(cfg.telemetry)
    health = HealthMonitor(cfg.health, cfg.jetson)
    server = build_telemetry_server(cfg, publisher, health)
    assert isinstance(server, TelemetryServer)
    assert server._lidar_raw_queue is publisher.get_lidar_raw_queue()
