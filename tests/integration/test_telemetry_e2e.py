"""End-to-end tests for the telemetry pipeline.

Tests the full flow: orchestrator tick → publisher → queue → consumer.
Server-level E2E tests are in test_telemetry_server.py (aiohttp TestClient).
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import torch

from mousedroid.config.schema import Settings, TelemetryConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.log_buffer import LogRingBuffer
from mousedroid.telemetry.protocol import TelemetryFrame
from mousedroid.telemetry.publisher import TelemetryPublisher


def _make_settings() -> Settings:
    os.environ["MOUSEDROID_MOCK_HARDWARE"] = "true"
    return Settings(mock_hardware=True)


async def test_orchestrator_to_publisher_pipeline():
    """Full pipeline: orchestrator tick → publisher → queue."""
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = _make_settings()
    telemetry_cfg = cfg.telemetry.model_copy(update={"enabled": True, "publish_hz": 1000.0})
    cfg = cfg.model_copy(update={"telemetry": telemetry_cfg})

    publisher = TelemetryPublisher(cfg.telemetry)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        None,
        None,
    )

    agent = MagicMock()
    agent.name = "test"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety = MagicMock()
    safety_ctx = MagicMock()
    safety_ctx.is_emergency = False
    safety_ctx.law_violations = ()
    safety_ctx.forward_clearance_ok = True
    safety.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = MouseDroidObservationBundle()

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        telemetry_publisher=publisher,
    )

    await orch.tick()

    q = publisher.get_queue()
    assert not q.empty()
    frame = q.get_nowait()
    assert isinstance(frame, TelemetryFrame)
    assert frame.tick_count == 1
    assert isinstance(frame.vision_norm, float)
    assert isinstance(frame.audio_rms, float)


async def test_multiple_ticks_produce_frames():
    cfg = _make_settings()
    telemetry_cfg = cfg.telemetry.model_copy(update={"enabled": True, "publish_hz": 1000.0})
    cfg = cfg.model_copy(update={"telemetry": telemetry_cfg})
    publisher = TelemetryPublisher(cfg.telemetry)

    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        None,
        None,
    )
    agent = MagicMock()
    agent.name = "test"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    safety = MagicMock()
    safety_ctx = MagicMock()
    safety_ctx.is_emergency = False
    safety_ctx.law_violations = ()
    safety_ctx.forward_clearance_ok = True
    safety.evaluate.return_value = safety_ctx
    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = MouseDroidObservationBundle()

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        telemetry_publisher=publisher,
    )

    for _ in range(5):
        publisher._last_publish = 0.0  # bypass rate limiting
        await orch.tick()

    assert publisher.stats["frames_published"] == 5
    q = publisher.get_queue()
    assert q.qsize() == 5


async def test_log_buffer_integration():
    """Log buffer captures entries and can be retrieved."""
    buf = LogRingBuffer(maxlen=50)

    # Simulate log entries
    for i in range(10):
        buf(None, "info", {"event": f"msg_{i}", "level": "info"})

    assert buf.size == 10
    recent = buf.get_recent(5)
    assert len(recent) == 5
    assert recent[-1]["event"] == "msg_9"


async def test_publisher_consumer_pattern():
    """Publisher produces, consumer reads from same queue."""
    cfg = TelemetryConfig(enabled=True, publish_hz=60.0, queue_size=32)
    publisher = TelemetryPublisher(cfg)
    q = publisher.get_queue()

    consumed = []

    async def consumer():
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=0.5)
                consumed.append(frame)
            except asyncio.TimeoutError:
                break

    # Produce
    for i in range(3):
        publisher._last_publish = 0.0
        await publisher.publish(TelemetryFrame(tick_count=i))

    # Consume
    await consumer()
    assert len(consumed) == 3
    assert [f.tick_count for f in consumed] == [0, 1, 2]
