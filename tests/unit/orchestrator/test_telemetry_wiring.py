"""Tests for PR #4 orchestrator wiring: raw LiDAR + sensor liveness.

Verifies the orchestrator actually:

1. Threads a ``SensorLivenessTracker`` through ``build_telemetry_frame``
   so every published frame carries a non-empty ``sensor_liveness`` map.
2. Calls ``publisher.publish_lidar_raw`` with the converted raw scan
   when the sensor manager exposes ``last_lidar_scan``.
3. Starts and stops the optional ``mock_telemetry_source`` alongside
   the rest of the lifecycle.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.sensing.lidar_scan import LidarScan
from mousedroid.telemetry.protocol import LidarRawScan
from mousedroid.telemetry.sensor_liveness import SensorLivenessTracker


def _make_observation() -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=1.0,
        _vision_features=np.zeros(8, dtype=np.float32),
        _distance_m=0.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 7.4], dtype=np.float32),
        _audio_chunk=np.zeros(0, dtype=np.float32),
        _valid_mask=np.ones(4, dtype=np.float32),
        _lidar_features=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.8, 0.7], dtype=np.float32),
    )


def _make_orch(
    *,
    publisher: Any | None,
    liveness_tracker: Any | None = None,
    mock_telemetry_source: Any | None = None,
    last_lidar_scan: Any | None = None,
    cfg: Settings | None = None,
) -> MouseDroidOrchestrator:
    cfg = cfg if cfg is not None else Settings(mock_hardware=True)
    world_model = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    world_model.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )

    agent = MagicMock()
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=_make_observation())
    sensor_manager.last_lidar_scan = last_lidar_scan

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        clock=MockClock(start=10.0),
        telemetry_publisher=publisher,
        liveness_tracker=liveness_tracker,
        mock_telemetry_source=mock_telemetry_source,
    )


@pytest.mark.asyncio
async def test_publish_telemetry_threads_liveness_tracker() -> None:
    """Frames published by the orchestrator carry the sensor_liveness map."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()

    tracker = SensorLivenessTracker(stale_s=10.0)
    tracker.register("lidar", enabled=True)
    tracker.register("vision", enabled=True)

    orch = _make_orch(publisher=publisher, liveness_tracker=tracker)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish.assert_called_once()
    published = publisher.publish.call_args.args[0]
    assert "lidar" in published.sensor_liveness
    assert published.sensor_liveness["lidar"]["state"] == "live"


@pytest.mark.asyncio
async def test_publish_telemetry_honours_configured_vision_feature_max_samples() -> None:
    """The orchestrator threads TelemetryConfig.vision_feature_max_samples through.

    Regression target: ``build_telemetry_frame``'s ``vision_feature_max_samples``
    kwarg has a real, validated ``Field`` on ``TelemetryConfig`` and is threaded
    all the way through -- but the orchestrator's own call site never passed it,
    so the field silently had zero effect on the running system. This test
    fails (published frame carries all 8 samples, not 3) against that call site.
    """
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()

    cfg = Settings(mock_hardware=True)
    cfg.telemetry.vision_feature_max_samples = 3

    orch = _make_orch(publisher=publisher, cfg=cfg)
    await orch._publish_telemetry(
        _make_observation(),  # 8-element vision_features -- exceeds the cap
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    published = publisher.publish.call_args.args[0]
    assert published.vision_features is not None
    assert len(published.vision_features) == 3


@pytest.mark.asyncio
async def test_publish_telemetry_calls_publish_lidar_raw_when_scan_available() -> None:
    """``publish_lidar_raw`` is invoked when sensor_manager has a fresh scan."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()

    scan = LidarScan(
        angles_deg=np.array([0.0, 90.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 2000.0], dtype=np.float32),
        confidences=np.array([255, 255], dtype=np.uint8),
        timestamp=42.0,
        n_points=2,
    )
    orch = _make_orch(publisher=publisher, last_lidar_scan=scan)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish_lidar_raw.assert_called_once()
    raw = publisher.publish_lidar_raw.call_args.args[0]
    assert isinstance(raw, LidarRawScan)
    assert raw.n_points == 2
    assert raw.distances_m == [1.0, 2.0]


@pytest.mark.asyncio
async def test_publish_telemetry_skips_raw_when_publisher_lacks_method() -> None:
    """Legacy publishers without ``publish_lidar_raw`` are tolerated."""
    publisher = MagicMock(spec=["publish", "get_queue", "stats"])
    publisher.publish = AsyncMock()
    orch = _make_orch(publisher=publisher)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish.assert_called_once()  # no crash


@pytest.mark.asyncio
async def test_publish_raw_lidar_swallows_conversion_error() -> None:
    """If ``lidar_scan_to_raw`` raises, the orchestrator logs and returns."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()
    # A scan that's not a LidarScan (no angles_deg attr) trips the
    # conversion adapter — orchestrator must swallow the AttributeError.
    bad_scan = object()
    orch = _make_orch(publisher=publisher, last_lidar_scan=bad_scan)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish.assert_called_once()
    publisher.publish_lidar_raw.assert_not_called()


@pytest.mark.asyncio
async def test_publish_raw_lidar_swallows_publish_error() -> None:
    """A failing ``publish_lidar_raw`` does not crash the control loop."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock(side_effect=RuntimeError("queue full"))
    scan = LidarScan(
        angles_deg=np.array([0.0], dtype=np.float32),
        distances_mm=np.array([1000.0], dtype=np.float32),
        confidences=np.array([255], dtype=np.uint8),
        timestamp=1.0,
        n_points=1,
    )
    orch = _make_orch(publisher=publisher, last_lidar_scan=scan)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish.assert_called_once()  # main publish still runs
    publisher.publish_lidar_raw.assert_called_once()  # attempted once


@pytest.mark.asyncio
async def test_mock_telemetry_source_start_failure_is_logged_not_fatal() -> None:
    """``orchestrator.start`` tolerates a mock_telemetry_source that raises."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()
    bad_source = MagicMock()
    bad_source.start = AsyncMock(side_effect=RuntimeError("synth broken"))
    bad_source.stop = AsyncMock()

    orch = _make_orch(publisher=publisher, mock_telemetry_source=bad_source)
    # We exercise just the start path; stop runs the suppress-context.
    # Stub the sensor manager start so it doesn't try real I/O.
    orch._sensor_manager.start = AsyncMock()  # type: ignore[method-assign]
    await orch.start()
    bad_source.start.assert_called_once()
    assert orch._running  # start did not raise


@pytest.mark.asyncio
async def test_publish_telemetry_skips_raw_when_no_scan_cached() -> None:
    """No ``last_lidar_scan`` → ``publish_lidar_raw`` is not called."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    publisher.publish_lidar_raw = AsyncMock()
    orch = _make_orch(publisher=publisher, last_lidar_scan=None)
    await orch._publish_telemetry(
        _make_observation(),
        SafetyContext(is_emergency=False),
        loop_time_ms=20.0,
    )
    publisher.publish.assert_called_once()
    publisher.publish_lidar_raw.assert_not_called()
