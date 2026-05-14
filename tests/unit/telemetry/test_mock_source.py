"""Tests for :class:`MockTelemetrySource`."""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.mock_source import MockTelemetrySource
from mousedroid.telemetry.publisher import TelemetryPublisher


@pytest.fixture
def cfg() -> TelemetryConfig:
    return TelemetryConfig(
        enabled=True,
        publish_hz=30.0,
        lidar_raw_publish_hz=30.0,
        queue_size=64,
        lidar_raw_queue_size=64,
    )


async def test_emits_frames_and_scans(cfg: TelemetryConfig) -> None:
    """A short run produces at least one frame and one raw scan."""
    publisher = TelemetryPublisher(cfg)
    source = MockTelemetrySource(cfg, publisher)
    await source.start()
    try:
        await asyncio.sleep(0.2)
    finally:
        await source.stop()
    assert publisher.stats["frames_published"] > 0
    assert publisher.stats["lidar_raw_published"] > 0


async def test_stop_is_idempotent(cfg: TelemetryConfig) -> None:
    publisher = TelemetryPublisher(cfg)
    source = MockTelemetrySource(cfg, publisher)
    await source.start()
    await source.stop()
    await source.stop()  # second stop must not raise


async def test_start_twice_keeps_single_task(cfg: TelemetryConfig) -> None:
    publisher = TelemetryPublisher(cfg)
    source = MockTelemetrySource(cfg, publisher)
    await source.start()
    first_task = source._task  # type: ignore[attr-defined]
    await source.start()
    second_task = source._task  # type: ignore[attr-defined]
    assert first_task is second_task
    await source.stop()


async def test_run_loop_recovers_from_emit_exception(cfg: TelemetryConfig) -> None:
    """A transient ``_emit_frame`` failure must not terminate the loop."""

    publisher = TelemetryPublisher(cfg)
    source = MockTelemetrySource(cfg, publisher)
    calls = {"count": 0}

    async def _flaky_emit_frame() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient")
        # subsequent calls succeed by delegating to the real method
        # (no-op for this test — we only need to prove the loop keeps
        # running after the first exception).

    source._emit_frame = _flaky_emit_frame  # type: ignore[method-assign]
    await source.start()
    try:
        await asyncio.sleep(0.2)
    finally:
        await source.stop()
    # Loop kept ticking through the exception → emit was called
    # multiple times instead of bailing after the first raise.
    assert calls["count"] >= 2


async def test_synthetic_scan_shape(cfg: TelemetryConfig) -> None:
    publisher = TelemetryPublisher(cfg)
    source = MockTelemetrySource(cfg, publisher, lidar_points_per_scan=16)
    await source.start()
    try:
        await asyncio.sleep(0.15)
    finally:
        await source.stop()
    queue = publisher.get_lidar_raw_queue()
    # Drain and inspect the freshest scan.
    last = None
    while not queue.empty():
        last = queue.get_nowait()
    assert last is not None
    assert last.n_points == 16
    assert len(last.angles_rad) == 16
    assert len(last.distances_m) == 16
    for d in last.distances_m:
        assert 0.0 < d <= 8.0
