"""End-to-end integration: writer-side instrumentation → /metrics HTTP scrape.

Spins up a real :class:`TelemetryServer` against a real
:class:`MetricsRegistry`, drives each instrumented subsystem (replay
reader, MockVLA, VLMProgressHead), hits ``GET /metrics`` over aiohttp,
and asserts the four PR-A2 metric families appear in the rendered
Prometheus exposition output.

This is the test that proves Tier A is no longer match-zero. If it
passes, the Grafana panels shipped in PR-B2 will populate the first
time their queries fire against a live Prometheus scrape.
"""

from __future__ import annotations

import asyncio
import struct
import time
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch

from mousedroid.config.schema import (
    ExperienceConfig,
    MetricsConfig,
    TelemetryConfig,
    VLMProgressConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame
from tests import TEST_EXPERIENCE_MAP_SIZE_GB

aiohttp = pytest.importorskip("aiohttp")

from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer


def _populate_replay(path: Path, *, n: int) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    base = time.time()
    with env.begin(write=True) as txn:
        for i in range(n):
            record = MouseDroidExperienceRecord(
                timestamp=base + i * 0.1,
                vision_features=np.zeros(256, dtype=np.float32),
                distance_m=1.0,
                motor_state=np.zeros(4, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=float(i),
                surprise=0.0,
            )
            key = struct.pack(">Q", int(base * 1_000_000) + i)
            txn.put(key, record.serialize())
    env.close()


def _experience_cfg(path: Path) -> ExperienceConfig:
    return ExperienceConfig(
        path=str(path),
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        flush_every_n=5,
    )


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={"status": "ok", "gpu_temp_c": 42.0, "gpu_load_pct": 15.0}
    )
    return monitor


def _build_telemetry_app(registry: MetricsRegistry) -> tuple[TelemetryServer, web.Application]:
    """Construct a minimal TelemetryServer + aiohttp app exposing /metrics.

    Mirrors the unsecured pattern from
    ``tests/integration/test_telemetry_secured.py::_make_secured_server`` but
    leaves auth disabled so the metrics scrape needs no token plumbing.
    """
    cfg = TelemetryConfig(
        enabled=True,
        api_key=None,
        metrics_path="/metrics",
    )
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=_make_health_monitor(),
        metrics_registry=registry,
        metrics_path="/metrics",
    )
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return server, app


async def _drive_subsystems(registry: MetricsRegistry, tmp_path: Path) -> None:
    """Fire every instrumented call site so the registry has data to surface."""
    # 1. Replay reader
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    _populate_replay(tmp_path, n=3)
    reader = LMDBReplayReader(_experience_cfg(tmp_path), metrics=registry)
    async for _chunk in reader.stream(chunk_size=10):
        pass

    # 2. MockVLA
    from mousedroid.vla.policy import MockVLA, VLAObservation

    MockVLA(action_dim=3, metrics=registry).predict(
        VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))
    )

    # 3. VLMProgressHead — fire both a miss and a hit so both counters appear
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from tests.unit.reward.test_vlm_progress import _CountingBackend

    head = VLMProgressHead(
        VLMProgressConfig(cache_size=8),
        backend=_CountingBackend(),
        metrics=registry,
    )
    prev = torch.zeros(1, 4)
    curr = torch.ones(1, 4)
    head.score(prev, curr, instruction="go")  # miss → populate cache
    head.score(prev, curr, instruction="go")  # identity-cache hit


@pytest.mark.parametrize("metrics_present", [True, False])
async def test_metrics_endpoint_surfaces_pr_a2_families(
    tmp_path: Path, metrics_present: bool
) -> None:
    """End-to-end: /metrics HTTP scrape exposes all PR-A2 families when wired.

    Parametrized over ``metrics_present`` so we verify both:
    - the **active** path (registry threaded → counters increment → /metrics
      exposes them)
    - the **default** path (no registry → subsystems remain byte-identical;
      /metrics renders without the new families and without errors)
    """
    if metrics_present:
        registry = MetricsRegistry(MetricsConfig())
        await _drive_subsystems(registry, tmp_path)
    else:
        # No-op: registry built but never observed. The subsystems still
        # run (sanity smoke) but with metrics=None — proving the default
        # path doesn't crash mid-burst.
        registry = MetricsRegistry(MetricsConfig())

        async def _drive_without_metrics() -> None:
            from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
            from mousedroid.vla.policy import MockVLA, VLAObservation

            _populate_replay(tmp_path, n=2)
            reader = LMDBReplayReader(_experience_cfg(tmp_path))  # no metrics
            async for _chunk in reader.stream(chunk_size=10):
                pass
            MockVLA(action_dim=3).predict(VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4)))

        await _drive_without_metrics()

    _server, app = _build_telemetry_app(registry)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/metrics")
        assert resp.status == 200
        body = await resp.text()

    ns = MetricsConfig().namespace

    if metrics_present:
        assert (
            f'{ns}_replay_records_total{{outcome="ok"}}' in body
        ), f"replay_records_total missing from /metrics scrape:\n{body[:1500]}..."
        assert (
            f"{ns}_vla_inference_seconds_count" in body
        ), f"vla_inference_seconds_count missing from /metrics scrape:\n{body[:1500]}..."
        assert (
            f"{ns}_vlm_progress_cache_hits_total" in body
        ), f"vlm_progress_cache_hits_total missing from /metrics scrape:\n{body[:1500]}..."
        assert (
            f"{ns}_vlm_progress_cache_misses_total" in body
        ), f"vlm_progress_cache_misses_total missing from /metrics scrape:\n{body[:1500]}..."
    else:
        # Default path: families are conditionally rendered (PR-A2
        # design) and must not appear when never observed.
        assert f"{ns}_replay_records_total" not in body
        assert f"{ns}_vla_inference_seconds_count" not in body
        assert f"{ns}_vlm_progress_cache_hits_total" not in body
        assert f"{ns}_vlm_progress_cache_misses_total" not in body
