"""Smoke test — writer-side instrumentation produces visible metric output.

Runs in well under 1 second; lives under ``pytest -m smoke`` so the CI
smoke stage catches any wiring regression in the four PR-A2 metric
families before the slower integration suite runs.

In-process only: builds a real :class:`MetricsRegistry`, drives the
three subsystems once each, calls ``render_prometheus()``, and asserts
the families appear. No HTTP server, no aiohttp client.
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
    VLMProgressConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.telemetry.metrics import MetricsRegistry
from tests import TEST_EXPERIENCE_MAP_SIZE_GB

pytestmark = pytest.mark.smoke


def _populate_replay(path: Path, *, n: int = 2) -> None:
    """Write N deterministic records to an LMDB store."""
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


def test_writer_side_instrumentation_smoke(tmp_path: Path) -> None:
    """Each subsystem populates its metric family in <1 second (in-process)."""
    registry = MetricsRegistry(MetricsConfig())

    # 1. Replay reader (async stream, drained synchronously here).
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    _populate_replay(tmp_path, n=2)
    reader = LMDBReplayReader(_experience_cfg(tmp_path), metrics=registry)

    async def _drive_reader() -> None:
        async for _chunk in reader.stream(chunk_size=10):
            pass

    asyncio.run(_drive_reader())

    # 2. MockVLA (sync inference).
    from mousedroid.vla.policy import MockVLA, VLAObservation

    MockVLA(action_dim=3, metrics=registry).predict(
        VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))
    )

    # 3. VLMProgressHead (one miss is enough for smoke).
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from tests.unit.reward.test_vlm_progress import _CountingBackend

    head = VLMProgressHead(
        VLMProgressConfig(cache_size=8),
        backend=_CountingBackend(),
        metrics=registry,
    )
    head.score(torch.zeros(1, 4), torch.ones(1, 4), instruction="go")

    text = registry.render_prometheus()
    ns = MetricsConfig().namespace

    # Assert each reliably-fired PR-A2 family appears.
    assert (
        f"{ns}_replay_records_total" in text
    ), f"replay_records_total missing from /metrics output:\n{text[:500]}..."
    assert (
        f"{ns}_vla_inference_seconds_count" in text
    ), f"vla_inference_seconds_count missing from /metrics output:\n{text[:500]}..."
    assert (
        f"{ns}_vlm_progress_cache_misses_total" in text
    ), f"vlm_progress_cache_misses_total missing from /metrics output:\n{text[:500]}..."
