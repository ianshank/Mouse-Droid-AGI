"""Concurrent E2E: all three subsystems fire in parallel, registry stays consistent.

Production hits these call sites from different threads / async contexts.
This test proves:

1. No deadlock / race when multiple subsystems write to the registry
   simultaneously.
2. Counter totals are exact under concurrent writes (relies on the
   existing ``threading.Lock`` in ``_Counter`` / ``_LabeledCounter`` /
   ``_Histogram``).
3. Rendered Prometheus output is well-formed even mid-burst.
"""

from __future__ import annotations

import asyncio
import re
import struct
import threading
import time
from pathlib import Path

import lmdb
import numpy as np
import torch

from mousedroid.config.schema import (
    ExperienceConfig,
    MetricsConfig,
    VLMProgressConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.telemetry.metrics import MetricsRegistry
from tests import TEST_EXPERIENCE_MAP_SIZE_GB

_REPLAY_RECORDS = 30
_VLA_CALLS = 30
_VLM_CALLS = 30


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


def test_three_subsystems_concurrent_emission(tmp_path: Path) -> None:
    """Drive replay (async) + VLA (sync thread) + VLM (sync thread) in parallel."""
    registry = MetricsRegistry(MetricsConfig())

    # Replay
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    _populate_replay(tmp_path, n=_REPLAY_RECORDS)
    reader = LMDBReplayReader(_experience_cfg(tmp_path), metrics=registry)

    # VLA
    from mousedroid.vla.policy import MockVLA, VLAObservation

    vla = MockVLA(action_dim=3, metrics=registry)
    obs = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))

    # VLM
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from tests.unit.reward.test_vlm_progress import _CountingBackend

    vlm = VLMProgressHead(
        VLMProgressConfig(cache_size=64),
        backend=_CountingBackend(),
        metrics=registry,
    )

    def _vla_worker() -> None:
        for _ in range(_VLA_CALLS):
            vla.predict(obs)

    def _vlm_worker() -> None:
        # Alternate two distinct (prev, curr) pairs to create both misses
        # and content-cache hits. Same tensor objects on repeat → identity-
        # cache hits as well.
        prev_a = torch.zeros(1, 4)
        curr_a = torch.ones(1, 4)
        prev_b = torch.ones(1, 4)
        curr_b = torch.zeros(1, 4)
        for i in range(_VLM_CALLS):
            if i % 2 == 0:
                vlm.score(prev_a, curr_a, instruction="x")
            else:
                vlm.score(prev_b, curr_b, instruction="x")

    vla_thread = threading.Thread(target=_vla_worker, name="vla-worker")
    vlm_thread = threading.Thread(target=_vlm_worker, name="vlm-worker")
    vla_thread.start()
    vlm_thread.start()

    async def _drive_reader() -> None:
        async for _chunk in reader.stream(chunk_size=10):
            pass

    asyncio.run(_drive_reader())

    vla_thread.join(timeout=30.0)
    vlm_thread.join(timeout=30.0)
    assert not vla_thread.is_alive(), "VLA worker deadlocked"
    assert not vlm_thread.is_alive(), "VLM worker deadlocked"

    text = registry.render_prometheus()
    ns = MetricsConfig().namespace

    # Replay reader emitted exactly N "ok" outcomes (count must be exact
    # — concurrent writes to the same labeled counter must not lose any).
    assert (
        f'{ns}_replay_records_total{{outcome="ok"}} {_REPLAY_RECORDS}' in text
    ), f"replay ok count off — rendered output:\n{text[:1500]}..."

    # VLA histogram saw exactly _VLA_CALLS observations.
    assert (
        f"{ns}_vla_inference_seconds_count {_VLA_CALLS}" in text
    ), f"vla inference count off — rendered output:\n{text[:1500]}..."

    # VLM cache emitted hits + misses summing to _VLM_CALLS. Exact hit/miss
    # split depends on the alternation + LRU eviction order; only the sum
    # is guaranteed.
    hit_match = re.search(rf"{ns}_vlm_progress_cache_hits_total (\d+)", text)
    miss_match = re.search(rf"{ns}_vlm_progress_cache_misses_total (\d+)", text)
    hits = int(hit_match.group(1)) if hit_match else 0
    misses = int(miss_match.group(1)) if miss_match else 0
    assert hits + misses == _VLM_CALLS, (
        f"VLM hits ({hits}) + misses ({misses}) != _VLM_CALLS ({_VLM_CALLS}). "
        f"Output:\n{text[:1500]}..."
    )


def test_render_during_concurrent_writes_is_well_formed() -> None:
    """A scrape mid-burst still produces valid Prometheus text format."""
    registry = MetricsRegistry(MetricsConfig())
    stop = threading.Event()

    def _hot_loop() -> None:
        from mousedroid.vla.policy import MockVLA, VLAObservation

        vla = MockVLA(action_dim=3, metrics=registry)
        obs = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))
        while not stop.is_set():
            vla.predict(obs)

    worker = threading.Thread(target=_hot_loop, daemon=True)
    worker.start()
    try:
        # Scrape 10 times during the burst; every scrape must produce
        # valid output.
        for _ in range(10):
            text = registry.render_prometheus()
            assert text.endswith("\n"), "render_prometheus output missing trailing newline"
            # Every metric line must follow "name [labels] value" shape.
            for line in text.splitlines():
                if line and not line.startswith("#"):
                    parts = line.rsplit(" ", 1)
                    assert len(parts) == 2, f"malformed metric line: {line!r}"
            time.sleep(0.005)  # let the worker get a few writes between scrapes
    finally:
        stop.set()
        worker.join(timeout=5.0)
