"""Integration: WS3 on-device update wired through ``build_orchestrator``.

Proves the Phase-6 WS3 slow-cadence path end-to-end via the factory, now over
the WS-E2 RSSM-refinement learner (the ``EWCOnlineLearner``-over-``nn.Linear``
stand-in has been replaced by :class:`RSSMRefiner`):

* with ``on_device_learning.enabled=True`` + a seeded replay store under the
  configured experience root, the orchestrator builds a coordinator and
  spawns the slow background task in ``start()``;
* driving the coordinator produces + persists a SHA-256-stamped candidate slot
  UNDER ``<experience.path>/<slot_dir>`` (the WS0 de-hardcode realized). The slot
  now holds a REFINED RSSM ``state_dict`` (a ``(B, T, ...)`` sequence-dict batch
  drives ``train_sequence``), which round-trips strictly into a fresh
  ``build_world_model(cfg)`` with NO ``decode_*`` decoder keys;
* the produce/persist path does NOT advance the 30 Hz hot loop — ``_tick_count``
  stays 0 (the slow task is fully isolated from ``tick()``);
* ``stop()`` cancels the slow task cleanly.

The seeded record count + the small ``refine_sequence_length`` /
``refine_batch_episodes`` are sized so one batch of
``refine_batch_episodes * refine_sequence_length`` records is always fillable
once the trigger fires.

Built with ``mock_hardware=True`` so no real device is required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_orchestrator, build_world_model
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore

# Refine batch geometry: 2 episodes x 3 steps = 6 records per batch.
_REFINE_SEQUENCE_LENGTH = 3
_REFINE_BATCH_EPISODES = 2
# Seed comfortably above both the trigger AND one full refine batch.
_N_SEEDED = 8
_TRIGGER = 5


async def _seed_replay_store(experience_path: str, n: int) -> None:
    """Write ``n`` experience records to an LMDB store at ``experience_path``.

    The LMDB writes are blocking disk I/O, so they run on a worker thread via
    ``asyncio.to_thread`` — these helpers are called from ``asyncio`` tests and
    must not block the event loop (CodeRabbit review; repo async-IO contract).
    """
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
        }
    )

    def _write_records() -> None:
        logger = ExperienceLogger(cfg.experience)
        logger.open()
        try:
            for _ in range(n):
                logger.log(MouseDroidExperienceRecord())
        finally:
            logger.close()

    await asyncio.to_thread(_write_records)


def _build_cfg(experience_path: str) -> Settings:
    """Settings with on-device learning enabled + a low trigger threshold."""
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": _TRIGGER,
                "update_steps": 2,
                "check_interval_s": 0.01,
                "refine_sequence_length": _REFINE_SEQUENCE_LENGTH,
                "refine_batch_episodes": _REFINE_BATCH_EPISODES,
            },
        }
    )


@pytest.mark.asyncio
async def test_coordinator_produces_stamped_slot_without_ticking(tmp_path: Path) -> None:
    """The wired coordinator persists a REFINED RSSM slot; the hot loop never runs."""
    experience_path = str(tmp_path / "experience_root")
    await _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    orchestrator = build_orchestrator(cfg)

    coordinator = orchestrator._on_device_coordinator
    assert coordinator is not None

    slot = await coordinator.maybe_update()

    # A stamped candidate slot landed under <experience.path>/<slot_dir>.
    assert slot is not None
    assert len(slot.digest) == 64
    expected_dir = (Path(experience_path) / cfg.on_device_learning.slot_dir).resolve()
    assert slot.path.resolve().parent == expected_dir

    # The slot is loadable + integrity-verified by the store.
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    loaded = store.load(slot)
    assert loaded

    # WS-E2: the slot is a REFINED RSSM state_dict — it carries the RSSM's keys
    # (NOT the throwaway reconstruction decoders) and strict-loads into a fresh
    # ``build_world_model(cfg)`` of identical dims.
    assert not any(key.startswith("decode") for key in loaded), "decoder keys leaked into slot"
    fresh = build_world_model(cfg)
    result = fresh.load_state_dict(loaded, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    # The 30 Hz hot loop was NEVER advanced by the on-device update.
    assert orchestrator._tick_count == 0


@pytest.mark.asyncio
async def test_start_spawns_then_stop_cancels_slow_task(tmp_path: Path) -> None:
    """``start()`` spawns the slow task and ``stop()`` cancels it cleanly."""
    experience_path = str(tmp_path / "experience_root")
    await _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    orchestrator = build_orchestrator(cfg)

    await orchestrator.start()
    try:
        task = orchestrator._on_device_task
        assert task is not None
        assert not task.done()
    finally:
        await orchestrator.stop()

    assert orchestrator._on_device_task is None
    # The hot loop still never ran (only the slow task was active).
    assert orchestrator._tick_count == 0


@pytest.mark.asyncio
async def test_slow_loop_runs_a_cycle_and_persists(tmp_path: Path) -> None:
    """A live slow-cadence loop iteration produces + persists a stamped slot.

    Drives the real ``_on_device_update_loop`` body (not just spawn/cancel) by
    starting the orchestrator and polling until the coordinator has written a
    slot under ``<experience.path>/<slot_dir>``.
    """
    experience_path = str(tmp_path / "experience_root")
    await _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    orchestrator = build_orchestrator(cfg)
    slot_dir = (Path(experience_path) / cfg.on_device_learning.slot_dir).resolve()

    await orchestrator.start()
    try:
        for _ in range(200):
            if slot_dir.exists() and list(slot_dir.glob("*.pt")):
                break
            await asyncio.sleep(0.02)
    finally:
        await orchestrator.stop()

    persisted = list(slot_dir.glob("*.pt")) if slot_dir.is_dir() else []
    assert persisted, "slow-cadence loop did not persist a candidate slot"
    # Hot loop was never advanced by the slow-cadence update.
    assert orchestrator._tick_count == 0


@pytest.mark.asyncio
async def test_slow_loop_survives_a_failing_cycle(tmp_path: Path) -> None:
    """A coordinator exception is logged and the loop keeps running.

    Pins the resilience branch of ``_on_device_update_loop`` (the broad
    ``except`` that logs ``on_device_update_cycle_failed`` and continues) so a
    transient replay-store / disk error never kills on-device learning.
    """
    experience_path = str(tmp_path / "experience_root")
    await _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    orchestrator = build_orchestrator(cfg)

    calls: list[int] = []

    class _BoomCoordinator:
        async def maybe_update(self) -> None:
            calls.append(1)
            raise RuntimeError("transient replay-store error")

    orchestrator._on_device_coordinator = _BoomCoordinator()  # type: ignore[assignment]

    await orchestrator.start()
    try:
        task = orchestrator._on_device_task
        assert task is not None
        for _ in range(200):
            if calls:
                break
            await asyncio.sleep(0.02)
        # Give it a beat to loop again after the failure (proving it survived).
        await asyncio.sleep(0.05)
        # The slow task is STILL alive after the exception — the broad-except
        # in _on_device_update_loop swallowed it and kept looping (it did not
        # die early and only get cleared later by stop()).
        assert not task.done()
    finally:
        await orchestrator.stop()

    assert calls, "the failing cycle never executed"
    # stop() cancelled + cleared the task.
    assert orchestrator._on_device_task is None
