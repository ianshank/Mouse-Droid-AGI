"""Integration: WS3 on-device update wired through ``build_orchestrator``.

Proves the Phase-6 WS3 slow-cadence path end-to-end via the factory:

* with ``on_device_learning.enabled=True`` + a seeded replay store under the
  configured experience root, the orchestrator builds a coordinator and
  spawns the slow background task in ``start()``;
* driving the coordinator produces + persists a SHA-256-stamped candidate slot
  UNDER ``<experience.path>/<slot_dir>`` (the WS0 de-hardcode realized);
* the produce/persist path does NOT advance the 30 Hz hot loop — ``_tick_count``
  stays 0 (the slow task is fully isolated from ``tick()``);
* ``stop()`` cancels the slow task cleanly.

Built with ``mock_hardware=True`` so no real device is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_orchestrator
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore

_N_SEEDED = 6
_TRIGGER = 5


def _seed_replay_store(experience_path: str, n: int) -> None:
    """Write ``n`` experience records to an LMDB store at ``experience_path``."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
        }
    )
    logger = ExperienceLogger(cfg.experience)
    logger.open()
    try:
        for _ in range(n):
            logger.log(MouseDroidExperienceRecord())
    finally:
        logger.close()


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
            },
        }
    )


@pytest.mark.asyncio
async def test_coordinator_produces_stamped_slot_without_ticking(tmp_path: Path) -> None:
    """The wired coordinator persists a stamped slot; the hot loop never runs."""
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
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

    # The 30 Hz hot loop was NEVER advanced by the on-device update.
    assert orchestrator._tick_count == 0


@pytest.mark.asyncio
async def test_start_spawns_then_stop_cancels_slow_task(tmp_path: Path) -> None:
    """``start()`` spawns the slow task and ``stop()`` cancels it cleanly."""
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
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
