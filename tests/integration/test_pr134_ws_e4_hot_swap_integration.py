"""Integration: WS-E4 off-loop hot-swap of a promoted slot wired through the factory.

Proves the SAFETY-LOCKED WS-E4 path end-to-end via the factory + orchestrator:

* with ``enable_hot_swap=True`` + a marked-active slot, the orchestrator's live
  ``self._world_model`` is REPLACED (identity changes) by the materialised slot
  engine through the C1 atomic-swap seam, AND the recurrent ``(h, z)`` latent
  state is reset to zeros;
* the MATERIALISATION (``torch.load`` + ``build_world_model`` + ``load_state_dict``)
  happens OFF the hot loop — the source's ``refresh_once`` builds the engine; the
  hot-loop loader (``take_materialized``) is a PURE reference return that does NO
  I/O (spied: it is NOT what builds the engine, and the per-tick swap is cheap);
* DEVICE PARITY: the swapped engine lives on the SAME device as the prior live
  model;
* a CORRUPT slot is fail-closed: ``inc_on_device_learning_reverted(
  "integrity_mismatch")`` increments off-loop AND the live ``self._world_model``
  identity is UNCHANGED (no swap).

Built with ``mock_hardware=True`` so no real device is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import (
    build_on_device_hot_swap_source,
    build_orchestrator,
    build_world_model,
)
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.rssm import RSSM

_N_SEEDED = 8


def _seed_replay_store(experience_path: str, n: int) -> None:
    cfg = Settings.model_validate(
        {"mock_hardware": True, "experience": {"path": experience_path, "map_size_gb": 0.01}}
    )
    logger = ExperienceLogger(cfg.experience)
    logger.open()
    try:
        for _ in range(n):
            logger.log(MouseDroidExperienceRecord())
    finally:
        logger.close()


def _build_cfg(experience_path: str, *, enable_hot_swap: bool) -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "enable_hot_swap": enable_hot_swap,
                "trigger_min_new_records": 5,
                "check_interval_s": 0.01,
                "refine_sequence_length": 3,
                "refine_batch_episodes": 2,
            },
        }
    )


def _persist_active_slot(cfg: Settings) -> tuple[OnDeviceSlotStore, str]:
    """Persist a state-dict-shaped slot from a fresh RSSM + mark it active."""
    assert cfg.on_device_learning is not None
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)
    slot = store.persist(wm.state_dict())
    store.mark_active(slot)
    return store, slot.digest


@pytest.mark.asyncio
async def test_disabled_no_hot_swap_source_wired(tmp_path: Path) -> None:
    """``enable_hot_swap=False`` ⇒ NO on-device source in the pollers mapping."""
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, enable_hot_swap=False)
    # Even with a marked-active slot present, a disabled flag wires no source.
    _persist_active_slot(cfg)

    orch = build_orchestrator(cfg)
    pollers = orch._weight_update_pollers  # type: ignore[attr-defined]
    # No world-model poller from the on-device path (cloud OTA is also off here).
    assert "world_model" not in pollers


@pytest.mark.asyncio
async def test_disabled_factory_helper_returns_none(tmp_path: Path) -> None:
    """``build_on_device_hot_swap_source`` returns None when the flag is off."""
    experience_path = str(tmp_path / "experience_root")
    cfg = _build_cfg(experience_path, enable_hot_swap=False)
    wm = build_world_model(cfg)
    source = build_on_device_hot_swap_source(cfg, world_model=wm)
    assert source is None


@pytest.mark.asyncio
async def test_enabled_swaps_live_world_model_off_loop(tmp_path: Path) -> None:
    """An active slot is swapped into the live world model + ``(h, z)`` reset."""
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, enable_hot_swap=True)
    _persist_active_slot(cfg)

    orch = build_orchestrator(cfg)
    source = orch._weight_update_pollers["world_model"]  # type: ignore[attr-defined]

    prior_model = orch._world_model  # type: ignore[attr-defined]
    prior_device = next(prior_model.parameters()).device

    # OFF-LOOP materialisation: the source's refresh builds the engine. Spy the
    # hot-loop loader to PROVE it does NOT do the materialisation.
    materialise_calls: list[str] = []
    original_take = source.take_materialized

    def _spy_take(update: object) -> object:
        materialise_calls.append("take_materialized")
        return original_take(update)

    source.take_materialized = _spy_take  # type: ignore[method-assign]

    surfaced = await source.refresh_once()
    assert surfaced is True
    # The loader was NOT called by refresh (materialisation is off the loop).
    assert materialise_calls == []

    # Drive ONE tick's swap helper (the only hot-loop touch-point).
    reset = orch._apply_pending_weight_update()  # type: ignore[attr-defined]

    assert reset is True  # world-model swap resets the recurrent state
    new_model = orch._world_model  # type: ignore[attr-defined]
    assert new_model is not prior_model  # identity changed -> atomic swap landed
    # The loader fired exactly once during the tick (pure reference return).
    assert materialise_calls == ["take_materialized"]
    # Device parity: swapped engine on the SAME device as the prior live model.
    assert next(new_model.parameters()).device == prior_device
    # Recurrent state zeroed.
    assert torch.count_nonzero(orch._h) == 0  # type: ignore[attr-defined]
    assert torch.count_nonzero(orch._z) == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_enabled_hot_loop_swap_is_cheap_reference_assignment(tmp_path: Path) -> None:
    """The hot-loop swap does NO disk I/O — proven by spying the materialiser.

    The materialise callable (off-loop, builds the engine) must be invoked by
    ``refresh_once`` and NOT during ``_apply_pending_weight_update`` (the tick).
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, enable_hot_swap=True)
    _persist_active_slot(cfg)

    orch = build_orchestrator(cfg)
    source = orch._weight_update_pollers["world_model"]  # type: ignore[attr-defined]

    # Spy the OFF-LOOP materialiser.
    original_mat = source._materialize  # type: ignore[attr-defined]
    mat_calls: list[str] = []

    def _spy_mat(digest: str) -> object:
        mat_calls.append(digest)
        return original_mat(digest)

    source._materialize = _spy_mat  # type: ignore[attr-defined]

    await source.refresh_once()
    assert len(mat_calls) == 1  # materialised ONCE, off the loop

    # The tick's swap must NOT re-materialise.
    orch._apply_pending_weight_update()  # type: ignore[attr-defined]
    assert len(mat_calls) == 1


@pytest.mark.asyncio
async def test_corrupt_slot_fail_closed_live_model_unchanged(tmp_path: Path) -> None:
    """A corrupt active slot reverts (integrity_mismatch) + leaves the live model untouched."""
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, enable_hot_swap=True)
    store, digest = _persist_active_slot(cfg)
    # Tamper with the on-disk slot so its SHA-256 no longer matches.
    (store.slot_dir / f"{digest}.pt").write_bytes(b"corrupted-not-a-state-dict")

    orch = build_orchestrator(cfg)
    source = orch._weight_update_pollers["world_model"]  # type: ignore[attr-defined]
    metrics = orch._metrics  # type: ignore[attr-defined]
    assert metrics is not None
    prior_model = orch._world_model  # type: ignore[attr-defined]

    surfaced = await source.refresh_once()

    # Fail-closed: nothing surfaced, integrity_mismatch counted off-loop.
    assert surfaced is False
    assert source.pending_update is None
    rendered = metrics.render_prometheus()
    assert "on_device_learning_reverted" in rendered
    assert 'reason="integrity_mismatch"' in rendered

    # A tick swaps nothing -> the live model identity is UNCHANGED.
    orch._apply_pending_weight_update()  # type: ignore[attr-defined]
    assert orch._world_model is prior_model  # type: ignore[attr-defined]


def test_materialiser_device_parity(tmp_path: Path) -> None:
    """The materialised engine lands on the SAME device as the live world model."""
    experience_path = str(tmp_path / "experience_root")
    cfg = _build_cfg(experience_path, enable_hot_swap=True)
    _store, digest = _persist_active_slot(cfg)

    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)
    live_device = next(wm.parameters()).device

    source = build_on_device_hot_swap_source(cfg, world_model=wm)
    assert source is not None
    engine = source._materialize(digest)  # type: ignore[attr-defined]
    assert next(engine.parameters()).device == live_device
    # The materialised engine round-trips the live arch (strict load succeeded).
    assert isinstance(engine, RSSM)
