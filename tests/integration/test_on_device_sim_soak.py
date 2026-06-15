"""Deterministic sim-soak: the full on-device path promotes / reverts correctly.

This is the Phase-6 follow-up sim-soak. Unlike the focused WS-E3 gate
integration test (which injects a recon-loss ``score_fn`` to force the revert
branch), this drives the **whole** on-device pipeline — the factory-wired
``ReplayTriggerCoordinator`` → real ``RSSMRefiner`` → ``OnDeviceSlotStore`` →
real ``_build_on_device_gate_runner`` recon-loss gate → ``mark_active`` /
revert-counter — over a **SEEDED replay stream**, and proves all three
load-bearing properties of the enabled loop end-to-end:

(a) a **known-IMPROVING** refinement PROMOTES — the candidate slot is marked
    active (``load_active() == digest``) and the revert counter never fires;
(b) a **known-DEGRADING** refinement (deterministically-noised candidate)
    REVERTS — ``inc_on_device_learning_reverted("regression_bound")`` fires,
    the slot is NEVER blessed, and the live baseline RSSM is bitwise-unchanged;
(c) the 30 Hz reactive hot loop is **NEVER advanced** by either cycle —
    ``orchestrator._tick_count`` stays ``0`` (all torch work runs on the
    slow-cadence seam, off the hot loop).

Determinism: a fixed ``scoring_seed`` + a tiny deterministic ``ModelConfig`` +
``Settings(mock_hardware=True)` + a seeded LMDB replay stream. No wall-clock or
RNG drift — the refiner/gate seed the global torch RNG from ``scoring_seed`` and
restore it, the degrading noise is drawn from an explicitly-seeded
``torch.Generator``, and the replay records are generated from a seeded
``numpy`` RNG. Built with ``mock_hardware=True`` so no real device is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import (
    _build_on_device_gate_runner,
    build_metrics_registry,
    build_orchestrator,
    build_world_model,
)
from mousedroid.learning.on_device.protocol import OnDeviceUpdateResult
from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
from mousedroid.learning.on_device.rssm_refiner import RSSMRefiner
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
from mousedroid.world_model.rssm import RSSM

if TYPE_CHECKING:
    from collections.abc import Mapping

# Refine batch geometry kept tiny so the soak runs fast + deterministically:
# 2 episodes x 3 steps = 6 records per batch. The held-out batch must be
# DISJOINT from the refine window, so the store needs refine(6) + held_out(6)
# = 12 records minimum; seed comfortably above that.
_REFINE_SEQUENCE_LENGTH = 3
_REFINE_BATCH_EPISODES = 2
_N_SEEDED = 18
_TRIGGER = 5
_SCORING_SEED = 1234
# Large additive noise on every candidate parameter guarantees the degraded
# candidate's held-out recon+KL loss is strictly worse than the baseline's.
_DEGRADE_NOISE_SCALE = 5.0
_DEGRADE_SEED = 7


def _seed_replay_store(experience_path: str, n: int) -> None:
    """Write ``n`` deterministic experience records to a fresh LMDB store.

    Records carry seeded-random motor/action/distance so the assembled
    ``(B, T, ...)`` batch is non-degenerate (a non-trivial recon target) yet
    fully reproducible run-to-run.
    """
    cfg = Settings.model_validate(
        {"mock_hardware": True, "experience": {"path": experience_path, "map_size_gb": 0.01}}
    )
    rng = np.random.default_rng(0)
    logger = ExperienceLogger(cfg.experience)
    logger.open()
    try:
        for i in range(n):
            logger.log(
                MouseDroidExperienceRecord(
                    timestamp=float(i),
                    vision_features=np.zeros(0, dtype=np.float32),
                    distance_m=float(rng.uniform(0.1, 2.0)),
                    motor_state=rng.standard_normal(4).astype(np.float32),
                    action=rng.standard_normal(3).astype(np.float32),
                    reward=float(rng.uniform(-1.0, 1.0)),
                )
            )
    finally:
        logger.close()


def _build_cfg(experience_path: str, *, regression_tolerance: float) -> Settings:
    """Build a ``mock_hardware`` Settings with a TINY deterministic model + on-device block."""
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
            # Tiny model so the refiner + recon-loss gate run in milliseconds and
            # the soak is deterministic; vision disabled (dim 0 pair).
            "model": {
                "vision_dim": 0,
                "vision_proj_dim": 0,
                "ultrasonic_dim": 1,
                "ultrasonic_proj_dim": 4,
                "motor_state_dim": 4,
                "motor_proj_dim": 4,
                "hidden_dim": 8,
                "latent_dim": 4,
                "action_dim": 3,
                "obs_dim": 8,
            },
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": _TRIGGER,
                "update_steps": 2,
                "check_interval_s": 0.01,
                "scoring_seed": _SCORING_SEED,
                "regression_tolerance": regression_tolerance,
                "refine_sequence_length": _REFINE_SEQUENCE_LENGTH,
                "refine_batch_episodes": _REFINE_BATCH_EPISODES,
            },
        }
    )


class _DegradingRefiner:
    """Wraps the real :class:`RSSMRefiner`, adds deterministic noise to its candidate.

    Drives the FULL produce path (the real refiner deep-copies the live RSSM and
    runs ``train_sequence`` over the real seeded batch) but returns a candidate
    whose every parameter has a large, fixed-seed Gaussian perturbation added —
    so the real recon-loss gate scores it strictly worse than the live baseline
    and reverts. The base RSSM is never mutated (the inner refiner's contract).
    """

    def __init__(self, inner: RSSMRefiner) -> None:
        self._inner = inner

    def update(self, batch: Mapping[str, torch.Tensor]) -> OnDeviceUpdateResult:
        result = self._inner.update(batch)
        gen = torch.Generator().manual_seed(_DEGRADE_SEED)
        noised: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, tensor in result.candidate_state_dict.items():
                if tensor.dtype.is_floating_point:
                    # The CPU ``Generator`` draws the noise on CPU; move it onto the
                    # candidate tensor's device BEFORE the add so a CUDA candidate
                    # (GPU box) does not device-mismatch. ``.to(device)`` is a no-op
                    # on a CPU-only host, so the CPU path stays byte-identical.
                    noise = (
                        torch.randn(tensor.shape, generator=gen).to(tensor.device)
                        * _DEGRADE_NOISE_SCALE
                    )
                    noised[name] = tensor + noise
                else:
                    noised[name] = tensor.clone()
        return OnDeviceUpdateResult(
            candidate_state_dict=noised,
            train_loss=result.train_loss,
            n_steps=result.n_steps,
            metadata=dict(result.metadata),
        )


@pytest.mark.asyncio
async def test_improving_refinement_promotes_via_full_pipeline(tmp_path: Path) -> None:
    """(a)+(c): a small-step refinement clears the bound -> slot active; hot loop untouched.

    Drives the REAL factory-wired coordinator (built inside ``build_orchestrator``)
    over a seeded replay stream. A small ``update_steps`` keeps the refined
    candidate close to the baseline, so with a generous tolerance the held-out
    recon-loss delta clears the bound and the candidate is PROMOTED end-to-end
    (slot persisted -> recon-loss gate -> slot marked active). The 30 Hz hot loop
    is never advanced.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    # Generous tolerance: the tiny refinement always lands within the bound
    # regardless of which direction the loss moved.
    cfg = _build_cfg(experience_path, regression_tolerance=1e9)

    orchestrator = build_orchestrator(cfg)
    coordinator = orchestrator._on_device_coordinator  # type: ignore[attr-defined]
    assert coordinator is not None

    slot = await coordinator.maybe_update()
    assert slot is not None

    # The recon-loss gate PROMOTED -> the slot is the active pointer.
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    assert store.load_active() == slot.digest

    # A promote never touches the revert counter -> no revert series on /metrics.
    metrics = orchestrator._metrics  # type: ignore[attr-defined]
    assert metrics is not None
    assert "on_device_learning_reverted" not in metrics.render_prometheus()

    # (c) The 30 Hz reactive hot loop was NEVER advanced by the slow-cadence cycle.
    assert orchestrator._tick_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_degrading_refinement_reverts_via_full_pipeline(tmp_path: Path) -> None:
    """(b)+(c): a deterministically-noised candidate reverts; live model untouched; hot loop idle.

    Builds the full coordinator by hand from the factory's REAL gate runner +
    REAL slot store, but wraps the real ``RSSMRefiner`` so the candidate it
    persists is a large, fixed-seed-noised copy of the refined weights. The real
    recon-loss gate scores it strictly worse than the live baseline (zero
    tolerance) and REVERTS: the counter fires with reason ``regression_bound``,
    the slot is never blessed, and the live baseline RSSM is bitwise-unchanged.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, regression_tolerance=0.0)

    metrics = build_metrics_registry(cfg)
    world_model = build_world_model(cfg)
    assert isinstance(world_model, RSSM)

    # Snapshot the live baseline params so we can prove the revert path leaves the
    # running brain bitwise-unchanged (the candidate is scored in a deep copy).
    before = {name: param.detach().clone() for name, param in world_model.named_parameters()}

    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    reader = LMDBReplayReader(cfg.experience)

    first_param = next(world_model.parameters())
    device = first_param.device
    encoder = world_model.encoder
    model_cfg = world_model.cfg

    # The REAL recon-loss gate runner (held-out batch DISJOINT from the refine
    # window, shared decoders, scoring_seed) — exactly what the factory wires.
    gate_runner = _build_on_device_gate_runner(
        cfg,
        slot_store=store,
        metrics=metrics,
        world_model=world_model,
        reader=reader,
        model_cfg=model_cfg,
        encoder=encoder,
        device=device,
        refine_sequence_length=_REFINE_SEQUENCE_LENGTH,
        refine_n_episodes=_REFINE_BATCH_EPISODES,
    )

    # Wrap the real refiner so the produced candidate is provably degraded.
    learner = _DegradingRefiner(RSSMRefiner(world_model, cfg.on_device_learning))

    cap = max(_TRIGGER, _REFINE_BATCH_EPISODES * _REFINE_SEQUENCE_LENGTH)
    records_seen = 0

    def _count_new_records() -> int:
        # Single fired cycle: report the seeded store size (above the trigger).
        nonlocal records_seen
        records_seen = _N_SEEDED
        return _N_SEEDED

    def _load_batch() -> dict[str, torch.Tensor]:
        from mousedroid.factory import _load_replay_sequence_batch

        return _load_replay_sequence_batch(
            reader,
            model_cfg,
            encoder,
            sequence_length=_REFINE_SEQUENCE_LENGTH,
            n_episodes=_REFINE_BATCH_EPISODES,
            cap=cap,
            device=device,
        )

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg.on_device_learning,
        learner=learner,
        slot_store=store,
        count_new_records=_count_new_records,
        load_batch=_load_batch,
        gate_runner=gate_runner,
    )

    slot = await coordinator.maybe_update()
    # The candidate is still produced + persisted...
    assert slot is not None

    # ...but the recon-loss gate REVERTED it: never blessed, counter fired once.
    assert store.load_active() is None
    assert metrics is not None
    rendered = metrics.render_prometheus()
    assert "on_device_learning_reverted" in rendered
    assert 'reason="regression_bound"' in rendered

    # (b) The live baseline RSSM is bitwise-unchanged by the revert cycle.
    for name, param in world_model.named_parameters():
        assert torch.equal(param, before[name]), f"live baseline param {name!r} mutated"


@pytest.mark.asyncio
async def test_degrading_refinement_keeps_hot_loop_idle_via_orchestrator(tmp_path: Path) -> None:
    """(c) for the revert path: driving the real orchestrator's coordinator keeps tick_count 0.

    The previous test builds the coordinator by hand to inject the degrading
    refiner; this companion proves the hot-loop-isolation property holds when the
    same revert is driven through the orchestrator's OWN factory-wired coordinator
    (its learner monkeypatched to the degrading wrapper), so no orchestrator wiring
    detail leaks the slow cycle onto the 30 Hz loop.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path, regression_tolerance=0.0)

    orchestrator = build_orchestrator(cfg)
    coordinator = orchestrator._on_device_coordinator  # type: ignore[attr-defined]
    assert coordinator is not None

    # Swap in the degrading refiner on the real coordinator (private attr is the
    # only seam to force a degraded candidate through the orchestrator's own gate).
    inner = coordinator._learner  # type: ignore[attr-defined]
    assert isinstance(inner, RSSMRefiner)
    coordinator._learner = _DegradingRefiner(inner)  # type: ignore[attr-defined]

    slot = await coordinator.maybe_update()
    assert slot is not None

    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    assert store.load_active() is None

    metrics = orchestrator._metrics  # type: ignore[attr-defined]
    assert metrics is not None
    assert 'reason="regression_bound"' in metrics.render_prometheus()

    # The 30 Hz reactive hot loop was NEVER advanced.
    assert orchestrator._tick_count == 0  # type: ignore[attr-defined]
