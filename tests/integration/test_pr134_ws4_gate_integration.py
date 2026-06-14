"""Integration: WS-E3 recon-loss regression gate wired through the factory.

Proves the Phase-6 WS-E3 produce -> refine -> score (RSSM-vs-RSSM recon loss) ->
promote/revert path end-to-end via the factory:

* with ``on_device_learning.enabled=True`` + a seeded replay store, the
  orchestrator builds a coordinator whose ``gate_runner`` is wired to the
  recon-loss gate (the WS-E2 RSSMRefiner produces the candidate; the gate scores
  the refined-slot candidate RSSM against the live baseline RSSM on a FIXED
  held-out batch DISJOINT from the refine batch, with shared decoders);
* driving the coordinator produces + persists a candidate slot AND runs the
  gate. With a generous ``regression_tolerance`` the (small-step) refined
  candidate clears the bound and is PROMOTED: the slot is marked ACTIVE;
* the live baseline world model is bitwise-unchanged by a gate evaluation (the
  candidate is loaded into a deep copy);
* a degraded candidate (injected via the gate's recon-loss ``score_fn`` seam)
  REVERTS: the slot is never marked active and the shared metrics registry's
  revert counter increments with reason ``regression_bound``;
* the 30 Hz hot loop is NEVER advanced — ``_tick_count`` stays 0.

Built with ``mock_hardware=True`` so no real device is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_metrics_registry, build_on_device_coordinator
from mousedroid.learning.on_device.regression_gate import RegressionGate
from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
from mousedroid.world_model.protocol import WorldModelProtocol

# Refine batch geometry: 2 episodes x 3 steps = 6 records per batch.
_REFINE_SEQUENCE_LENGTH = 3
_REFINE_BATCH_EPISODES = 2
# The held-out batch must be DISJOINT from the refine window, so the store needs
# refine(6) + held_out(6) = 12 records minimum; seed comfortably above that.
_N_SEEDED = 16
_TRIGGER = 5


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


def _build_cfg(experience_path: str) -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": _TRIGGER,
                "update_steps": 2,
                "check_interval_s": 0.01,
                "scoring_seed": 99,
                "refine_sequence_length": _REFINE_SEQUENCE_LENGTH,
                "refine_batch_episodes": _REFINE_BATCH_EPISODES,
            },
        }
    )


@pytest.mark.asyncio
async def test_gate_promotes_refined_candidate_end_to_end(tmp_path: Path) -> None:
    """The wired coordinator refines + the recon-loss gate marks the slot active.

    A small ``update_steps`` keeps the refined candidate close to the baseline, so
    with a generous tolerance the recon-loss delta clears the bound and the
    candidate is PROMOTED end-to-end (slot persisted -> gate run -> slot active).
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)
    # Generous tolerance so the (small-step) refined candidate always clears the
    # recon-loss bound regardless of which direction the tiny refinement moved.
    cfg = cfg.model_copy(
        update={
            "on_device_learning": cfg.on_device_learning.model_copy(
                update={"regression_tolerance": 1e9}
            )
        }
    )

    metrics = build_metrics_registry(cfg)
    coordinator = build_on_device_coordinator(cfg, metrics=metrics)
    assert coordinator is not None

    slot = await coordinator.maybe_update()  # type: ignore[attr-defined]
    assert slot is not None

    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    assert store.load_active() == slot.digest

    # A promote never touches the revert counter -> /metrics has no revert series.
    assert metrics is not None
    assert "on_device_learning_reverted" not in metrics.render_prometheus()


@pytest.mark.asyncio
async def test_hot_loop_untouched_by_gate(tmp_path: Path) -> None:
    """Running the gate via the orchestrator never advances the 30 Hz hot loop."""
    from mousedroid.factory import build_orchestrator

    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    orchestrator = build_orchestrator(cfg)
    coordinator = orchestrator._on_device_coordinator  # type: ignore[attr-defined]
    assert coordinator is not None

    await coordinator.maybe_update()

    assert orchestrator._tick_count == 0  # type: ignore[attr-defined]


def test_gate_runner_loads_slot_into_copy_baseline_untouched(tmp_path: Path) -> None:
    """The gate runner scores the slot's weights in a COPY; the live RSSM is untouched.

    The recon-loss gate materialises the candidate by loading the persisted slot's
    refined weights into a DEEP COPY of the live RSSM and scores it against the
    live baseline. Running the gate must NOT mutate the live world model's params
    (a revert leaves the running brain bitwise-unchanged).
    """
    from mousedroid.factory import _build_on_device_gate_runner, build_world_model
    from mousedroid.world_model.rssm import RSSM

    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)

    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    reader = LMDBReplayReader(cfg.experience)
    first_param = next(wm.parameters())
    gate_runner = _build_on_device_gate_runner(
        cfg,
        slot_store=store,
        metrics=None,
        world_model=wm,
        reader=reader,
        model_cfg=wm.cfg,
        encoder=wm.encoder,
        device=first_param.device,
        refine_sequence_length=_REFINE_SEQUENCE_LENGTH,
        refine_n_episodes=_REFINE_BATCH_EPISODES,
    )

    # Persist a slot whose weights ARE the live model's (a no-change candidate) so
    # the load + score path runs against a real, dim-matching state dict.
    slot = store.persist(wm.state_dict())
    before = {n: p.detach().clone() for n, p in wm.named_parameters()}

    gate_runner(slot)

    for name, param in wm.named_parameters():
        assert torch.equal(param, before[name]), f"live baseline param {name!r} mutated by gate"


@pytest.mark.asyncio
async def test_degraded_candidate_reverts_and_increments_counter(tmp_path: Path) -> None:
    """A degraded candidate reverts E2E: slot un-blessed + revert counter fires.

    Exercises the REVERT branch through the real recon-loss gate + real shared
    metrics registry by injecting a recon-loss ``score_fn`` where the candidate's
    loss strictly exceeds the baseline's, with a zero tolerance.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)
    cfg = cfg.model_copy(
        update={
            "on_device_learning": cfg.on_device_learning.model_copy(
                update={"regression_tolerance": 0.0}
            )
        }
    )

    metrics = build_metrics_registry(cfg)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)

    baseline_wm = object()
    candidate_wm = object()
    losses = {id(baseline_wm): 1.0, id(candidate_wm): 2.0}  # candidate loss WORSE

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    gate = RegressionGate(
        cfg=cfg.on_device_learning, slot_store=store, metrics=metrics, score_fn=_score_fn
    )
    slot: CandidateSlot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
    )

    assert decision.promoted is False
    assert store.load_active() is None
    assert metrics is not None
    rendered = metrics.render_prometheus()
    assert "on_device_learning_reverted" in rendered
    assert 'reason="regression_bound"' in rendered


def test_gate_runner_corrupt_slot_counts_integrity_mismatch(tmp_path: Path) -> None:
    """A corrupt slot fails its SHA-256 check on load -> integrity_mismatch, no promote."""
    from mousedroid.factory import _build_on_device_gate_runner, build_world_model
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
    from mousedroid.world_model.rssm import RSSM

    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)

    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)

    class _SpyCounter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
            self.calls.append(reason)

    counter = _SpyCounter()
    reader = LMDBReplayReader(cfg.experience)
    first_param = next(wm.parameters())
    gate_runner = _build_on_device_gate_runner(
        cfg,
        slot_store=store,
        metrics=counter,
        world_model=wm,
        reader=reader,
        model_cfg=wm.cfg,
        encoder=wm.encoder,
        device=first_param.device,
        refine_sequence_length=_REFINE_SEQUENCE_LENGTH,
        refine_n_episodes=_REFINE_BATCH_EPISODES,
    )

    slot = store.persist(wm.state_dict())
    # Tamper with the on-disk blob so its digest no longer matches slot.digest.
    slot.path.write_bytes(b"corrupted-not-a-state-dict")

    gate_runner(slot)

    assert counter.calls == ["integrity_mismatch"]
    assert store.load_active() is None


@pytest.mark.asyncio
async def test_gate_skips_when_no_disjoint_held_out_window(tmp_path: Path) -> None:
    """Too few records for a disjoint held-out window -> gate is a no-op (no promote).

    With only enough records to fill the REFINE batch (but not a disjoint held-out
    window), the gate cannot score and must NOT promote — a fresh/sparse Jetson
    never crashes; promotion simply waits for more experience.
    """
    experience_path = str(tmp_path / "experience_root")
    # Exactly the refine batch size (6): no room for a disjoint held-out window.
    _seed_replay_store(experience_path, _REFINE_SEQUENCE_LENGTH * _REFINE_BATCH_EPISODES)
    cfg = _build_cfg(experience_path)
    cfg = cfg.model_copy(
        update={
            "on_device_learning": cfg.on_device_learning.model_copy(
                update={"regression_tolerance": 1e9}
            )
        }
    )

    metrics = build_metrics_registry(cfg)
    coordinator = build_on_device_coordinator(cfg, metrics=metrics)
    assert coordinator is not None

    slot = await coordinator.maybe_update()  # type: ignore[attr-defined]
    assert slot is not None  # the candidate is still produced + persisted

    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    # ...but the gate could not score it (no held-out window) -> never marked active.
    assert store.load_active() is None
    assert metrics is not None
    assert "on_device_learning_reverted" not in metrics.render_prometheus()


def test_held_out_batch_none_when_no_reader(tmp_path: Path) -> None:
    """``_build_held_out_sequence_batch`` returns None without a reader (no-op gate)."""
    from mousedroid.factory import _build_held_out_sequence_batch, build_world_model
    from mousedroid.world_model.rssm import RSSM

    cfg = _build_cfg(str(tmp_path / "experience_root"))
    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)

    result = _build_held_out_sequence_batch(
        None,
        wm.cfg,
        wm.encoder,
        sequence_length=_REFINE_SEQUENCE_LENGTH,
        n_episodes=_REFINE_BATCH_EPISODES,
        refine_offset=_REFINE_SEQUENCE_LENGTH * _REFINE_BATCH_EPISODES,
        device=next(wm.parameters()).device,
    )

    assert result is None
