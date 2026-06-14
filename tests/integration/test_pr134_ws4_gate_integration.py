"""Integration: WS4 safety-regression gate wired through ``build_orchestrator``.

Proves the Phase-6 WS4 produce -> score -> promote/revert path end-to-end via
the factory:

* with ``on_device_learning.enabled=True`` + a seeded replay store, the
  orchestrator builds a coordinator whose ``gate_runner`` is wired;
* driving the coordinator produces + persists a candidate slot AND runs the
  safety gate, which (with the WS4 config-sized stand-in policy + a generous
  ``regression_tolerance``) PROMOTES: the slot is marked ACTIVE in the slot
  store. The candidate + baseline adapters wrap SEPARATE stand-in nets (so a
  candidate weight load never aliases the baseline);
* a deterministically-degraded candidate (injected via the gate seam) REVERTS:
  the slot is never marked active and the shared metrics registry's revert
  counter increments with reason ``regression_bound``;
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
from mousedroid.learning.on_device.scoring import PolicyProtocol
from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore

# Refine batch geometry: 2 episodes x 3 steps = 6 records per batch.
_REFINE_SEQUENCE_LENGTH = 3
_REFINE_BATCH_EPISODES = 2
# Seed comfortably above both the trigger AND one full refine batch.
_N_SEEDED = 8
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
                "rollout_horizon": 4,
                "n_scoring_rollouts": 2,
                "scoring_seed": 99,
                "refine_sequence_length": _REFINE_SEQUENCE_LENGTH,
                "refine_batch_episodes": _REFINE_BATCH_EPISODES,
            },
        }
    )


@pytest.mark.asyncio
async def test_gate_promotes_stand_in_candidate_end_to_end(tmp_path: Path) -> None:
    """The wired coordinator produces a slot AND the gate marks it active.

    The candidate + baseline adapters now wrap SEPARATE stand-in nets (so a
    candidate weight load can never alias into the baseline), so their scores
    are no longer guaranteed equal. A generous ``regression_tolerance`` makes
    the PROMOTE deterministic regardless of which random stand-in net scores
    higher — the assertion under test is the end-to-end promote path
    (slot persisted -> gate run -> slot marked active), not the gate's bound.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)
    # Generous tolerance so the separate stand-in nets always clear the bound.
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


def test_gate_runner_uses_separate_module_instances(tmp_path: Path) -> None:
    """Candidate + baseline adapters wrap DISTINCT ``nn.Module`` instances.

    If both adapters aliased the same net, loading candidate weights would
    mutate the baseline in place and collapse the regression delta to zero.
    The two stand-in adapters must be independent objects backed by separate
    modules so a candidate weight load never bleeds into the baseline.
    """
    from mousedroid.factory import _build_on_device_gate_runner

    experience_path = str(tmp_path / "experience_root")
    cfg = _build_cfg(experience_path)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)

    gate_runner = _build_on_device_gate_runner(cfg, slot_store=store, metrics=None)

    # The closure captures the candidate + baseline adapters; pull them out.
    adapters = [
        cell.cell_contents
        for cell in (gate_runner.__closure__ or ())
        if hasattr(cell.cell_contents, "_module")
    ]
    assert len(adapters) == 2, "expected exactly the candidate + baseline adapters"
    candidate_adapter, baseline_adapter = adapters

    assert candidate_adapter is not baseline_adapter
    candidate_net = candidate_adapter._module
    baseline_net = baseline_adapter._module
    assert candidate_net is not baseline_net

    # Loading candidate weights must NOT mutate the baseline net.
    baseline_before = baseline_net.weight.detach().clone()
    with torch.no_grad():
        candidate_net.weight.add_(1.0)
    assert torch.equal(baseline_net.weight, baseline_before)


@pytest.mark.asyncio
async def test_degraded_candidate_reverts_and_increments_counter(tmp_path: Path) -> None:
    """A degraded candidate reverts E2E: slot un-blessed + revert counter fires.

    Exercises the REVERT branch through the real gate + real shared metrics
    registry by injecting a candidate that scores strictly below the baseline
    via the gate's ``score_fn`` seam (the WS5 hook), with a zero tolerance.
    """
    experience_path = str(tmp_path / "experience_root")
    _seed_replay_store(experience_path, _N_SEEDED)
    cfg = _build_cfg(experience_path)
    # Zero tolerance so any drop reverts.
    cfg = cfg.model_copy(
        update={
            "on_device_learning": cfg.on_device_learning.model_copy(
                update={"regression_tolerance": 0.0}
            )
        }
    )

    metrics = build_metrics_registry(cfg)
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)

    scores = {"baseline": 1.0, "candidate": 0.0}

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

        def act(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros(hidden.shape[0], 1)

    def _score_fn(policy: PolicyProtocol) -> float:
        return scores[policy.name]  # type: ignore[attr-defined]

    gate = RegressionGate(
        cfg=cfg.on_device_learning, slot_store=store, metrics=metrics, score_fn=_score_fn
    )
    slot: CandidateSlot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(candidate=_Named("candidate"), baseline=_Named("baseline"), slot=slot)

    assert decision.promoted is False
    assert store.load_active() is None
    assert metrics is not None
    rendered = metrics.render_prometheus()
    assert "on_device_learning_reverted" in rendered
    assert 'reason="regression_bound"' in rendered
