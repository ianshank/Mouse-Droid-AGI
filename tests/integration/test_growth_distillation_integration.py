"""Integration test: the real factory-built growth coordinator distils end-to-end.

Unlike ``tests/unit/growth/test_coordinator.py`` (which injects fakes), this wires
the REAL collaborators through the factory — the compact ``StudentVLAPolicy``, the
``VLATeacherModule`` around a live ``MockVLA``, the regression ``KnowledgeDistiller``,
the SHA-256 ``GrowthSlotStore``, and the world-model rollout sampler
(``imagine_step`` under the teacher) — and runs one full ``maybe_distill`` cycle.
Replay is empty, so the new-record trigger is forced ON to exercise the distill +
persist + metric path (the trigger logic itself is unit-tested separately).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_growth_coordinator, build_metrics_registry, build_vla_policy
from mousedroid.growth.coordinator import GrowthDistillationCoordinator


def _enabled_cfg(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
            "vla": {"backend": "mock"},
            "growth": {
                "enabled": True,
                "trigger_min_new_records": 1,
                "batch_size": 4,
                "distill_steps": 3,
                "student_hidden_dim": 16,
            },
        }
    )


@pytest.mark.asyncio
async def test_real_coordinator_distills_and_persists(tmp_path: Path) -> None:
    """A full cycle rolls the live world model, distils, and persists a slot."""
    cfg = _enabled_cfg(tmp_path)
    metrics = build_metrics_registry(cfg)
    coord = build_growth_coordinator(cfg, vla_policy=build_vla_policy(cfg), metrics=metrics)
    assert isinstance(coord, GrowthDistillationCoordinator)

    # Replay is empty; force the trigger ON to exercise the distill+persist path.
    coord._count_new_records = lambda: 5  # type: ignore[attr-defined,assignment]

    slot = await coord.maybe_distill()
    assert slot is not None
    assert slot.path.exists()
    assert len(slot.digest) == 64
    # The distillation counter surfaced on /metrics with the completed outcome.
    rendered = metrics.render_prometheus()
    assert "growth_distillations_total" in rendered
    assert 'outcome="completed"' in rendered


@pytest.mark.asyncio
async def test_real_coordinator_persisted_slot_reloads(tmp_path: Path) -> None:
    """The persisted student slot round-trips through the slot store's integrity check."""
    cfg = _enabled_cfg(tmp_path)
    coord = build_growth_coordinator(cfg, vla_policy=build_vla_policy(cfg))
    assert isinstance(coord, GrowthDistillationCoordinator)
    coord._count_new_records = lambda: 5  # type: ignore[attr-defined,assignment]

    slot = await coord.maybe_distill()
    assert slot is not None
    # The store re-verifies the SHA-256 digest on load (fail-closed on tamper).
    loaded = coord._slot_store.load(slot)  # type: ignore[attr-defined]
    assert loaded  # non-empty student state-dict
