"""Unit tests for the off-loop ``GrowthDistillationCoordinator``.

Exercises the ``maybe_distill`` cycle with injected fakes (no world model /
hardware): below-threshold no-op, no-batch skip, and the full produce-and-persist
path with metric + consumed-offset accounting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import GrowthConfig
from mousedroid.growth.coordinator import GrowthDistillationCoordinator
from mousedroid.growth.slot_store import StudentSlot


class _FakeDistiller:
    def __init__(self) -> None:
        self.calls = 0

    def distill_step(
        self, x: torch.Tensor, hard_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls += 1
        return torch.tensor(0.25)


class _FakeStudent:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"w": torch.zeros(2, 2)}


class _FakeSlotStore:
    def __init__(self) -> None:
        self.persisted: object | None = None

    def persist(self, sd: object) -> StudentSlot:
        self.persisted = sd
        return StudentSlot(path=Path("/tmp/growth/" + "a" * 64 + ".pt"), digest="a" * 64)


class _MetricsSpy:
    def __init__(self) -> None:
        self.outcomes: list[str] = []

    def inc_growth_distilled(self, outcome: str, amount: int = 1) -> None:
        self.outcomes.append(outcome)


def _coordinator(
    *,
    new_records: int,
    batch: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: GrowthConfig | None = None,
) -> tuple[GrowthDistillationCoordinator, _FakeDistiller, _FakeSlotStore, _MetricsSpy, list[int]]:
    distiller = _FakeDistiller()
    slot_store = _FakeSlotStore()
    metrics = _MetricsSpy()
    consumed: list[int] = []
    coord = GrowthDistillationCoordinator(
        cfg=cfg or GrowthConfig(enabled=True, trigger_min_new_records=5, distill_steps=3),
        distiller=distiller,  # type: ignore[arg-type]
        student=_FakeStudent(),  # type: ignore[arg-type]
        sample_batch=lambda: batch,
        slot_store=slot_store,  # type: ignore[arg-type]
        count_new_records=lambda: new_records,
        on_consumed=consumed.append,
        metrics=metrics,  # type: ignore[arg-type]
    )
    return coord, distiller, slot_store, metrics, consumed


@pytest.mark.asyncio
async def test_below_threshold_is_noop() -> None:
    """Fewer new records than the trigger → no distill, no persist, no metric."""
    coord, distiller, slot_store, metrics, consumed = _coordinator(new_records=2, batch=None)
    result = await coord.maybe_distill()
    assert result is None
    assert distiller.calls == 0
    assert slot_store.persisted is None
    assert metrics.outcomes == []
    assert consumed == []


@pytest.mark.asyncio
async def test_no_batch_skips_and_records_metric() -> None:
    """Trigger armed but sampler yields None → skip, ``skipped_no_batch`` metric."""
    coord, distiller, slot_store, metrics, consumed = _coordinator(new_records=10, batch=None)
    result = await coord.maybe_distill()
    assert result is None
    assert distiller.calls == 0
    assert slot_store.persisted is None
    assert metrics.outcomes == ["skipped_no_batch"]
    assert consumed == []


@pytest.mark.asyncio
async def test_full_cycle_persists_and_accounts() -> None:
    """Armed + batch → runs ``distill_steps``, persists, advances offset, metric."""
    batch = (torch.randn(4, 5), torch.randn(4, 3))
    coord, distiller, slot_store, metrics, consumed = _coordinator(new_records=10, batch=batch)
    slot = await coord.maybe_distill()
    assert slot is not None
    assert distiller.calls == 3  # distill_steps
    assert slot_store.persisted is not None
    assert metrics.outcomes == ["completed"]
    assert consumed == [10]  # on_consumed advanced by the new-record count


@pytest.mark.asyncio
async def test_student_state_dict_is_detached_clone() -> None:
    """The persisted state-dict is a detached clone, not the live tensors."""
    batch = (torch.randn(2, 5), torch.randn(2, 3))
    coord, _distiller, slot_store, _metrics, _consumed = _coordinator(new_records=10, batch=batch)
    await coord.maybe_distill()
    persisted = slot_store.persisted
    assert isinstance(persisted, dict)
    assert all(not t.requires_grad for t in persisted.values())
