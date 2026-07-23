"""Unit tests for the growth SHA-256 distilled-student slot store."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, GrowthConfig
from mousedroid.growth.slot_store import GrowthSlotIntegrityError, GrowthSlotStore, StudentSlot


def _store(tmp_path: Path) -> GrowthSlotStore:
    exp = ExperienceConfig.model_validate({"path": str(tmp_path / "exp"), "map_size_gb": 0.01})
    growth = GrowthConfig(enabled=True, slot_dir="growth_slot")
    return GrowthSlotStore(experience_cfg=exp, growth_cfg=growth)


def test_slot_dir_resolved_under_experience_root(tmp_path: Path) -> None:
    """The slot dir is ``<experience.path>/<slot_dir>`` — never an absolute leak."""
    store = _store(tmp_path)
    assert store.slot_dir == (tmp_path / "exp" / "growth_slot").resolve()


def test_persist_then_load_round_trip(tmp_path: Path) -> None:
    """A persisted student round-trips and its filename is the 64-hex digest."""
    store = _store(tmp_path)
    sd = {"net.0.weight": torch.zeros(3, 8), "net.0.bias": torch.ones(3)}
    slot = store.persist(sd)
    assert slot.path.exists()
    assert slot.path.name == f"{slot.digest}.pt"
    assert len(slot.digest) == 64
    loaded = store.load(slot)
    assert set(loaded) == set(sd)
    assert torch.equal(loaded["net.0.bias"], torch.ones(3))


def test_load_rejects_tampered_slot(tmp_path: Path) -> None:
    """A digest mismatch raises ``GrowthSlotIntegrityError`` (fail-closed)."""
    store = _store(tmp_path)
    slot = store.persist({"w": torch.zeros(2, 2)})
    bogus = StudentSlot(path=slot.path, digest="0" * 64)
    with pytest.raises(GrowthSlotIntegrityError):
        store.load(bogus)
