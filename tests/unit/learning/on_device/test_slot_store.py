"""Unit tests for the on-device candidate slot store (WS3).

Pins the WS3 persistence contract:

* a candidate state-dict round-trips through ``persist`` -> ``load`` with the
  SHA-256 digest reused from the C1 OTA helper (``verify_sha256``);
* the persisted file lands UNDER the configured experience root +
  ``slot_dir`` leaf — never CWD, never a hardcoded absolute path (the WS0
  de-hardcode realized);
* a corrupted slot (digest mismatch) raises so WS4 can map it to the
  ``integrity_mismatch`` revert reason;
* parent directories are created defensively.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, OnDeviceLearningConfig
from mousedroid.learning.on_device.slot_store import (
    CandidateSlot,
    OnDeviceSlotStore,
    SlotIntegrityError,
)


def _make_state_dict() -> dict[str, torch.Tensor]:
    """Build a tiny deterministic candidate state-dict."""
    torch.manual_seed(0)
    return {
        "layer.weight": torch.randn(3, 4),
        "layer.bias": torch.randn(3),
    }


def _make_store(tmp_path: Path) -> OnDeviceSlotStore:
    """Build a slot store rooted at ``tmp_path`` via config (no hardcoded path)."""
    experience = ExperienceConfig(path=str(tmp_path / "experience_root"))
    on_device = OnDeviceLearningConfig(enabled=True, slot_dir="on_device_slot")
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=on_device)


def test_persist_then_load_round_trips(tmp_path: Path) -> None:
    """A persisted candidate loads back with identical tensors."""
    store = _make_store(tmp_path)
    state = _make_state_dict()

    slot = store.persist(state)
    loaded = store.load(slot)

    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key])


def test_persist_returns_sha256_digest(tmp_path: Path) -> None:
    """The returned slot carries a 64-char lowercase hex SHA-256 digest."""
    store = _make_store(tmp_path)

    slot = store.persist(_make_state_dict())

    assert isinstance(slot, CandidateSlot)
    assert len(slot.digest) == 64
    assert all(c in "0123456789abcdef" for c in slot.digest)


def test_slot_path_resolves_under_experience_root(tmp_path: Path) -> None:
    """The slot file lands under ``<experience.path>/<slot_dir>`` — not CWD."""
    root = tmp_path / "experience_root"
    store = _make_store(tmp_path)

    slot = store.persist(_make_state_dict())

    expected_dir = (root / "on_device_slot").resolve()
    assert slot.path.resolve().parent == expected_dir
    assert slot.path.is_file()
    # The digest stamps the filename so concurrent candidates never collide.
    assert slot.digest in slot.path.name


def test_persist_creates_parent_dirs_defensively(tmp_path: Path) -> None:
    """A missing experience root + slot dir is created on first persist."""
    store = _make_store(tmp_path)
    # Nothing exists yet under tmp_path.
    assert not (tmp_path / "experience_root").exists()

    slot = store.persist(_make_state_dict())

    assert slot.path.is_file()


def test_load_raises_on_digest_mismatch(tmp_path: Path) -> None:
    """A tampered slot file fails the integrity check on load."""
    store = _make_store(tmp_path)
    slot = store.persist(_make_state_dict())

    # Corrupt the on-disk blob after stamping.
    slot.path.write_bytes(b"corrupted-not-a-torch-payload")

    with pytest.raises(SlotIntegrityError):
        store.load(slot)


def test_load_raises_on_missing_file(tmp_path: Path) -> None:
    """Loading a slot whose file was deleted fails closed."""
    store = _make_store(tmp_path)
    slot = store.persist(_make_state_dict())
    slot.path.unlink()

    with pytest.raises(SlotIntegrityError):
        store.load(slot)
