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


def test_slot_dir_property_resolves_under_experience_root(tmp_path: Path) -> None:
    """The ``slot_dir`` property exposes the resolved ``<root>/<slot_dir>`` path."""
    root = tmp_path / "experience_root"
    store = _make_store(tmp_path)

    assert store.slot_dir == (root / "on_device_slot").resolve()


def test_mark_active_and_load_active_round_trip(tmp_path: Path) -> None:
    """``mark_active`` writes an active manifest read back by ``load_active``."""
    store = _make_store(tmp_path)
    slot = store.persist(_make_state_dict())

    assert store.load_active() is None  # nothing blessed yet

    store.mark_active(slot)

    assert store.load_active() == slot.digest


def test_load_active_none_when_no_manifest(tmp_path: Path) -> None:
    """A store with no active manifest reports ``None`` (no slot blessed)."""
    store = _make_store(tmp_path)
    store.persist(_make_state_dict())  # candidate exists but not blessed
    assert store.load_active() is None


def test_mark_active_is_idempotent_and_overwrites(tmp_path: Path) -> None:
    """A second ``mark_active`` re-points the active manifest to the new slot."""
    store = _make_store(tmp_path)
    first = store.persist(_make_state_dict())
    torch.manual_seed(1)
    second = store.persist({"layer.weight": torch.randn(2, 2)})

    store.mark_active(first)
    assert store.load_active() == first.digest
    store.mark_active(second)
    assert store.load_active() == second.digest


def test_load_active_ignores_corrupt_manifest(tmp_path: Path) -> None:
    """A non-JSON / malformed active manifest reads back as ``None`` (fail-safe)."""
    store = _make_store(tmp_path)
    store.persist(_make_state_dict())
    store.slot_dir.mkdir(parents=True, exist_ok=True)
    from mousedroid.learning.on_device.slot_store import _ACTIVE_MANIFEST_NAME

    (store.slot_dir / _ACTIVE_MANIFEST_NAME).write_text("not-json{", encoding="utf-8")

    assert store.load_active() is None


def test_load_active_rejects_malformed_digest_shape(tmp_path: Path) -> None:
    """A digest that is not 64-char lowercase hex reads back as ``None`` + warns.

    A valid-JSON manifest whose ``active_digest`` is the wrong shape (short,
    uppercase, non-hex, or a non-string type) must fail safe to "no active
    slot" rather than hand a bogus content-address to the live-policy loader.
    """
    import json

    import structlog

    from mousedroid.learning.on_device.slot_store import (
        _ACTIVE_DIGEST_KEY,
        _ACTIVE_MANIFEST_NAME,
    )

    store = _make_store(tmp_path)
    store.slot_dir.mkdir(parents=True, exist_ok=True)
    manifest = store.slot_dir / _ACTIVE_MANIFEST_NAME

    for bad in ("deadbeef", "Z" * 64, "ABC123" * 10 + "ABCD", "NOTHEX" + "0" * 58):
        manifest.write_text(json.dumps({_ACTIVE_DIGEST_KEY: bad}), encoding="utf-8")
        with structlog.testing.capture_logs() as captured:
            result = store.load_active()
        assert result is None, f"expected None for malformed digest {bad!r}"
        events = [entry.get("event", "") for entry in captured]
        assert "on_device_slot_active_digest_malformed" in events

    # A correctly-shaped digest still round-trips through the same path.
    valid = "a" * 64
    manifest.write_text(json.dumps({_ACTIVE_DIGEST_KEY: valid}), encoding="utf-8")
    assert store.load_active() == valid


def test_persist_overwrites_stale_tmp_file(tmp_path: Path) -> None:
    """A leftover temp blob from an interrupted write never corrupts a persist.

    The write-then-rename uses a fixed temp name; a stale temp file from a prior
    crash must be transparently overwritten so the next persist still produces a
    correctly content-addressed slot.
    """
    from mousedroid.learning.on_device.slot_store import _TMP_SLOT_NAME

    store = _make_store(tmp_path)
    store.slot_dir.mkdir(parents=True, exist_ok=True)
    stale = store.slot_dir / _TMP_SLOT_NAME
    stale.write_bytes(b"stale-interrupted-write")

    slot = store.persist(_make_state_dict())

    assert slot.path.is_file()
    assert not stale.exists()  # renamed away to <digest>.pt
    loaded = store.load(slot)
    assert set(loaded) == set(_make_state_dict())
