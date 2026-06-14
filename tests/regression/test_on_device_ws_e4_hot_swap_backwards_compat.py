"""Regression: WS-E4 off-loop hot-swap is additive + default-OFF byte-identical.

Pins the SAFETY-LOCKED WS-E4 default-OFF invariant: with
``enable_hot_swap=False`` (the default), NO on-device weight-update source and NO
on-device loader are wired AT ALL — the orchestrator's weight-update surface is
byte-identical to #134:

* ``build_on_device_hot_swap_source`` returns ``None`` when the flag is off, when
  on-device learning is disabled, and when the block is absent;
* the orchestrator built from a config WITHOUT hot-swap has NO ``world_model``
  poller from the on-device path (and an unchanged ``world_model`` cloud poller
  when cloud OTA is off — i.e. empty);
* a marked-active slot present on disk is NEVER swapped when the flag is off (the
  swap helper short-circuits — no pending update is ever produced);
* enabling hot-swap requires ``enabled=True`` (cross-field validator, pinned).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import (
    build_on_device_hot_swap_source,
    build_orchestrator,
    build_world_model,
)
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.rssm import RSSM


def _cfg(experience_path: str, *, enable_hot_swap: bool, enabled: bool = True) -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": experience_path, "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": enabled,
                "enable_hot_swap": enable_hot_swap,
                "trigger_min_new_records": 5,
                "check_interval_s": 0.01,
            },
        }
    )


def test_source_none_when_flag_off(tmp_path: Path) -> None:
    """``enable_hot_swap=False`` ⇒ ``build_on_device_hot_swap_source`` is ``None``."""
    cfg = _cfg(str(tmp_path / "exp"), enable_hot_swap=False)
    wm = build_world_model(cfg)
    assert build_on_device_hot_swap_source(cfg, world_model=wm) is None


def test_source_none_when_learning_disabled(tmp_path: Path) -> None:
    """On-device learning disabled ⇒ no hot-swap source even if asked."""
    cfg = _cfg(str(tmp_path / "exp"), enable_hot_swap=False, enabled=False)
    wm = build_world_model(cfg)
    assert build_on_device_hot_swap_source(cfg, world_model=wm) is None


def test_source_none_when_block_absent(tmp_path: Path) -> None:
    """No ``on_device_learning`` block ⇒ no hot-swap source."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01}}
    )
    wm = build_world_model(cfg)
    assert cfg.on_device_learning is None
    assert build_on_device_hot_swap_source(cfg, world_model=wm) is None


def test_orchestrator_no_world_model_poller_when_flag_off(tmp_path: Path) -> None:
    """A flag-off orchestrator has NO on-device world-model swap poller wired."""
    cfg = _cfg(str(tmp_path / "exp"), enable_hot_swap=False)
    orch = build_orchestrator(cfg)
    pollers = orch._weight_update_pollers  # type: ignore[attr-defined]
    # Cloud OTA is off (default poll_interval_s=0) AND hot-swap is off -> empty.
    assert "world_model" not in pollers


def test_marked_active_slot_never_swapped_when_flag_off(tmp_path: Path) -> None:
    """A marked-active slot is NOT swapped into the live model when the flag is off."""
    cfg = _cfg(str(tmp_path / "exp"), enable_hot_swap=False)
    # Promote a slot (mark_active) — the activation step is separate + gated.
    store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=cfg.on_device_learning)
    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)
    slot = store.persist(wm.state_dict())
    store.mark_active(slot)
    assert store.load_active() == slot.digest  # promotion recorded...

    orch = build_orchestrator(cfg)
    prior_model = orch._world_model  # type: ignore[attr-defined]
    # ...but a tick swaps nothing (no poller -> swap helper short-circuits).
    reset = orch._apply_pending_weight_update()  # type: ignore[attr-defined]
    assert reset is False
    assert orch._world_model is prior_model  # type: ignore[attr-defined]


def test_enable_hot_swap_requires_enabled() -> None:
    """``enable_hot_swap=true`` without ``enabled=true`` is rejected at load."""
    with pytest.raises(ValueError, match="enable_hot_swap=true requires"):
        Settings.model_validate(
            {
                "mock_hardware": True,
                "on_device_learning": {"enabled": False, "enable_hot_swap": True},
            }
        )
