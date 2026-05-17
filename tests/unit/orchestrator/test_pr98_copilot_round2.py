"""Tests for the second Copilot review pass on PR #98.

Three new behaviours pinned here:

* Item #A (MED): MissionLifecycle must tick on the emergency-stop branch
  too — without this, a stuck-emergency condition freezes the lifecycle's
  stall counter and silently extends in-flight missions past their stall
  window.
* Item #C (MED): the public ``WeightUpdatePollerProtocol`` keeps its
  pre-C1.2 surface (no ``engine_type``) so external pollers predating
  this PR still satisfy ``isinstance(..., WeightUpdatePollerProtocol)``
  at runtime. The new ``engine_type`` lives on a separate
  ``EngineTypedWeightUpdatePollerProtocol`` extension.
* Item #D (MED): ``gcs_artifact_prefix`` schema validator must reject
  empty / whitespace prefixes — a blank prefix would enumerate the
  entire training bucket and publish every artifact extension to HF Hub.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.cloud.protocol import (
    EngineTypedWeightUpdatePollerProtocol,
    WeightUpdatePollerProtocol,
)
from mousedroid.config.schema import MissionConfig, Settings, WeightUpdatePollConfig
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext

# ---------------------------------------------------------------------------
# Item #A: emergency-stop branch must still tick the mission lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_ticks_on_emergency_stop_branch() -> None:
    """When safety_ctx.is_emergency, the lifecycle must still tick once.

    Otherwise a stuck-emergency condition freezes the lifecycle's stall
    counter and silently extends in-flight missions past their stall
    window. Verified by forcing two emergency-stop ticks and asserting
    that lifecycle.tick was called once on the second (after prev_obs
    is cached).
    """
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)

    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()
    lifecycle.current_state = MissionLifecycleState.RUNNING

    obs = MagicMock()
    obs.vision_features = np.ones(8, dtype=np.float32)

    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)
    # recovery_attempt is awaited from the emergency branch — must be async.
    sensor_manager.recovery_attempt = AsyncMock(return_value=False)

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )

    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    # Force emergency on every tick — sensor recovery also returns
    # an emergency, so the orch hits the emergency-stop early-return.
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=True)

    esp32 = AsyncMock()

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )

    # Two emergency ticks: tick 1 caches prev_obs, tick 2 actually
    # invokes lifecycle.tick — proving the emergency branch routes
    # through ``_maybe_tick_mission_lifecycle`` instead of returning
    # before it.
    await orch.tick()
    await orch.tick()

    assert lifecycle.tick.await_count == 1, (
        "lifecycle.tick MUST fire on the emergency-stop branch too — "
        "otherwise the stall counter freezes during emergencies"
    )
    # And the emergency stop was actually executed (sanity check).
    assert esp32.emergency_stop.await_count >= 2


# ---------------------------------------------------------------------------
# Item #C: protocol back-compat for external pre-C1.2 pollers
# ---------------------------------------------------------------------------


class _LegacyPollerNoEngineType:
    """Minimal poller satisfying the pre-C1.2 protocol surface only.

    No ``engine_type`` attribute, no ``_engine_type`` either — mirrors an
    external poller written before the multi-engine factory landed.
    """

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    @property
    def pending_update(self) -> None:
        return None

    def acknowledge_swap(self, update: object) -> None:
        del update


class _ExtendedPollerWithEngineType(_LegacyPollerNoEngineType):
    """Poller exposing the new ``engine_type`` property."""

    @property
    def engine_type(self) -> str:
        return "world_model"


def test_legacy_poller_satisfies_base_protocol() -> None:
    """Pre-C1.2 external pollers still satisfy WeightUpdatePollerProtocol.

    Regression for Copilot MED: moving ``engine_type`` from the base
    protocol to the extension would otherwise have been a silent
    breaking change for any external poller using
    ``isinstance(poller, WeightUpdatePollerProtocol)`` runtime checks.
    """
    poller = _LegacyPollerNoEngineType()
    assert isinstance(poller, WeightUpdatePollerProtocol)
    # The same poller does NOT satisfy the extension protocol, since it
    # doesn't expose ``engine_type`` — that's the contract.
    assert not isinstance(poller, EngineTypedWeightUpdatePollerProtocol)


def test_extended_poller_satisfies_both_protocols() -> None:
    """A poller exposing ``engine_type`` satisfies both protocols."""
    poller = _ExtendedPollerWithEngineType()
    assert isinstance(poller, WeightUpdatePollerProtocol)
    assert isinstance(poller, EngineTypedWeightUpdatePollerProtocol)
    assert poller.engine_type == "world_model"


def test_orchestrator_accepts_legacy_poller_via_getattr_fallback() -> None:
    """Orchestrator's constructor fallback chain routes legacy pollers safely.

    With no ``engine_type`` property and no ``_engine_type`` attribute,
    the legacy poller defaults to the ``policy`` slot — preserving the
    pre-C1.2 single-poller contract.
    """
    from tests.unit.orchestrator.test_weight_update_swap import _build_orch

    cfg = Settings(mock_hardware=True)
    poller = _LegacyPollerNoEngineType()
    orch, _, _ = _build_orch(cfg, poller=poller)  # type: ignore[arg-type]
    assert "policy" in orch._weight_update_pollers
    assert orch._weight_update_pollers["policy"] is poller


# ---------------------------------------------------------------------------
# Item #D: gcs_artifact_prefix validator must reject blank prefixes
# ---------------------------------------------------------------------------


def test_gcs_artifact_prefix_rejects_empty_string() -> None:
    """Empty string is blocked by min_length=1."""
    with pytest.raises(ValueError, match="gcs_artifact_prefix"):
        WeightUpdatePollConfig(gcs_artifact_prefix="")


def test_gcs_artifact_prefix_rejects_whitespace_only() -> None:
    """Whitespace-only prefix is blocked by the field_validator.

    ``min_length=1`` alone would let ``"   "`` slip through, which
    ``bucket.list_blobs(prefix="   ")`` would treat equivalently to the
    bucket root after server-side normalisation — the validator catches
    this case explicitly.
    """
    with pytest.raises(ValueError, match="non-blank"):
        WeightUpdatePollConfig(gcs_artifact_prefix="   ")


def test_gcs_artifact_prefix_accepts_valid_non_root_prefixes() -> None:
    """Non-empty, non-whitespace prefixes load cleanly.

    Covers the documented defaults (``"trained/"``) plus a fleet-specific
    subpath an operator might wire from a custom YAML.
    """
    assert WeightUpdatePollConfig(gcs_artifact_prefix="trained/").gcs_artifact_prefix == "trained/"
    assert (
        WeightUpdatePollConfig(gcs_artifact_prefix="fleets/alpha/trained/").gcs_artifact_prefix
        == "fleets/alpha/trained/"
    )
    # No trailing slash is also acceptable.
    assert WeightUpdatePollConfig(gcs_artifact_prefix="trained").gcs_artifact_prefix == "trained"
