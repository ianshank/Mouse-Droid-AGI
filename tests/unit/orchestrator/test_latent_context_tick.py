"""Orchestrator-seam tests for the F-023 bounded-context latent memory.

Uses a deterministic INPUT-DEPENDENT stub world model (not a fixed-return
MagicMock — that would make OFF-vs-ON trajectory comparisons vacuously equal).
Pins: the disabled path is byte-identical to the pre-feature recurrence, the
enabled blend reaches the carried state AND action selection (S1b
incorporation), raw (pre-blend) states are what the memory stores, NaN ticks
skip the memory entirely, the OTA swap resets + re-arms, and the mission
boundary re-arms the sink under ``recapture_on_mission``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from mousedroid.cloud.protocol import PendingWeightUpdate
from mousedroid.config.schema import Settings
from mousedroid.factory import build_latent_context
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


class _RecurrentStubWorldModel:
    """Deterministic input-dependent dynamics: h' = 0.9h + 0.05, z' = 0.9z - 0.05.

    NaN ticks can be scheduled via ``nan_on_calls`` to exercise the recovery
    path.
    """

    def __init__(self, nan_on_calls: set[int] | None = None) -> None:
        self._nan_on_calls = nan_on_calls or set()
        self.calls = 0

    def observe_step(
        self,
        observation: Any,
        prev_action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        self.calls += 1
        if self.calls in self._nan_on_calls:
            return torch.full_like(h, float("nan")), z.clone(), h.clone(), 0.0
        return 0.9 * h + 0.05, 0.9 * z - 0.05, h.clone(), 0.1

    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return h, z, torch.zeros(1, 1)


def _build_orch(
    cfg: Settings,
    *,
    world_model: _RecurrentStubWorldModel | None = None,
    poller: Any | None = None,
    loader: Any | None = None,
) -> tuple[MouseDroidOrchestrator, _RecurrentStubWorldModel, MagicMock]:
    wm = world_model or _RecurrentStubWorldModel()
    agent = MagicMock(name="agent")
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    safety_monitor = MagicMock(name="safety_monitor")
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    sensor_manager = AsyncMock(name="sensor_manager")
    sensor_manager.read_all = AsyncMock(return_value=MagicMock())
    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(name="esp32"),
        sensor_manager=sensor_manager,
        cfg=cfg,
        weight_update_poller=poller,
        weight_update_loader=loader,
        latent_context=build_latent_context(cfg),
    )
    return orch, wm, agent


async def _run_ticks(orch: MouseDroidOrchestrator, n: int) -> list[torch.Tensor]:
    trajectory: list[torch.Tensor] = []
    for _ in range(n):
        await orch.tick()
        trajectory.append(orch._h.clone())
    return trajectory


def _expected_baseline(n: int, h_dim: int) -> list[torch.Tensor]:
    """The pre-feature recurrence under the stub: h_{t+1} = 0.9 h_t + 0.05."""
    h = torch.zeros(1, h_dim)
    out: list[torch.Tensor] = []
    for _ in range(n):
        h = 0.9 * h + 0.05
        out.append(h.clone())
    return out


_MEMORY_ON = {
    "enabled": True,
    "sink_warmup_ticks": 0,
    "recent_size": 4,
    "stride": 2,
    "blend_weight": 0.3,
}


@pytest.mark.asyncio
async def test_block_absent_is_none_and_trajectory_byte_identical() -> None:
    cfg = Settings(mock_hardware=True)
    orch, _, _ = _build_orch(cfg)
    assert orch._latent_context is None
    trajectory = await _run_ticks(orch, 5)
    h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    for got, want in zip(trajectory, _expected_baseline(5, h_dim), strict=True):
        assert torch.equal(got, want)


@pytest.mark.asyncio
async def test_disabled_block_is_none() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": {"enabled": False}})
    orch, _, _ = _build_orch(cfg)
    assert orch._latent_context is None


@pytest.mark.asyncio
async def test_enabled_blend_reaches_carried_state_and_action_selection() -> None:
    """S1b incorporation: the blend changes the carried h AND what the agent sees."""
    cfg_on = Settings.model_validate(
        {"mock_hardware": True, "world_model_memory": dict(_MEMORY_ON)}
    )
    cfg_lambda0 = Settings.model_validate(
        {
            "mock_hardware": True,
            "world_model_memory": {**_MEMORY_ON, "blend_weight": 0.0},
        }
    )
    orch_on, _, agent_on = _build_orch(cfg_on)
    orch_l0, _, agent_l0 = _build_orch(cfg_lambda0)
    await _run_ticks(orch_on, 10)
    await _run_ticks(orch_l0, 10)
    assert not torch.equal(orch_on._h, orch_l0._h)
    h_seen_on = agent_on.act.call_args[0][0]
    h_seen_l0 = agent_l0.act.call_args[0][0]
    assert not torch.equal(h_seen_on, h_seen_l0)
    # λ=0 keeps the pre-feature trajectory exactly (identity contextualize).
    h_dim = cfg_on.model.hidden_dim + cfg_on.model.cfc_hidden_dim
    assert torch.equal(orch_l0._h, _expected_baseline(10, h_dim)[-1])


@pytest.mark.asyncio
async def test_memory_stores_raw_pre_blend_state() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": dict(_MEMORY_ON)})
    orch, _, _ = _build_orch(cfg)
    # Tick 1: every key equals the query, so the blend is exactly identity and
    # the carried state stays on the baseline recurrence. Tick 2's keys differ
    # from its query, so the blend shifts the carried state — while the ring
    # still holds the RAW (pre-blend) tick-2 state.
    await _run_ticks(orch, 2)
    ctx = orch._latent_context
    assert ctx is not None
    h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    raw_h2 = _expected_baseline(2, h_dim)[1]
    z1 = 0.9 * torch.zeros(1, cfg.model.latent_dim) - 0.05
    raw_z2 = 0.9 * z1 - 0.05
    raw_hz2 = torch.cat([raw_h2, raw_z2], dim=-1).reshape(-1)
    assert torch.equal(ctx._ring[-1], raw_hz2)
    # The carried state was blended (keys ≠ query on tick 2), so it differs
    # from the raw stored state.
    assert not torch.equal(orch._h, raw_h2)


@pytest.mark.asyncio
async def test_nan_tick_skips_memory_and_preserves_recovery() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": dict(_MEMORY_ON)})
    wm = _RecurrentStubWorldModel(nan_on_calls={3})
    orch, _, _ = _build_orch(cfg, world_model=wm)
    await _run_ticks(orch, 2)
    ctx = orch._latent_context
    assert ctx is not None
    len_before = len(ctx)
    h_before = orch._h.clone()
    await _run_ticks(orch, 1)  # the NaN tick
    assert len(ctx) == len_before  # nothing ingested
    assert bool(torch.isfinite(orch._h).all())  # buffer recovery still works
    # Recovery restores the RAW pre-blend state from the recovery buffer —
    # NOT the blended state the orchestrator was carrying before the NaN tick.
    raw_last_h, _raw_last_z = orch._latent_buffer[-1]
    assert torch.equal(orch._h, raw_last_h)
    assert not torch.equal(orch._h, h_before)


@pytest.mark.asyncio
async def test_ota_world_model_swap_resets_memory_and_rearms_sink() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": dict(_MEMORY_ON)})
    update = PendingWeightUpdate(
        repo_id="ianshank/test",
        filename="weights.bin",
        revision="sha-A",
        sha256="0" * 64,
        local_path=Path("/tmp/weights.bin"),
        downloaded_at=time.time(),
        engine_type="world_model",
    )
    # A stateful stub (not a MagicMock attribute) so the in-tick swap helper
    # sees no update during warm-up and the ack actually clears the slot.
    poller = MagicMock(name="poller")
    poller.pending_update = None
    poller.acknowledge_swap = MagicMock(
        side_effect=lambda _u: setattr(poller, "pending_update", None)
    )
    loader = MagicMock(return_value=_RecurrentStubWorldModel())
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)
    await _run_ticks(orch, 4)
    ctx = orch._latent_context
    assert ctx is not None
    sink_before = ctx._sink
    assert sink_before is not None
    assert len(ctx) > 0
    poller.pending_update = update  # surface the update for the swap helper
    orch._apply_pending_weight_update()
    assert len(ctx) == 0  # everything cleared
    assert ctx._sink is None  # sink gone, warmup re-armed
    assert poller.pending_update is None  # ACK-ed
    await _run_ticks(orch, 1)
    # Warmup re-armed (warmup=0): a FRESH sink is captured under the new
    # weights on the first post-swap tick. The swap also zeroed (h, z), so the
    # fresh sink is exactly the first-step recurrence from zero state.
    assert ctx._sink is not None
    h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    fresh_h = _expected_baseline(1, h_dim)[0]
    fresh_z = 0.9 * torch.zeros(1, cfg.model.latent_dim) - 0.05
    fresh_hz = torch.cat([fresh_h, fresh_z], dim=-1).reshape(-1)
    assert torch.equal(ctx._sink, fresh_hz)


@pytest.mark.asyncio
async def test_mission_boundary_rearms_sink_when_configured() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": dict(_MEMORY_ON)})
    orch, _, _ = _build_orch(cfg)
    await _run_ticks(orch, 4)
    ctx = orch._latent_context
    assert ctx is not None
    assert ctx._sink is not None
    ring_len = len(ctx._ring)
    orch._maybe_rearm_latent_sink(mission_completed=True)
    assert ctx._sink is None  # sink re-armed
    assert len(ctx._ring) == ring_len  # ring retained


@pytest.mark.asyncio
async def test_mission_boundary_keeps_sink_when_recapture_disabled() -> None:
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "world_model_memory": {**_MEMORY_ON, "recapture_on_mission": False},
        }
    )
    orch, _, _ = _build_orch(cfg)
    await _run_ticks(orch, 4)
    ctx = orch._latent_context
    assert ctx is not None
    sink = ctx._sink
    assert sink is not None
    orch._maybe_rearm_latent_sink(mission_completed=True)
    assert ctx._sink is not None
    assert torch.equal(ctx._sink, sink)
