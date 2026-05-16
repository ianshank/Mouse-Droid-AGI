"""Tests for the Tier C1 atomic post-tick OTA weight-update swap."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from mousedroid.cloud.protocol import PendingWeightUpdate
from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.telemetry.metrics import MetricsRegistry

# ---------------------------------------------------------------------------
# Stub poller — surfaces a single PendingWeightUpdate, acks via slot clear
# ---------------------------------------------------------------------------


class _StubPoller:
    """Minimal stub conforming to ``WeightUpdatePollerProtocol`` for tests."""

    def __init__(self, updates: list[PendingWeightUpdate] | None = None) -> None:
        self._queue: list[PendingWeightUpdate] = list(updates or [])
        self._pending: PendingWeightUpdate | None = None
        self._advance()
        self.ack_calls: list[PendingWeightUpdate] = []

    def _advance(self) -> None:
        if self._pending is None and self._queue:
            self._pending = self._queue.pop(0)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    @property
    def pending_update(self) -> PendingWeightUpdate | None:
        return self._pending

    def acknowledge_swap(self, update: PendingWeightUpdate) -> None:
        if self._pending is update:
            self.ack_calls.append(update)
            self._pending = None
            self._advance()


def _make_update(
    *,
    engine_type: str = "world_model",
    revision: str = "sha-A",
    repo_id: str = "ianshank/test",
    local_path: Path | None = None,
) -> PendingWeightUpdate:
    return PendingWeightUpdate(
        repo_id=repo_id,
        filename="weights.bin",
        revision=revision,
        sha256="0" * 64,
        local_path=local_path or Path("/tmp/weights.bin"),
        downloaded_at=time.time(),
        engine_type=engine_type,
    )


def _build_orch(
    cfg: Settings,
    *,
    poller: Any | None = None,
    loader: Any | None = None,
    metrics: MetricsRegistry | None = None,
) -> tuple[MouseDroidOrchestrator, MagicMock, MagicMock]:
    """Build a minimal orchestrator with mocked subsystems."""
    world_model = MagicMock(name="world_model")
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        0.1,
    )
    agent = MagicMock(name="agent")
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety_monitor = MagicMock(name="safety_monitor")
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    sensor_manager = AsyncMock(name="sensor_manager")
    sensor_manager.read_all = AsyncMock(return_value=MagicMock())

    esp32 = AsyncMock(name="esp32")

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        weight_update_poller=poller,
        weight_update_loader=loader,
        metrics=metrics,
    )
    return orch, world_model, agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pending_update_none_is_noop():
    """No poller wired => swap helper is a no-op (byte-identical pre-C1)."""
    cfg = Settings(mock_hardware=True)
    orch, original_wm, _ = _build_orch(cfg)
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm


def test_pending_update_with_no_loader_logs_and_skips(capsys):
    """Poller has an update but loader is None — log warning, no swap."""
    cfg = Settings(mock_hardware=True)
    update = _make_update()
    poller = _StubPoller([update])
    orch, original_wm, _ = _build_orch(cfg, poller=poller)
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm
    assert poller.pending_update is update  # not cleared
    captured = capsys.readouterr()
    assert "cloud_weight_update_swap_skipped_no_loader" in (captured.out + captured.err)


@pytest.mark.asyncio
async def test_swap_happens_after_select_action():
    """Swap fires AFTER ``_select_action`` returns — call-order regression."""
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    orch, original_wm, _ = _build_orch(cfg, poller=poller, loader=loader)

    # Track the order: _select_action sees the OLD world_model; the swap
    # happens immediately after. We capture the world_model identity at
    # the moment select_action runs.
    captured: dict[str, Any] = {}

    def _select(self_ref, safety_ctx, observation, loop_time_ms):
        # Use the orchestrator's private attr at the time of selection.
        captured["wm_at_select"] = orch._world_model
        return torch.zeros(cfg.model.action_dim)

    orch._select_action = _select.__get__(orch, MouseDroidOrchestrator)  # type: ignore[method-assign]
    await orch.tick()
    assert captured["wm_at_select"] is original_wm
    assert orch._world_model is new_engine  # swap landed after select_action


def test_swap_resets_h_and_z_when_configured():
    """``reset_state_on_swap=True`` (default) zeros h/z + prev_action + clears buffer."""
    cfg = Settings(mock_hardware=True)
    assert cfg.cloud.weight_update.reset_state_on_swap is True
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)
    # Seed h/z and the latent buffer with non-zero values.
    orch._h = torch.ones_like(orch._h) * 3.14
    orch._z = torch.ones_like(orch._z) * 2.71
    orch._latent_buffer.append((orch._h.clone(), orch._z.clone()))
    assert orch._latent_buffer

    orch._apply_pending_weight_update()
    assert torch.all(orch._h == 0)
    assert torch.all(orch._z == 0)
    assert torch.all(orch._prev_action == 0)
    assert len(orch._latent_buffer) == 0


def test_swap_preserves_state_when_reset_disabled():
    """``reset_state_on_swap=False`` keeps the recurrent state intact."""
    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.reset_state_on_swap = False
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)
    orch._h = torch.ones_like(orch._h)
    orch._z = torch.ones_like(orch._z) * 5.0

    orch._apply_pending_weight_update()
    assert torch.all(orch._h == 1)
    assert torch.all(orch._z == 5.0)


def test_swap_emits_metric(capsys):
    """A successful swap increments ``cloud_weight_update_swaps_total``."""
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    metrics = MetricsRegistry(MetricsConfig())
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader, metrics=metrics)

    orch._apply_pending_weight_update()
    rendered = metrics.render_prometheus()
    assert "mousedroid_cloud_weight_update_swaps_total" in rendered
    assert 'engine_type="world_model"' in rendered
    # Structured log fired.
    captured = capsys.readouterr()
    assert "cloud_weight_update_swap_applied" in (captured.out + captured.err)


def test_multiple_updates_apply_in_order():
    """Two pending updates apply on two consecutive helper calls."""
    cfg = Settings(mock_hardware=True)
    updates = [
        _make_update(engine_type="world_model", revision="sha-A"),
        _make_update(engine_type="world_model", revision="sha-B"),
    ]
    poller = _StubPoller(updates)
    seen_revisions: list[str] = []

    def _loader(update):
        seen_revisions.append(update.revision)
        return MagicMock(name=f"wm_{update.revision}")

    orch, _, _ = _build_orch(cfg, poller=poller, loader=_loader)
    orch._apply_pending_weight_update()
    orch._apply_pending_weight_update()
    assert seen_revisions == ["sha-A", "sha-B"]
    assert poller.ack_calls == updates


def test_loader_exception_does_not_corrupt_live_model(capsys):
    """A loader raising mid-load leaves the live model untouched."""
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])

    def _bad_loader(update):
        raise RuntimeError("simulated load failure")

    orch, original_wm, _ = _build_orch(cfg, poller=poller, loader=_bad_loader)
    # Must not raise — the helper swallows + logs.
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm
    # Pending slot NOT cleared so a manual retry / next poll can re-apply.
    assert poller.pending_update is update
    captured = capsys.readouterr()
    assert "cloud_weight_update_swap_failed" in (captured.out + captured.err)


def test_swap_routes_policy_engine_type():
    """``engine_type="policy"`` swaps the VLA policy slot, not the world model."""
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="policy")
    poller = _StubPoller([update])
    new_policy = MagicMock(name="new_policy")
    loader = MagicMock(return_value=new_policy)
    orch, original_wm, _ = _build_orch(cfg, poller=poller, loader=loader)
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm
    assert orch._vla_policy is new_policy
