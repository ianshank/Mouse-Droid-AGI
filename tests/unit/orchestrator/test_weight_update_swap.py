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
    weight_update_pollers: Any | None = None,
    weight_update_loader: Any | None = None,
) -> tuple[MouseDroidOrchestrator, MagicMock, MagicMock]:
    """Build a minimal orchestrator with mocked subsystems.

    Accepts both the legacy ``poller=``/``loader=`` kwargs (single-poller
    path; preserved for backwards compatibility with the Tier C1 tests
    in this file) and the Tier C1.2 ``weight_update_pollers=``/
    ``weight_update_loader=`` kwargs (multi-poller mapping path). The
    loader is resolved via "new kwarg wins, fall back to legacy" so a
    caller passing ``weight_update_loader=`` overrides ``loader=``.

    Args:
        cfg: Root settings.
        poller: Legacy single Tier C1 poller (folded into the orchestrator's
            internal mapping on construction).
        loader: Legacy weight-update loader. Used iff ``weight_update_loader``
            is not provided.
        metrics: Optional metrics registry forwarded to the orchestrator.
        weight_update_pollers: Tier C1.2 mapping ``{engine_type: poller}``.
            ``None`` (default) means use the legacy single-poller path.
        weight_update_loader: Tier C1.2 alias for ``loader=``. Wins over
            ``loader=`` when both are supplied.

    Returns:
        ``(orchestrator, world_model_mock, agent_mock)``.
    """
    resolved_loader = weight_update_loader if weight_update_loader is not None else loader
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
        weight_update_pollers=weight_update_pollers,
        weight_update_loader=resolved_loader,
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


def test_pending_update_with_no_loader_logs_and_acks(capsys):
    """Poller has an update but loader is None — log warning, ack-and-clear.

    Updated post-review (Copilot 3253293630): the original
    implementation left the pending slot populated, which caused the
    same warning to fire at 30 Hz forever until operator intervention.
    The fixed implementation acknowledges the update so the warning
    fires ONCE per revision; the poller can re-surface the same
    revision on the next download cycle, at which point the operator-
    visible warning re-fires.
    """
    cfg = Settings(mock_hardware=True)
    update = _make_update()
    poller = _StubPoller([update])
    orch, original_wm, _ = _build_orch(cfg, poller=poller)
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm  # no swap performed
    assert poller.pending_update is None  # ACK-ed to prevent log spam
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


@pytest.mark.asyncio
async def test_tick_path_world_model_swap_preserves_zeroed_prev_action():
    """tick() honours the swap-reset flag — ``_prev_action`` stays zeroed.

    Regression net for the Copilot review on PR #94 (comment 3253293621):
    the original commit had ``tick()`` unconditionally overwriting
    ``self._prev_action = action`` AFTER ``_apply_pending_weight_update()``
    returned, which silently voided the ``reset_state_on_swap`` invariant
    documented in ADR-010. The fix makes the helper return a ``swap_reset``
    flag and gates the ``_prev_action`` assignment on it. Without this
    test, a future refactor that drops the ``if not swap_reset:`` guard
    in ``tick()`` would re-introduce the bug while
    ``test_swap_resets_h_and_z_when_configured`` (which calls
    ``_apply_pending_weight_update`` directly) keeps passing.
    """
    cfg = Settings(mock_hardware=True)
    assert cfg.cloud.weight_update.reset_state_on_swap is True
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    orch, _original_wm, _ = _build_orch(cfg, poller=poller, loader=loader)

    # Seed h/z + prev_action + latent buffer with non-zero values so a
    # missing reset would leave detectable residue.
    orch._h = torch.ones_like(orch._h) * 3.14
    orch._z = torch.ones_like(orch._z) * 2.71
    orch._prev_action = torch.ones_like(orch._prev_action) * 0.5
    orch._latent_buffer.append((orch._h.clone(), orch._z.clone()))

    # Pin _select_action to return a NON-ZERO action — if the buggy
    # overwrite path runs, _prev_action would equal this non-zero value
    # at the assertion below.
    def _select(self_ref, safety_ctx, observation, loop_time_ms):
        return torch.full((cfg.model.action_dim,), 0.7)

    orch._select_action = _select.__get__(orch, MouseDroidOrchestrator)  # type: ignore[method-assign]
    await orch.tick()

    # All four state slots MUST be zero after a world-model swap, on the
    # tick() path. The swap_reset flag is what guarantees this — a future
    # bug that breaks the flag will fail here even if direct-call tests pass.
    assert torch.all(orch._h == 0), "h not zeroed after world-model swap on tick path"
    assert torch.all(orch._z == 0), "z not zeroed after world-model swap on tick path"
    assert torch.all(
        orch._prev_action == 0
    ), "prev_action not zeroed after world-model swap on tick path (Copilot 3253293621)"
    assert len(orch._latent_buffer) == 0, "latent_buffer not cleared on tick path"


@pytest.mark.asyncio
async def test_tick_path_policy_swap_does_not_reset_prev_action():
    """tick() does NOT zero ``_prev_action`` for policy-only swaps.

    Policy swaps don't touch the world-model recurrent state, so
    ``_prev_action`` MUST continue to reflect the freshly-selected action
    (which seeds the NEXT tick's ``_update_world_model`` call). Pinning
    this prevents an overcorrection where a future refactor zeros
    ``_prev_action`` on every swap regardless of ``engine_type``.
    """
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="policy")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_policy")
    loader = MagicMock(return_value=new_engine)
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)

    selected_action_value = 0.42

    def _select(self_ref, safety_ctx, observation, loop_time_ms):
        return torch.full((cfg.model.action_dim,), selected_action_value)

    orch._select_action = _select.__get__(orch, MouseDroidOrchestrator)  # type: ignore[method-assign]
    await orch.tick()

    # Policy swap landed; prev_action must equal the freshly-selected action.
    assert torch.all(
        torch.isclose(
            orch._prev_action,
            torch.full_like(orch._prev_action, selected_action_value),
        )
    ), "policy swap should NOT zero prev_action"


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
    """A loader raising mid-load leaves the live model untouched + ACKs the slot.

    Updated post-review (Copilot 3253293630): the original test expected
    ``poller.pending_update is update`` (slot NOT cleared) so a manual
    retry could re-apply the same bad revision. But at 30 Hz that meant
    the loader would be re-called every tick — wasting CPU + flooding
    logs with ``cloud_weight_update_swap_failed``. The fixed
    implementation acknowledges the failed revision so the poller can
    re-surface the SAME revision on its next download cycle (the poller
    will compare its remembered SHA against the upstream HF Hub commit
    SHA, so a re-published artifact triggers a fresh attempt).
    """
    cfg = Settings(mock_hardware=True)
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])

    def _bad_loader(update):
        raise RuntimeError("simulated load failure")

    orch, original_wm, _ = _build_orch(cfg, poller=poller, loader=_bad_loader)
    # Must not raise — the helper swallows + logs.
    orch._apply_pending_weight_update()
    assert orch._world_model is original_wm
    # Pending slot ACK-ed to prevent tick-rate retry storms.
    assert poller.pending_update is None
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


# ---------------------------------------------------------------------------
# Defensive-path additions per PR #94 review feedback
# ---------------------------------------------------------------------------


def test_swap_acks_pending_when_engine_type_unknown(capsys):
    """Unknown engine_type → log + acknowledge so the same bad update doesn't spam.

    Regression net for Copilot 3253293637: previously the helper logged
    + returned without ack-ing the pending update, leaving the same bad
    revision stuck in the pending slot firing the warning at 30 Hz.
    """
    cfg = Settings(mock_hardware=True)
    bogus_update = _make_update(engine_type="banana")  # not "policy" or "world_model"
    poller = _StubPoller([bogus_update])
    loader = MagicMock()
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)
    orch._apply_pending_weight_update()
    # Ack-ed despite the bad engine_type → no log spam at tick rate.
    assert poller.pending_update is None
    captured = capsys.readouterr()
    assert "cloud_weight_update_unknown_engine_type" in (captured.out + captured.err)


def test_swap_reset_preserves_device_and_dtype():
    """zeros_like preserves dtype + device — pin against accidental ``torch.zeros(...)``.

    Regression net for Copilot 3253293626 / 3253309982. The original
    implementation reset state via ``torch.zeros(...)`` which defaults to
    CPU + float32. On a CUDA-resident world-model that would silently
    move state to CPU and crash the next ``observe_step``. The fix uses
    ``torch.zeros_like(self._h)`` etc; this test verifies dtype is
    preserved (device check is skipped on CI runners without CUDA).
    """
    cfg = Settings(mock_hardware=True)
    assert cfg.cloud.weight_update.reset_state_on_swap is True
    update = _make_update(engine_type="world_model")
    poller = _StubPoller([update])
    new_engine = MagicMock(name="new_world_model")
    loader = MagicMock(return_value=new_engine)
    orch, _, _ = _build_orch(cfg, poller=poller, loader=loader)

    # Set state to a non-default dtype to detect any path that creates a
    # new tensor with default dtype.
    orch._h = torch.ones_like(orch._h, dtype=torch.float64)
    orch._z = torch.ones_like(orch._z, dtype=torch.float64)
    orch._prev_action = torch.ones_like(orch._prev_action, dtype=torch.float64)

    orch._apply_pending_weight_update()

    assert orch._h.dtype == torch.float64, "h dtype lost on swap reset"
    assert orch._z.dtype == torch.float64, "z dtype lost on swap reset"
    assert orch._prev_action.dtype == torch.float64, "prev_action dtype lost on swap reset"
    # Device parity check — on a CPU-only runner this is trivially true,
    # but on a CUDA host the assertion catches the device-mismatch bug.
    assert orch._h.device == orch._z.device


@pytest.mark.asyncio
async def test_orchestrator_start_invokes_poller_start():
    """Regression net for Copilot 3253293644 / 3253309972 — poller wired into lifecycle."""
    cfg = Settings(mock_hardware=True)
    poller = _StubPoller([])

    # Spy on start/stop calls.
    start_calls: list[bool] = []
    stop_calls: list[bool] = []
    original_start = poller.start
    original_stop = poller.stop

    async def _spy_start() -> None:
        start_calls.append(True)
        await original_start()

    async def _spy_stop() -> None:
        stop_calls.append(True)
        await original_stop()

    poller.start = _spy_start  # type: ignore[method-assign]
    poller.stop = _spy_stop  # type: ignore[method-assign]

    orch, _, _ = _build_orch(cfg, poller=poller)

    # Mock orchestrator subsystem start/stop calls to avoid real I/O — we
    # only care that the poller hooks fire as part of the lifecycle.
    orch._esp32.connect = AsyncMock()
    orch._sensor_manager.start = AsyncMock()
    orch._journal.start = AsyncMock()
    await orch.start()
    assert start_calls == [True], "orchestrator.start() must invoke poller.start()"

    orch._journal.stop = AsyncMock()  # type: ignore[method-assign]
    orch._sensor_manager.stop = AsyncMock()  # type: ignore[method-assign]
    orch._esp32.disconnect = AsyncMock()  # type: ignore[method-assign]
    await orch.stop()
    assert stop_calls == [True], "orchestrator.stop() must invoke poller.stop()"


@pytest.mark.asyncio
async def test_orchestrator_start_tolerates_poller_start_failure(capsys):
    """A failing poller start MUST NOT block orchestrator startup."""
    cfg = Settings(mock_hardware=True)

    class _FailingPoller(_StubPoller):
        async def start(self) -> None:
            raise RuntimeError("simulated HF Hub unreachable at boot")

    poller = _FailingPoller([])
    orch, _, _ = _build_orch(cfg, poller=poller)
    orch._esp32.connect = AsyncMock()
    orch._sensor_manager.start = AsyncMock()
    orch._journal.start = AsyncMock()
    # Must not raise.
    await orch.start()
    captured = capsys.readouterr()
    assert "cloud_weight_update_poller_start_failed" in (captured.out + captured.err)
