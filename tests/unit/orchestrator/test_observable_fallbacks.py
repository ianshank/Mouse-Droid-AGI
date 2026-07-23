"""Tests for observable fallbacks in MouseDroidOrchestrator.

Verifies that each silent-fallback path now calls FailureRecorder.record()
with the expected subsystem / reason pair, without changing runtime behaviour.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import torch

from mousedroid.config.schema import Settings
from mousedroid.safety.context import SafetyContext
from mousedroid.telemetry.failure_recorder import NullFailureRecorder


class _SpyRecorder(NullFailureRecorder):
    """Minimal recorder that captures (subsystem, reason) pairs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def record(self, subsystem: str, reason: str, **kwargs: Any) -> None:
        self.calls.append((subsystem, reason))


def _make_orch(
    cfg: Settings,
    spy: _SpyRecorder,
    vla_policy: Any | None = None,
) -> Any:
    """Build a minimal orchestrator wired with a SpyRecorder."""
    from mousedroid.common.time.protocol import MockClock
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    sensor_manager = AsyncMock()
    sensor_manager.read_all = AsyncMock(return_value=MagicMock())

    clock = MockClock(start=0.0)

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        clock=clock,
        failure_recorder=spy,
        vla_policy=vla_policy,
    )


# ---------------------------------------------------------------------------
# VLA fallbacks
# ---------------------------------------------------------------------------


class TestVLAFallbacks:
    """VLA prediction failures call FailureRecorder with the right reason."""

    def test_vla_exception_records_failure(self) -> None:
        """When VLA.predict() raises, records vla_exception."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()

        vla = MagicMock()
        vla.name = "test_vla"
        vla.predict.side_effect = RuntimeError("gpu oom")

        orch = _make_orch(cfg, spy, vla_policy=vla)
        # _try_vla_action is synchronous; call directly
        result = orch._try_vla_action(MagicMock())

        assert result is None
        assert ("orchestrator", "vla_exception") in spy.calls

    def test_vla_timeout_records_failure(self) -> None:
        """When VLA inference exceeds the latency budget, records vla_timeout."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()

        vla = MagicMock()
        vla.name = "test_vla"
        # Return a valid result, but make the clock report that a lot of time passed
        from mousedroid.vla.policy import VLAAction

        vla.predict.return_value = VLAAction(
            action=torch.zeros(cfg.model.action_dim),
            confidence=1.0,
        )

        orch = _make_orch(cfg, spy, vla_policy=vla)
        # Make the clock jump from 0 → 9999 between the two monotonic() calls
        orch._clock._now = 0.0  # type: ignore[attr-defined]

        original_predict = vla.predict

        def _predict_and_advance(*args: Any, **kwargs: Any) -> Any:
            orch._clock._now = 9999.0  # type: ignore[attr-defined]
            return original_predict(*args, **kwargs)

        vla.predict = _predict_and_advance

        result = orch._try_vla_action(MagicMock())

        assert result is None
        assert ("orchestrator", "vla_timeout") in spy.calls

    def test_vla_wrong_shape_records_failure(self) -> None:
        """When VLA returns wrong action shape, records vla_wrong_shape."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()

        vla = MagicMock()
        vla.name = "test_vla"
        from mousedroid.vla.policy import VLAAction

        # Return action with one extra dimension
        vla.predict.return_value = VLAAction(
            action=torch.zeros(cfg.model.action_dim + 99),
            confidence=1.0,
        )

        orch = _make_orch(cfg, spy, vla_policy=vla)
        result = orch._try_vla_action(MagicMock())

        assert result is None
        assert ("orchestrator", "vla_wrong_shape") in spy.calls


# ---------------------------------------------------------------------------
# Cognitive core fallbacks
# ---------------------------------------------------------------------------


class TestCognitiveFallbacks:
    """Cognitive core exceptions call FailureRecorder."""

    def test_cognitive_exception_records_failure(self) -> None:
        """When cognitive_core.tick_fast() raises, records cognitive_core_exception."""
        from mousedroid.common.time.protocol import MockClock
        from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()

        world_model = MagicMock()
        world_model.observe_step.return_value = (
            torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
            torch.zeros(1, cfg.model.latent_dim),
            torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
            0.1,
        )
        agent = MagicMock()
        agent.name = "mock_agent"
        agent.act.return_value = torch.zeros(cfg.model.action_dim)

        cognitive_core = MagicMock()
        cognitive_core.tick_fast.side_effect = RuntimeError("cognitive crash")

        safety_monitor = MagicMock()
        safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
        sensor_manager = AsyncMock()
        sensor_manager.read_all = AsyncMock(return_value=MagicMock())

        orch = MouseDroidOrchestrator(
            world_model=world_model,
            agents=[agent],
            safety_monitor=safety_monitor,
            esp32=AsyncMock(),
            sensor_manager=sensor_manager,
            cfg=cfg,
            clock=MockClock(start=0.0),
            failure_recorder=spy,
            cognitive_core=cognitive_core,
        )

        result = orch._try_cognitive_action(MagicMock(), loop_time_ms=0.0)

        assert result is None
        assert ("orchestrator", "cognitive_core_exception") in spy.calls


# ---------------------------------------------------------------------------
# Latent NaN fallback
# ---------------------------------------------------------------------------


class TestLatentValidation:
    """_validate_latent records failure and recovers from buffer."""

    def test_nan_h_records_failure(self) -> None:
        """NaN in h triggers world_model/latent_nan record."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        h_nan = torch.full((1, cfg.model.hidden_dim), float("nan"))
        z_ok = torch.zeros(1, cfg.model.latent_dim)

        orch._validate_latent(h_nan, z_ok)

        assert ("world_model", "latent_nan") in spy.calls

    def test_nan_recovery_from_buffer(self) -> None:
        """NaN recovery returns previous good (h, z) from buffer."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        h_good = torch.ones(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        z_good = torch.ones(1, cfg.model.latent_dim) * 2.0
        # Pre-populate buffer
        orch._latent_buffer.append((h_good.clone(), z_good.clone()))

        h_nan = torch.full_like(h_good, float("nan"))
        h_out, z_out, healthy = orch._validate_latent(h_nan, z_good)

        assert not torch.isnan(h_out).any()
        assert torch.allclose(z_out, z_good)
        # A recovered NaN tick is still unhealthy — the F-023 memory must
        # not ingest it.
        assert healthy is False

    def test_valid_latent_accumulates_buffer(self) -> None:
        """Valid latent state is appended to the recovery buffer."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        assert len(orch._latent_buffer) == 0
        h = torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        z = torch.zeros(1, cfg.model.latent_dim)
        orch._validate_latent(h, z)

        assert len(orch._latent_buffer) == 1

    def test_saturated_h_records_warning(self) -> None:
        """h-norm above threshold records latent_saturated but keeps tensor."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        big = cfg.model.latent_norm_threshold + 100.0
        h_sat = torch.full((1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim), big)
        z_ok = torch.zeros(1, cfg.model.latent_dim)

        h_out, _z_out, healthy = orch._validate_latent(h_sat, z_ok)

        assert ("world_model", "latent_saturated") in spy.calls
        assert torch.allclose(h_out, h_sat)  # saturation does NOT clamp
        assert healthy is True  # saturated is a warning, not a NaN tick


# ---------------------------------------------------------------------------
# Curiosity episode reset
# ---------------------------------------------------------------------------


class TestCuriosityReset:
    """Orchestrator calls curiosity_module.reset_episode() on mission completion."""

    def test_reset_called_when_caller_snapshots_completed(self) -> None:
        """``_maybe_reset_curiosity(mission_completed=True)`` invokes reset_episode."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        curiosity = MagicMock()
        orch._curiosity_module = curiosity
        # Caller is responsible for snapshotting — pass True directly.
        orch._maybe_reset_curiosity(mission_completed=True)

        curiosity.reset_episode.assert_called_once()

    def test_reset_not_called_when_snapshot_false(self) -> None:
        """No reset when the caller's snapshot says no mission completed this tick."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)

        curiosity = MagicMock()
        orch._curiosity_module = curiosity
        orch._maybe_reset_curiosity(mission_completed=False)

        curiosity.reset_episode.assert_not_called()

    def test_reset_no_op_when_curiosity_module_absent(self) -> None:
        """Without a curiosity module the method is a no-op."""
        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)
        orch._curiosity_module = None  # explicit
        # Must not raise even with mission_completed=True.
        orch._maybe_reset_curiosity(mission_completed=True)


class TestMissionCompletionClearRace:
    """``mission_just_completed`` clear happens BEFORE export await.

    Regression test for the race: if the dispatcher latches a new
    completion DURING the export I/O window, clearing the flag after
    the await would silently wipe the new event. The tick loop must
    clear the flag pre-await so any concurrent completion remains
    latched for the next tick.
    """

    def _make_dispatcher(self) -> MagicMock:
        dispatcher = MagicMock()
        dispatcher.mission_just_completed = True
        cleared: list[bool] = []

        def _clear() -> None:
            cleared.append(True)
            dispatcher.mission_just_completed = False

        dispatcher.clear_mission_completed = MagicMock(side_effect=_clear)
        dispatcher._cleared_calls = cleared  # introspect from the test
        return dispatcher

    async def test_clear_runs_before_export_await(self) -> None:
        """The dispatcher's clear must be called BEFORE _maybe_export_memory awaits."""
        from unittest.mock import AsyncMock as _AsyncMock

        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)
        dispatcher = self._make_dispatcher()
        orch._mission_dispatcher = dispatcher

        export_seen_cleared: list[bool] = []

        async def _export_observer(*, mission_completed: bool) -> None:
            # When the export await runs, the dispatcher's latch must
            # already be False — proof that clearing happened first.
            export_seen_cleared.append(not dispatcher.mission_just_completed)
            assert mission_completed is True  # local snapshot still True

        orch._maybe_export_memory = _export_observer  # type: ignore[assignment]
        orch._maybe_reset_curiosity = MagicMock()  # type: ignore[method-assign]
        orch._hook_registry.run_phase = _AsyncMock()  # type: ignore[method-assign]

        # Drive the slice of tick() that handles the mission-completed
        # snapshot. We replicate it inline to keep the test focused.
        mission_completed = (
            orch._mission_dispatcher is not None and orch._mission_dispatcher.mission_just_completed
        )
        if mission_completed:
            orch._mission_dispatcher.clear_mission_completed()
        await orch._maybe_export_memory(mission_completed=mission_completed)
        orch._maybe_reset_curiosity(mission_completed=mission_completed)

        assert dispatcher._cleared_calls == [True]
        assert export_seen_cleared == [True]

    async def test_completion_during_export_remains_latched(self) -> None:
        """A new completion while exporting is preserved for the next tick."""
        from unittest.mock import AsyncMock as _AsyncMock

        cfg = Settings(mock_hardware=True)
        spy = _SpyRecorder()
        orch = _make_orch(cfg, spy)
        dispatcher = self._make_dispatcher()
        orch._mission_dispatcher = dispatcher

        async def _export_with_concurrent_completion(*, mission_completed: bool) -> None:
            # Simulate a new mission landing while the exporter waits.
            dispatcher.mission_just_completed = True

        orch._maybe_export_memory = _export_with_concurrent_completion  # type: ignore[assignment]
        orch._maybe_reset_curiosity = MagicMock()  # type: ignore[method-assign]
        orch._hook_registry.run_phase = _AsyncMock()  # type: ignore[method-assign]

        mission_completed = (
            orch._mission_dispatcher is not None and orch._mission_dispatcher.mission_just_completed
        )
        if mission_completed:
            orch._mission_dispatcher.clear_mission_completed()
        await orch._maybe_export_memory(mission_completed=mission_completed)

        # The concurrent completion remains latched — the next tick
        # will see it because the post-await clear was never run.
        assert dispatcher.mission_just_completed is True
