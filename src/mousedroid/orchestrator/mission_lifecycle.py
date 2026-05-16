"""Mission lifecycle state machine (Tier C2 / C2.2).

Wraps an :class:`~mousedroid.harness.protocol.TaskTrackerProtocol` and
adds VLM-driven goal-progress feedback plus LLM-driven adaptive replan
on stall. Implemented as a small, deterministic state machine over the
:class:`MissionLifecycleState` enum (an extension of ``TaskStatus`` with
a ``REPLANNING`` state — :class:`TaskStatus` itself stays unchanged to
preserve backward compatibility with the existing tracker).

States and transitions
----------------------
::

    PENDING ──> RUNNING ──┬─> SUCCEEDED       (VLM score >= success_threshold)
                          ├─> REPLANNING ──> RUNNING   (LLM returned new GoalVector)
                          ├─> REPLANNING ──> FAILED    (LLM returned None or limit hit)
                          └─> FAILED                   (explicit fail() call)

Architecture invariants
-----------------------
* Pure-Python state machine, no I/O on the hot path apart from the
  optional VLM score call and the (async) LLM replan call.
* Every threshold and window comes from
  :class:`mousedroid.config.schema.MissionConfig`; nothing is hardcoded.
* When ``cfg.mission.replan_enabled is False`` (the default), the
  lifecycle never transitions to ``REPLANNING`` and never calls the LLM
  gateway. Existing deployments stay byte-identical.
* Structured logs and Prometheus counters surface every transition.
"""

from __future__ import annotations

import enum
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover - 3.10 fallback only

    class StrEnum(str, enum.Enum):
        """Backport of ``enum.StrEnum`` for Python 3.10."""


if TYPE_CHECKING:
    from torch import Tensor

    from mousedroid.config.schema import MissionConfig
    from mousedroid.harness.protocol import TaskTrackerProtocol
    from mousedroid.llm_gateway.protocol import GoalVector
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class MissionLifecycleState(StrEnum):
    """Lifecycle states for a managed mission.

    Mirrors :class:`mousedroid.harness.protocol.TaskStatus` for the
    overlap (PENDING / RUNNING / SUCCEEDED / FAILED) and adds the
    ``REPLANNING`` state used during LLM-driven adaptive replan.
    """

    PENDING = "pending"
    RUNNING = "running"
    REPLANNING = "replanning"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TERMINAL_STATES: frozenset[MissionLifecycleState] = frozenset(
    {MissionLifecycleState.SUCCEEDED, MissionLifecycleState.FAILED}
)


@runtime_checkable
class MissionReplannerProtocol(Protocol):
    """Async LLM-backed replanner used by :class:`MissionLifecycle`.

    Implementations submit a fresh replan request given the current
    mission goal text and the last VLM progress score, returning a
    :class:`GoalVector` on success or ``None`` to fail the mission.
    """

    async def submit_replan_request(
        self,
        *,
        mission_id: str,
        goal_text: str,
        last_progress: float,
    ) -> GoalVector | None:
        """Request a replan from the LLM gateway."""
        ...


@dataclass
class _MissionState:
    """Mutable per-mission lifecycle state."""

    mission_id: str
    goal_text: str
    state: MissionLifecycleState = MissionLifecycleState.PENDING
    started_at_s: float | None = None
    last_progress: float = 0.0
    stall_counter: int = 0
    replan_count: int = 0
    progress_history: list[float] = field(default_factory=list)
    last_goal_vector: GoalVector | None = None


@dataclass(frozen=True)
class MissionTickResult:
    """Result of one :meth:`MissionLifecycle.tick` call."""

    state: MissionLifecycleState
    progress: float
    transitioned: bool
    reason: str | None = None


class MissionLifecycle:
    """Tier C2 / C2.2 mission state machine.

    Construct one lifecycle per orchestrator. Submit one mission via
    :meth:`start_mission`, then call :meth:`tick` once per control-loop
    iteration with the current and previous observation tensors and the
    natural-language goal text.
    """

    def __init__(
        self,
        cfg: MissionConfig,
        *,
        task_tracker: TaskTrackerProtocol | None = None,
        vlm_progress: VLMProgressHead | None = None,
        replanner: MissionReplannerProtocol | None = None,
        metrics: MetricsRegistry | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Build the lifecycle.

        Args:
            cfg: Mission configuration (thresholds + replan toggles).
            task_tracker: Optional :class:`TaskTrackerProtocol` for
                bookkeeping. The lifecycle does NOT depend on the tracker
                for state transitions; it forwards terminal states for
                operators that want a unified active-task list.
            vlm_progress: Optional :class:`VLMProgressHead`. When ``None``,
                :meth:`tick` returns a constant zero progress score so the
                lifecycle stays in RUNNING until externally completed.
            replanner: Optional async replanner. Required only when
                ``cfg.replan_enabled`` is True; otherwise unused.
            metrics: Optional shared metrics registry. When supplied, every
                state transition / replan increments the corresponding
                Tier C2 counter family.
            clock: Optional callable returning ``float`` seconds (e.g.
                :func:`time.monotonic`). Used only for active-duration
                accounting on the SUCCEEDED / FAILED transitions.
        """
        self._cfg = cfg
        self._tracker = task_tracker
        self._vlm = vlm_progress
        self._replanner = replanner
        self._metrics = metrics
        self._clock = clock if clock is not None else time.monotonic
        self._mission: _MissionState | None = None
        _log.info(
            "mission_lifecycle_init",
            replan_enabled=cfg.replan_enabled,
            success_threshold=cfg.success_threshold,
            stall_threshold=cfg.stall_threshold,
            stall_window_ticks=cfg.stall_window_ticks,
            max_replans_per_mission=cfg.max_replans_per_mission,
        )

    @property
    def current_state(self) -> MissionLifecycleState | None:
        """Current mission state, or ``None`` when no mission is active."""
        return self._mission.state if self._mission is not None else None

    @property
    def replan_count(self) -> int:
        """How many times the current mission has replanned."""
        return self._mission.replan_count if self._mission is not None else 0

    def start_mission(self, mission_id: str, goal_text: str) -> None:
        """Begin a new mission, transitioning from PENDING to RUNNING.

        Args:
            mission_id: Stable identifier for the mission.
            goal_text: Natural-language description scored by the VLM
                progress head every tick.
        """
        now = float(self._clock())
        self._mission = _MissionState(
            mission_id=mission_id,
            goal_text=goal_text,
            state=MissionLifecycleState.PENDING,
        )
        self._transition(MissionLifecycleState.RUNNING, reason="start")
        self._mission.started_at_s = now
        _log.info("mission_started", mission_id=mission_id, goal_text=goal_text)

    async def tick(
        self,
        obs_t: Tensor,
        obs_tminus1: Tensor,
    ) -> MissionTickResult:
        """Advance the lifecycle by one tick.

        Args:
            obs_t: Current observation tensor.
            obs_tminus1: Previous observation tensor (same shape).

        Returns:
            :class:`MissionTickResult` describing the new state and the
            progress score from the VLM.
        """
        if self._mission is None or self._mission.state in _TERMINAL_STATES:
            return MissionTickResult(
                state=self._mission.state if self._mission else MissionLifecycleState.PENDING,
                progress=0.0,
                transitioned=False,
            )

        progress = self._score_progress(obs_t, obs_tminus1)
        self._mission.last_progress = progress
        self._mission.progress_history.append(progress)

        # Success path takes priority over stall — high progress wins.
        if progress >= self._cfg.success_threshold:
            self._transition(MissionLifecycleState.SUCCEEDED, reason="progress_threshold")
            return MissionTickResult(
                state=MissionLifecycleState.SUCCEEDED,
                progress=progress,
                transitioned=True,
                reason="progress_threshold",
            )

        # Stall detection: progress below stall_threshold counts as one
        # stalled tick; once we accumulate stall_window_ticks of them in
        # a row, transition to REPLANNING (replan-enabled) or stay put.
        if progress < self._cfg.stall_threshold:
            self._mission.stall_counter += 1
        else:
            self._mission.stall_counter = 0

        if (
            self._cfg.replan_enabled
            and self._mission.state == MissionLifecycleState.RUNNING
            and self._mission.stall_counter >= self._cfg.stall_window_ticks
        ):
            return await self._handle_stall()

        return MissionTickResult(
            state=self._mission.state,
            progress=progress,
            transitioned=False,
        )

    def fail(self, *, reason: str) -> None:
        """Force the mission to FAILED with ``reason``."""
        if self._mission is None or self._mission.state in _TERMINAL_STATES:
            return
        self._transition(MissionLifecycleState.FAILED, reason=reason)

    def _score_progress(self, obs_t: Tensor, obs_tminus1: Tensor) -> float:
        if self._vlm is None or self._mission is None:
            return 0.0
        # VLMProgressHead returns shape (B, 1) — extract the scalar.
        score_tensor = self._vlm.score(
            obs_tminus1,
            obs_t,
            instruction=self._mission.goal_text,
        )
        return float(score_tensor.view(-1)[0].item())

    async def _handle_stall(self) -> MissionTickResult:
        assert self._mission is not None

        # Always transition through REPLANNING first so the
        # ``mission_state_transitions_total{from_state="replanning",
        # to_state="failed"}`` counter labels match ADR-011 and the
        # state-machine documented in the module docstring. The replan
        # itself may immediately fail (limit exceeded, no replanner
        # wired, LLM returned None, LLM raised) — in every case the
        # subsequent FAILED transition originates from REPLANNING.
        self._transition(MissionLifecycleState.REPLANNING, reason="stalled")
        self._mission.stall_counter = 0

        if self._mission.replan_count >= self._cfg.max_replans_per_mission:
            return self._transition_to_failed(reason="replan_limit_exceeded")

        if self._replanner is None:
            # No replanner wired — replan is impossible, fail the mission.
            return self._transition_to_failed(reason="llm_replan_unavailable")

        try:
            new_goal: GoalVector | None = await self._replanner.submit_replan_request(
                mission_id=self._mission.mission_id,
                goal_text=self._mission.goal_text,
                last_progress=self._mission.last_progress,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning(
                "mission_replan_exception",
                mission_id=self._mission.mission_id,
                error=type(exc).__name__,
            )
            new_goal = None

        if new_goal is None:
            return self._transition_to_failed(reason="llm_replan_unavailable")

        # Success — resume RUNNING with the new goal vector.
        self._mission.last_goal_vector = new_goal
        self._mission.replan_count += 1
        self._transition(MissionLifecycleState.RUNNING, reason="replan_succeeded")
        if self._metrics is not None:
            self._metrics.inc_mission_replan("succeeded")
        return MissionTickResult(
            state=MissionLifecycleState.RUNNING,
            progress=self._mission.last_progress,
            transitioned=True,
            reason="replan_succeeded",
        )

    def _transition_to_failed(self, *, reason: str) -> MissionTickResult:
        """Transition the active mission to FAILED + emit the replan-fail metric.

        Centralises the three stall-path FAILED branches (replan limit
        exceeded, no replanner wired, LLM returned None / raised) so the
        state transition + the ``inc_mission_replan('failed')`` increment
        + the returned :class:`MissionTickResult` stay in lock-step.
        """
        assert self._mission is not None
        self._transition(MissionLifecycleState.FAILED, reason=reason)
        if self._metrics is not None:
            self._metrics.inc_mission_replan("failed")
        return MissionTickResult(
            state=MissionLifecycleState.FAILED,
            progress=self._mission.last_progress,
            transitioned=True,
            reason=reason,
        )

    def _transition(
        self,
        to_state: MissionLifecycleState,
        *,
        reason: str,
    ) -> None:
        assert self._mission is not None
        from_state = self._mission.state
        if from_state == to_state:
            return
        self._mission.state = to_state
        _log.info(
            "mission_state_transition",
            mission_id=self._mission.mission_id,
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
        )
        if self._metrics is not None:
            self._metrics.inc_mission_state_transition(from_state.value, to_state.value)
        if to_state in _TERMINAL_STATES:
            self._record_terminal_duration()

    def _record_terminal_duration(self) -> None:
        assert self._mission is not None
        if self._metrics is None or self._mission.started_at_s is None:
            return
        elapsed = float(self._clock()) - self._mission.started_at_s
        if elapsed < 0:
            return
        self._metrics.observe_mission_active_duration_seconds(elapsed)


__all__ = [
    "MissionLifecycle",
    "MissionLifecycleState",
    "MissionReplannerProtocol",
    "MissionTickResult",
]
