"""MouseDroid orchestrator — mission processing mixin.

Handles natural language mission acceptance and lifecycle coordination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.llm_gateway.protocol import GoalVector
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)


class _MissionMixin:
    """Mission processing and lifecycle management for the orchestrator."""

    async def process_mission(self, nl_command: str) -> GoalVector:
        """Process a natural language mission command.

        Uses a fallback chain: rule-based parser first (< 1ms), then
        LLM gateway for unknown/ambiguous commands when available.

        Args:
            nl_command: Natural language mission command.

        Returns:
            GoalVector with velocity targets in [-1, 1].
        """
        from mousedroid.llm_gateway.mission_parser import IntentType
        from mousedroid.llm_gateway.protocol import GoalVector

        if not nl_command or not nl_command.strip():
            _log.debug("process_mission_empty_command")
            return GoalVector()

        # Stage 1: Rule-based parser (fast path, < 1ms)
        if self._mission_parser is not None:  # type: ignore[attr-defined]
            intent = self._mission_parser.parse(nl_command)
            threshold = self._cfg.mission_parser.llm_fallback_confidence  # type: ignore[attr-defined]
            if intent.confidence >= threshold and intent.intent_type != IntentType.UNKNOWN:
                _log.info(
                    "mission_parsed_rule_based",
                    command=nl_command,
                    intent=intent.intent_type.value,
                    confidence=intent.confidence,
                )
                # Defer lifecycle ``start_mission`` until AFTER a parser
                # accepts (Copilot MED). Starting the lifecycle on every
                # NL command — even unrecognized ones — leaves the
                # state machine RUNNING on the neutral fallback
                # ``GoalVector()`` path below and lets it stall/fail
                # missions that were never actually accepted.
                self._start_mission_lifecycle_if_wired(nl_command)
                return intent.goal_vector

        # Stage 2: LLM fallback (slow path, ~100-500ms)
        if self._llm_gateway is not None:  # type: ignore[attr-defined]
            try:
                goal = await self._llm_gateway.translate_mission(nl_command)  # type: ignore[attr-defined]
                _log.info(
                    "mission_parsed_llm",
                    command=nl_command,
                    vx=goal.vx_target,
                    vy=goal.vy_target,
                    omega=goal.omega_target,
                )
                # See Stage-1 note above: start the lifecycle only once
                # the LLM has produced an accepted goal.
                self._start_mission_lifecycle_if_wired(nl_command)
                return goal
            except Exception:
                _log.warning("mission_llm_fallback_failed", exc_info=True)

        # Stage 3: Fallback to zero (safe default). The lifecycle is
        # NOT started here — an unrecognized command must not leave the
        # state machine RUNNING on the neutral fallback goal.
        _log.warning("mission_unresolved", command=nl_command)
        return GoalVector()

    def _start_mission_lifecycle_if_wired(self, nl_command: str) -> None:
        """Begin a MissionLifecycle mission iff one is wired.

        Centralises the bump-counter + build-id + ``start_mission``
        sequence used from both accepted Stage-1 (parser) and Stage-2
        (LLM) paths. Failures are logged and swallowed so a misbehaving
        lifecycle never crashes the mission-acceptance hot path.
        """
        if self._mission_lifecycle is None:  # type: ignore[attr-defined]
            return
        self._mission_seq += 1  # type: ignore[attr-defined]
        mission_id = f"mission-{self._mission_seq:06d}"  # type: ignore[attr-defined]
        try:
            self._mission_lifecycle.start_mission(  # type: ignore[attr-defined]
                mission_id=mission_id,
                goal_text=nl_command,
            )
        except Exception:  # pragma: no cover - defensive
            _log.warning("mission_lifecycle_start_failed", exc_info=True)

    async def _maybe_tick_mission_lifecycle(
        self,
        observation: ObservationProtocol,
    ) -> None:
        """Tier C2.1 — drive the optional MissionLifecycle once per tick.

        Caches ``observation.vision_features`` between ticks so the
        lifecycle's ``(obs_t, obs_tminus1)`` contract is honoured. No-op
        when no lifecycle is wired or the observation has a zero-length
        vision-feature vector. Failures in the lifecycle never propagate
        into the control loop — they're logged and the orchestrator tick
        continues.
        """
        import torch

        if self._mission_lifecycle is None:  # type: ignore[attr-defined]
            return
        vf = observation.vision_features
        # ObservationProtocol.vision_features is typed NDArray[np.float32]
        # (never None per src/mousedroid/sensing/protocol.py). The only
        # degenerate case is the zero-length / zero-d fallback array that
        # mock_hardware emits before the camera warms up. Invalidate the
        # cached previous frame so the next non-empty observation does
        # NOT get paired with a stale pre-dropout frame — that would
        # silently violate the lifecycle's ``(obs_t, obs_tminus1)``
        # adjacency contract and corrupt VLM progress scoring.
        if vf.size == 0:
            self._prev_obs_for_vlm = None  # type: ignore[attr-defined]
            return
        # ``torch.tensor`` performs a single copy and tolerates a non-
        # contiguous ``vf`` (camera/sensor manager may recycle the
        # underlying numpy buffer between ticks — Jetson ring-buffer
        # pattern, see CLAUDE.md deque(maxlen=N) convention). The owned
        # copy means a subsequent ``shared_buffer[:] = ...`` mutation
        # never aliases the cached prev-frame. Explicit ``dtype`` is
        # required even though ``vf`` is already ``np.float32`` because
        # the sensor protocol pins the numpy dtype, not the torch dtype,
        # so a future widening of the protocol shouldn't silently change
        # the VLM input dtype.
        obs_t = torch.tensor(vf, dtype=torch.float32).unsqueeze(0)
        prev = self._prev_obs_for_vlm  # type: ignore[attr-defined]
        self._prev_obs_for_vlm = obs_t  # type: ignore[attr-defined]
        if prev is None:
            return
        # Local rebind so mypy --strict sees Tensor (not Tensor | None)
        # at the tick() call site.
        prev_t: torch.Tensor = prev
        try:
            await self._mission_lifecycle.tick(obs_t, prev_t)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            _log.warning("mission_lifecycle_tick_failed", exc_info=True)
