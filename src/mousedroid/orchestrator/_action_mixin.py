"""MouseDroid orchestrator — action selection and execution mixin.

Handles VLA/cognitive policy dispatch, action validation, and safety projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.common.actions import normalize_action_numpy
from mousedroid.constants import DEFAULT_BATTERY_VOLTAGE, MOTOR_STATE_BATTERY_INDEX
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator._state import _OrchestratorState

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)


class _ActionMixin(_OrchestratorState):
    """Action selection and execution for the orchestrator."""

    def _select_action(
        self,
        safety_ctx: SafetyContext,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor:
        """Select action using cognitive core (primary) or MCTS agent (fallback).

        Args:
            safety_ctx: Current safety context.
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor.
        """
        if self._cognitive_core is not None:
            action = self._try_cognitive_action(observation, loop_time_ms)
            if action is not None:
                return action

        # VLA branch (Phase 3a). Default policy_selector='nav_agent'
        # short-circuits this and preserves byte-identical pre-Phase-3a
        # behavior. Only active when the orchestrator was wired with a
        # VLA policy AND the config selects it.
        selector = self._cfg.loop.policy_selector
        if selector != "nav_agent" and self._vla_policy is not None:
            vla_action = self._try_vla_action(observation)
            if vla_action is not None:
                return vla_action
            # Fallthrough to nav_agent below in 'auto' mode, or to a
            # zero-action safe stop in strict 'vla' mode.
            if selector == "vla" and not self._cfg.vla.fallback_on_timeout:
                _log.warning("vla_timeout_safe_stop", selector=selector)
                return torch.zeros(int(self._cfg.model.action_dim), dtype=torch.float32)

        return self._agents[0].act(self._h, self._z, safety_ctx)

    def _maybe_project_action(
        self,
        action: torch.Tensor,
        safety_ctx: SafetyContext,
    ) -> torch.Tensor:
        """Apply the optional geometric safety projector to ``action``.

        Wraps the four return sites of :meth:`_select_action` at a single
        seam in :meth:`tick`. The projector is a soft constraint applied
        AFTER the policy returns — it is the complement of the hard E-stop
        short-circuit at the top of :meth:`tick`.

        When ``self._safety_projector is None`` (the default, gated by
        ``cfg.safety.projector.enabled=False``) this method is a pure
        identity pass-through so existing deployments produce
        byte-identical actions to pre-C2.

        Args:
            action: Proposed action from :meth:`_select_action`. Shape is
                ``(action_dim,)`` or ``(1, action_dim)`` depending on the
                upstream policy branch.
            safety_ctx: Frozen safety context for the current tick.

        Returns:
            Either the original ``action`` (when no projector is wired)
            or a clamped copy with the same shape and dtype.
        """
        if self._safety_projector is None:
            return action

        was_unbatched = action.dim() == 1
        flat: torch.Tensor = action if was_unbatched else action.squeeze(0)
        action_np = flat.detach().cpu().numpy().astype(np.float32, copy=False)
        # ``project()`` is typed to return ``NDArray[np.float32]`` and the
        # implementation guarantees that dtype, so no defensive
        # ``np.asarray(..., dtype=np.float32)`` cast is required here.
        projected_np = self._safety_projector.project(action_np, safety_ctx)
        projected = torch.from_numpy(projected_np).to(flat.device)
        if not was_unbatched:
            projected = projected.unsqueeze(0)
        return projected

    def _try_vla_action(
        self,
        observation: ObservationProtocol,
    ) -> torch.Tensor | None:
        """Run a single VLA inference under the per-tick latency budget.

        Returns ``None`` when the policy raises, when inference exceeds
        ``cfg.loop.inference_timeout_s`` (defaulting to
        ``1.0 / cfg.loop.control_hz``), or when the resulting action
        tensor has the wrong shape. The orchestrator decides how to
        respond to ``None`` (fall back to nav_agent or emit a safe
        stop) via ``policy_selector``/``vla.fallback_on_timeout``.

        Args:
            observation: Current sensor observation (unused by MockVLA;
                forwarded for richer policies).

        Returns:
            The VLA action tensor on success, otherwise ``None``.
        """
        del observation  # forwarded via VLAObservation below
        assert self._vla_policy is not None  # narrowed by caller

        from mousedroid.vla.policy import VLAObservation

        budget = self._cfg.loop.inference_timeout_s
        if budget is None:
            budget = 1.0 / float(self._cfg.loop.control_hz)

        start = self._clock.monotonic()
        try:
            with torch.no_grad():
                result = self._vla_policy.predict(VLAObservation(h=self._h, z=self._z))
        except Exception as exc:  # never let VLA crash the loop
            # Surface the exception type so dashboards can distinguish
            # CUDA-OOM from logic errors at a glance (Gemini review).
            self._failure_recorder.record(
                "orchestrator",
                "vla_exception",
                level="warning",
                extra={"error": type(exc).__name__},
            )
            _log.warning("vla_predict_failed", policy=self._vla_policy.name, exc_info=True)
            return None

        elapsed = self._clock.monotonic() - start
        if elapsed > budget:
            self._failure_recorder.record(
                "orchestrator",
                "vla_timeout",
                level="warning",
                extra={"elapsed_s": round(elapsed, 4), "budget_s": round(budget, 4)},
            )
            if self._metrics is not None:
                # Cast is safe: the policy_selector gate at the caller
                # guarantees ``self._vla_policy is not None``, which means
                # ``cfg.vla.backend != "none"``. The runtime value belongs
                # to :data:`VLAActiveBackendLiteral` by construction; mypy
                # can't see the upstream gate so we cast explicitly.
                from mousedroid.config.schema import VLAActiveBackendLiteral

                self._metrics.inc_vla_timeout(cast(VLAActiveBackendLiteral, self._cfg.vla.backend))
            _log.warning(
                "vla_inference_timeout",
                policy=self._vla_policy.name,
                elapsed_s=elapsed,
                budget_s=budget,
                mode=self._cfg.vla.backend,
            )
            return None

        action_dim = int(self._cfg.model.action_dim)
        if result.action.shape != (action_dim,):
            # Record the full tensor shape (stringified for Prometheus
            # label-friendliness) so dashboards distinguish 0-D outputs
            # from rank-2 outputs like ``(1, action_dim)``.
            self._failure_recorder.record(
                "orchestrator",
                "vla_wrong_shape",
                level="warning",
                extra={
                    "expected": str((action_dim,)),
                    "got": str(tuple(result.action.shape)),
                },
            )
            _log.warning(
                "vla_action_shape_mismatch",
                policy=self._vla_policy.name,
                expected=(action_dim,),
                got=tuple(result.action.shape),
            )
            return None

        return result.action

    def _try_cognitive_action(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor | None:
        """Attempt action selection via cognitive core.

        Args:
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor if successful, None on failure.
        """
        try:
            battery_v = (
                float(observation.motor_state[MOTOR_STATE_BATTERY_INDEX])
                if observation.motor_state.size > MOTOR_STATE_BATTERY_INDEX
                else DEFAULT_BATTERY_VOLTAGE
            )
            belief_dim = int(self._cfg.model.belief_dim)
            bdi_state_vec = self._h.numpy().flatten().astype(np.float32, copy=False)
            state_vec: NDArray[np.float32] = bdi_state_vec
            if state_vec.size < belief_dim:
                state_vec = np.pad(state_vec, (0, belief_dim - state_vec.size))
            else:
                state_vec = state_vec[:belief_dim]
            state_vec = state_vec.astype(np.float32, copy=False)

            obs_dict: dict[str, object] = {
                "state": state_vec,
                "bdi_state": bdi_state_vec,
                "battery_v": battery_v,
                "obstacle_dist_m": float(observation.distance_m),
                "mcts_sims": int(self._cfg.mcts.n_simulations_base),
                "loop_time_ms": loop_time_ms,
                "curiosity": self._compute_curiosity_scores(),
            }
            cognitive_core = self._cognitive_core
            assert cognitive_core is not None
            action_np, violations = cognitive_core.tick_fast(obs_dict)
            if violations:
                _log.info(
                    "orchestrator_constitutional_violations_summary",
                    violation_count=len(violations),
                    violations=violations,
                )

            return self._normalize_cognitive_action(action_np)
        except Exception as e:
            # Surface the exception type so dashboards can distinguish
            # the failure mode (Gemini review).
            self._failure_recorder.record(
                "orchestrator",
                "cognitive_core_exception",
                level="warning",
                extra={"error": type(e).__name__},
            )
            _log.warning(
                "cognitive_core_action_selection_failed",
                error=str(e),
                falling_back_to_mcts=True,
            )
            return None

    def _normalize_cognitive_action(
        self,
        action_np: NDArray[np.float32] | NDArray[np.float64],
    ) -> torch.Tensor:
        """Normalize cognitive core action to match expected action_dim.

        Args:
            action_np: Raw action from cognitive core.

        Returns:
            Normalized 1-D torch tensor with correct dimensions.
        """
        return normalize_action_numpy(action_np, int(self._cfg.model.action_dim))

    def _project_action_to_executable_axes(self, action: torch.Tensor) -> torch.Tensor:
        """Zero action components the configured chassis cannot execute.

        A skid-steer command set (``waveshare_stock``) has no lateral axis, so
        the driver drops ``vy``. Zeroing it HERE — before the action is
        executed, stored as ``ctx.executed_action`` and written to the
        experience log — keeps the recorded action equal to the action the
        wheels actually performed. Without this the world model and any
        replay-trained policy are fit on a physically inert ``action[1]``,
        and the discrepancy is invisible downstream because an encoder-less
        chassis reports no lateral motion to contradict it.

        Returns the tensor unchanged (no copy) when every axis is
        executable, so the default legacy path is allocation-identical.

        Args:
            action: Normalised action tensor with values in ``[-1, 1]``.

        Returns:
            The action restricted to executable axes.
        """
        if self._supports_lateral or action.shape[0] <= 1:
            return action
        if float(action[1]) == 0.0:
            return action
        projected = action.clone()
        projected[1] = 0.0
        return projected

    async def _execute_action(self, action: torch.Tensor) -> None:
        """Scale and send action to ESP32 motors.

        Args:
            action: Action tensor with values in [-1, 1]; assumed already
                restricted to executable axes by
                :meth:`_project_action_to_executable_axes`.
        """
        max_v = self._cfg.esp32.max_velocity_mps
        max_omega = self._cfg.esp32.max_omega_rads
        vx = float(action[0]) * max_v
        vy = float(action[1]) * max_v if action.shape[0] > 1 else 0.0
        omega = float(action[2]) * max_omega if action.shape[0] > 2 else 0.0
        await self._esp32.send_velocity(vx, vy, omega)
