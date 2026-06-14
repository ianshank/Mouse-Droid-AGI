"""Geometric safety action projector (Tier C2 / C2.1).

Implements :class:`SafetyActionProjectorProtocol` as a stateless geometric
constraint projection — pure function of the frozen
:class:`~mousedroid.safety.context.SafetyContext` plus the proposed
action. Clamps three families of constraints:

1. **Forward-velocity clamp** — ``forward_clearance_ok=False`` or
   ``lidar_min_dist_m < lidar_brake_distance_m`` clamps the forward
   velocity component (index 0) to ``crawl_velocity_mps`` (sign-preserving:
   reverse motion is never blocked by an obstacle ahead).
2. **Human-proximity clamp** — ``human_detected and human_dist_m <
   human_keepout_m`` caps the magnitude of every action component to
   ``human_proximity_speed_mps``.
3. **Rotational clamp** — ``lidar_min_dist_m < tight_quarters_dist_m`` caps
   the angular-velocity magnitude (index 2 when present) to
   ``tight_quarters_omega_max_rads``.

All thresholds come from :class:`SafetyProjectorConfig`; nothing is
hardcoded. The projector NEVER mutates its input — callers may keep
references to the original action tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.config.schema import SafetyProjectorConfig
    from mousedroid.safety.context import SafetyContext
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

_REASON_FORWARD_VELOCITY = "forward_velocity"
_REASON_HUMAN_PROXIMITY = "human_proximity"
_REASON_TIGHT_QUARTERS = "tight_quarters"

# Action-vector index conventions. The mouse-droid action layout is
# ``[vx, vy, omega]`` per :attr:`ModelConfig.action_dim`. Index 0 is the
# forward velocity; index 2 is the angular velocity when present.
_VX_INDEX = 0
_OMEGA_INDEX = 2


class GeometricSafetyProjector:
    """Stateless geometric clamp implementing the projector protocol.

    The projector is fully deterministic and CPU-only. Operators can drop
    it into the orchestrator tick at the seam right after
    ``_select_action`` returns — exactly one place, regardless of which
    internal policy branch produced the action.
    """

    def __init__(
        self,
        cfg: SafetyProjectorConfig,
        *,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Build the projector.

        Args:
            cfg: Projector thresholds. Every clamp limit comes from here.
            metrics: Optional shared metrics registry. When supplied, the
                projector increments ``mousedroid_safety_action_clamps_total``
                with one of the reason labels above on every materially
                different clamp.
        """
        self._cfg = cfg
        self._metrics = metrics
        _log.info(
            "safety_projector_init",
            lidar_brake_distance_m=cfg.lidar_brake_distance_m,
            crawl_velocity_mps=cfg.crawl_velocity_mps,
            human_keepout_m=cfg.human_keepout_m,
            human_proximity_speed_mps=cfg.human_proximity_speed_mps,
            tight_quarters_dist_m=cfg.tight_quarters_dist_m,
            tight_quarters_omega_max_rads=cfg.tight_quarters_omega_max_rads,
        )

    def project(
        self,
        action: NDArray[np.float32],
        safety_ctx: SafetyContext,
    ) -> NDArray[np.float32]:
        """Return a clamped copy of ``action``.

        Args:
            action: Proposed action vector. Shape is policy-defined; index
                ``0`` is the forward velocity; index ``2`` (when present)
                is angular velocity.
            safety_ctx: Frozen safety context produced by the safety
                monitor for this tick.

        Returns:
            A new ``np.float32`` array with the same shape as ``action``.
            Returns the unchanged action (still a copy) when no clamping
            rule fires.
        """
        cfg = self._cfg
        clamped = np.asarray(action, dtype=np.float32).copy()
        reasons: list[str] = []

        # Forward-velocity clamp. Only clamps positive forward motion —
        # reversing away from an obstacle is always permitted.
        if clamped.size > _VX_INDEX:
            should_brake = (
                not safety_ctx.forward_clearance_ok
                or safety_ctx.lidar_min_dist_m < cfg.lidar_brake_distance_m
            )
            if should_brake and clamped[_VX_INDEX] > cfg.crawl_velocity_mps:
                clamped[_VX_INDEX] = np.float32(cfg.crawl_velocity_mps)
                reasons.append(_REASON_FORWARD_VELOCITY)

        # Human-proximity clamp. Magnitude cap is applied to every
        # component so a lateral pivot toward a human is also dampened.
        if safety_ctx.human_detected and safety_ctx.human_dist_m < cfg.human_keepout_m:
            cap = np.float32(cfg.human_proximity_speed_mps)
            if np.any(np.abs(clamped) > cap):
                # ``np.sign`` / ``np.minimum`` of float32 inputs preserve
                # dtype, so no explicit ``.astype(np.float32)`` cast is
                # required here (the return-site cast at the bottom of
                # this method covers the ``Any`` numpy stubs return).
                clamped = np.sign(clamped) * np.minimum(np.abs(clamped), cap)
                reasons.append(_REASON_HUMAN_PROXIMITY)

        # Rotational clamp in tight quarters. Caps |omega| only — leaves
        # vx/vy alone so the rover can keep crawling forward.
        if clamped.size > _OMEGA_INDEX and safety_ctx.lidar_min_dist_m < cfg.tight_quarters_dist_m:
            omega_cap = np.float32(cfg.tight_quarters_omega_max_rads)
            if abs(clamped[_OMEGA_INDEX]) > omega_cap:
                clamped[_OMEGA_INDEX] = np.float32(np.sign(clamped[_OMEGA_INDEX]) * omega_cap)
                reasons.append(_REASON_TIGHT_QUARTERS)

        if reasons:
            _log.info(
                "safety_action_clamped",
                reasons=tuple(reasons),
                lidar_min_dist_m=safety_ctx.lidar_min_dist_m,
                human_detected=safety_ctx.human_detected,
                human_dist_m=safety_ctx.human_dist_m,
                forward_clearance_ok=safety_ctx.forward_clearance_ok,
            )
            if self._metrics is not None:
                for reason in reasons:
                    self._metrics.inc_safety_action_clamp(reason)

        # Explicit cast keeps mypy --strict happy: ``np.asarray(...)`` /
        # ``np.sign(...) * np.minimum(...)`` return ``Any`` per numpy's
        # current stubs, even though the runtime values are guaranteed
        # ``ndarray[..., np.float32]`` by the ``.astype(np.float32)`` /
        # ``np.float32(...)`` wrappers above. Casting at the return site
        # documents the invariant + lets ``mypy --strict`` pass without
        # a module-wide inline type suppression.
        result: NDArray[np.float32] = clamped
        return result


__all__ = ["GeometricSafetyProjector"]
