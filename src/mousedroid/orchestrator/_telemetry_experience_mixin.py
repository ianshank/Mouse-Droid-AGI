"""MouseDroid orchestrator — telemetry, experience logging, and curiosity mixin.

Handles frame publishing, experience logging, and curiosity scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mousedroid.common.async_utils import spawn_tracked
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)


class _TelemetryExperienceMixin:
    """Telemetry publishing and experience logging for the orchestrator."""

    async def _publish_telemetry(
        self,
        observation: ObservationProtocol,
        safety_ctx: SafetyContext,
        loop_time_ms: float,
    ) -> None:
        """Build and publish a telemetry frame from current state.

        Non-blocking: silently drops if queue is full.

        Args:
            observation: Current sensor observation bundle.
            safety_ctx: Current safety context.
            loop_time_ms: Control loop iteration time (ms).
        """
        if self._telemetry_publisher is None and self._cloud_sink is None:  # type: ignore[attr-defined]
            return

        try:
            from mousedroid.telemetry.frame_builder import build_telemetry_frame

            frame = build_telemetry_frame(
                observation,
                safety_ctx,
                loop_time_ms,
                self._tick_count,  # type: ignore[attr-defined]
                vision_feature_max_samples=self._cfg.telemetry.vision_feature_max_samples,  # type: ignore[attr-defined]
                liveness_tracker=self._liveness_tracker,  # type: ignore[attr-defined]
                now_s=self._clock.monotonic(),  # type: ignore[attr-defined]
            )
            if self._telemetry_publisher is not None:  # type: ignore[attr-defined]
                await self._telemetry_publisher.publish(frame)
                # PR #4: also publish the latest raw LiDAR scan to the
                # streaming channel when both the publisher and the
                # sensor manager expose them.
                await self._publish_raw_lidar()
            if self._cloud_sink is not None:  # type: ignore[attr-defined]
                await self._cloud_sink.publish_telemetry(frame.to_dict())
        except Exception:
            _log.debug("telemetry_publish_failed", exc_info=True)

    async def _publish_raw_lidar(self) -> None:
        """Publish the latest raw LiDAR scan to the streaming channel.

        No-op when:

        * The publisher does not expose ``publish_lidar_raw`` (legacy
          publishers without raw-LiDAR support).
        * The sensor manager has no ``last_lidar_scan`` (LiDAR not
          configured or no scan yet).

        Exceptions are swallowed and logged at DEBUG so a publisher
        backpressure event never crashes the 30 Hz control loop.
        """
        publish_raw = getattr(self._telemetry_publisher, "publish_lidar_raw", None)  # type: ignore[attr-defined]
        if publish_raw is None:
            return
        scan_source = getattr(self._sensor_manager, "last_lidar_scan", None)  # type: ignore[attr-defined]
        if scan_source is None:
            return
        try:
            from mousedroid.telemetry.protocol import lidar_scan_to_raw

            raw = lidar_scan_to_raw(scan_source)
        except Exception:
            _log.debug("lidar_raw_conversion_failed", exc_info=True)
            return
        try:
            await publish_raw(raw)
        except Exception:
            _log.debug("lidar_raw_publish_failed", exc_info=True)

    def _log_experience(
        self,
        observation: ObservationProtocol,
        action: torch.Tensor,
    ) -> None:
        """Log experience to memory tier and LMDB.

        Builds a ``MouseDroidExperienceRecord`` from the current observation
        and action, then pushes it to episodic replay, working memory, and
        the persistent experience logger.

        Args:
            observation: Current sensor observation.
            action: Action tensor just executed.
        """
        if self._memory_tier is None and self._experience_logger is None:  # type: ignore[attr-defined]
            return

        from mousedroid.experience.record import MouseDroidExperienceRecord

        action_np = action.detach().cpu().numpy().flatten().astype(np.float32)
        record = MouseDroidExperienceRecord(
            vision_features=observation.vision_features,
            distance_m=float(observation.distance_m),
            motor_state=observation.motor_state,
            action=action_np,
        )

        # Compute intrinsic reward for surprise-based prioritization
        surprise = 0.0
        if self._curiosity_module is not None:  # type: ignore[attr-defined]
            with torch.no_grad():
                s = self._h.flatten().unsqueeze(0)  # type: ignore[attr-defined]
                a = action.unsqueeze(0) if action.dim() == 1 else action
                s_next = self._z.flatten().unsqueeze(0)  # type: ignore[attr-defined]
                intrinsic = self._curiosity_module.intrinsic_reward(s, a, s_next)  # type: ignore[attr-defined]
                surprise = float(intrinsic.item())
        record.surprise = surprise

        if self._memory_tier is not None:  # type: ignore[attr-defined]
            min_priority = self._cfg.memory.min_episodic_priority  # type: ignore[attr-defined]
            self._memory_tier.episodic.push(record, priority=max(surprise, min_priority))  # type: ignore[attr-defined]
            latent = self._h.detach().clone()  # type: ignore[attr-defined]
            self._memory_tier.working.push(latent)  # type: ignore[attr-defined]

        if self._experience_logger is not None:  # type: ignore[attr-defined]
            record.reward = surprise
            self._experience_logger.log(record)  # type: ignore[attr-defined]

        if self._cloud_sink is not None:  # type: ignore[attr-defined]
            spawn_tracked(
                self._cloud_publish_tasks,  # type: ignore[attr-defined]
                self._cloud_sink.publish_experience(record),  # type: ignore[attr-defined]
                name="cloud_publish_experience",
            )

    def _compute_curiosity_scores(self) -> dict[str, float]:
        """Compute curiosity channel scores for cognitive core obs_dict.

        Returns:
            Dictionary with 'intrinsic' and 'epistemic' curiosity channels.
        """
        scores: dict[str, float] = {"intrinsic": 0.0, "epistemic": 0.0}

        if self._curiosity_module is not None:  # type: ignore[attr-defined]
            with torch.no_grad():
                s = self._h.flatten().unsqueeze(0)  # type: ignore[attr-defined]
                a = self._prev_action  # type: ignore[attr-defined]
                s_next = self._z.flatten().unsqueeze(0)  # type: ignore[attr-defined]
                intrinsic = self._curiosity_module.intrinsic_reward(s, a, s_next)  # type: ignore[attr-defined]
                scores["intrinsic"] = float(intrinsic.item())

        if self._memory_tier is not None and self._memory_tier.semantic.size > 0:  # type: ignore[attr-defined]
            query = self._h.detach().cpu().numpy().flatten().astype(np.float32)  # type: ignore[attr-defined]
            k = self._cfg.memory.semantic_retrieve_k  # type: ignore[attr-defined]
            results = self._memory_tier.semantic.retrieve(query, k=k)  # type: ignore[attr-defined]
            if results:
                _, distance = results[0]
                scores["epistemic"] = float(distance)

        return scores

    async def _maybe_export_memory(self, *, mission_completed: bool) -> None:
        """Run the OpenClaw MEMORY.md exporter if all three gates pass.

        Gates (any failing gate makes this a no-op):

        1. ``memory_exporter`` was injected (OpenClaw enabled with a
           configured ``shared_memory_path``).
        2. ``memory_tier.episodic`` is non-None (replay buffer exists).
        3. ``mission_completed`` (caller-snapshotted) is ``True`` AND the
           tick count is a multiple of
           ``OpenClawConfig.export_every_n_ticks``.

        The caller is responsible for clearing
        ``mission_just_completed`` exactly once after ALL observers
        (memory exporter, curiosity reset, …) have run; this method no
        longer touches the dispatcher's flag.

        Exceptions are swallowed and logged so a transient filesystem
        failure on the shared path never crashes the control loop.

        Args:
            mission_completed: Snapshot of the dispatcher's
                ``mission_just_completed`` latch taken once per tick.
        """
        if not mission_completed:
            return
        if self._memory_exporter is None or self._memory_tier is None:  # type: ignore[attr-defined]
            return
        if self._memory_export_every_n <= 0:  # type: ignore[attr-defined]
            return
        if self._tick_count % self._memory_export_every_n != 0:  # type: ignore[attr-defined]
            return
        episodic = getattr(self._memory_tier, "episodic", None)  # type: ignore[attr-defined]
        if episodic is None:
            return
        try:
            _log.info("memory_export_started", path_known=True)
            await self._memory_exporter.export(episodic)  # type: ignore[attr-defined]
        except Exception as exc:
            _log.warning(
                "memory_export_hook_failed",
                error=f"{type(exc).__name__}:{exc}",
            )

    def _maybe_reset_curiosity(self, *, mission_completed: bool) -> None:
        """Reset curiosity accumulator at episode boundaries.

        Args:
            mission_completed: Snapshot of the dispatcher's
                ``mission_just_completed`` latch from the tick's
                centralised read so reset fires exactly once per
                mission boundary even when the memory exporter is
                disabled.
        """
        if not mission_completed:
            return
        if self._curiosity_module is None:  # type: ignore[attr-defined]
            return
        self._curiosity_module.reset_episode()  # type: ignore[attr-defined]
        _log.info("curiosity_episode_reset", tick=self._tick_count)  # type: ignore[attr-defined]
