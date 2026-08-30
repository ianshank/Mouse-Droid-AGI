"""MouseDroid orchestrator — voice and face control mixin.

Handles audio output and facial expression control based on state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator._state import _OrchestratorState

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.protocol import ObservationProtocol

_log = get_logger(__name__)


class _VoiceFaceMixin(_OrchestratorState):
    """Voice and face control for the orchestrator."""

    async def _voice_lifecycle(self, event: str) -> None:
        """Fire a lifecycle voice event (startup/shutdown) without an observation.

        Args:
            event: Lifecycle event name (e.g. ``"startup"``, ``"shutdown"``).
        """
        if self._voice_engine is None:
            return
        try:
            await self._voice_engine.speak(event, {"valence": 1.0})
        except Exception:
            _log.warning("voice_lifecycle_failed", voice_event=event, exc_info=True)

    async def _voice_event(
        self,
        event: str,
        observation: ObservationProtocol,
        **extra_context: float,
    ) -> None:
        """Fire a voice event if the voice engine is active.

        Enriches context with sensor data (distance, LiDAR min, audio RMS)
        so the voice engine can modulate speech accordingly.

        Non-blocking: delegates to the engine's async queue.

        Args:
            event: Semantic event name.
            observation: Current sensor observation for context.
            **extra_context: Additional key-value context for the voice engine.
        """
        if self._voice_engine is None:
            return
        context = {"distance_m": float(observation.distance_m)}

        # Enrich with LiDAR minimum distance if features are available
        lidar_features = observation.lidar_features
        if lidar_features is not None and lidar_features.size > 0:
            context["lidar_min_dist_m"] = float(np.min(lidar_features))

        # Enrich with audio level RMS if audio chunk is available
        audio_chunk = observation.audio_chunk
        if audio_chunk is not None and audio_chunk.size > 0:
            context["audio_level_rms"] = float(np.sqrt(np.mean(audio_chunk**2)))

        context.update(extra_context)
        try:
            await self._voice_engine.speak(event, context)
        except Exception:
            _log.warning("voice_event_failed", voice_event=event, exc_info=True)

    async def _voice_observe(
        self,
        observation: ObservationProtocol,
        safety_ctx: SafetyContext,
    ) -> None:
        """Derive voice events from the current observation and safety state.

        Checks safety thresholds from config to avoid hardcoded values.

        Args:
            observation: Current sensor observation.
            safety_ctx: Current safety context.
        """
        if self._voice_engine is None:
            return

        if not safety_ctx.forward_clearance_ok:
            await self._voice_event("obstacle_detected", observation)
        elif safety_ctx.battery_voltage < self._cfg.safety.battery_warn_v:
            await self._voice_event(
                "low_battery",
                observation,
                battery_v=safety_ctx.battery_voltage,
            )
        elif safety_ctx.gpu_temp_c >= self._cfg.safety.gpu_warn_temp_c:
            await self._voice_event(
                "error",
                observation,
                gpu_temp_c=safety_ctx.gpu_temp_c,
            )

    async def _update_face(
        self,
        *,
        safety_ctx: SafetyContext,
        action: torch.Tensor | None,
    ) -> None:
        """Drive the face controller from BDI affect + safety state.

        Pulls ``(valence, arousal)`` from
        :meth:`CognitiveCore.get_latest_affect`, defaulting to neutral when
        the slow loop has not produced a result yet. Idle is inferred from
        ``action`` magnitude — emergency-stop callers pass ``action=None``.

        Args:
            safety_ctx: Current safety context (provides ``is_emergency``).
            action: Most recent commanded action, or ``None`` in the
                emergency path.
        """
        if self._face_controller is None:
            return

        if self._cognitive_core is not None:
            valence, arousal = self._cognitive_core.get_latest_affect()
        else:
            valence, arousal = 0.0, 0.0

        # When emergency wins, is_idle is irrelevant — skip the .item() call
        # to avoid an unnecessary GPU↔CPU sync on the hot path.
        if safety_ctx.is_emergency:
            is_idle = False
        elif action is None:
            is_idle = True
        else:
            epsilon = self._cfg.face_display.idle_action_epsilon if self._cfg.face_display else 1e-3
            is_idle = bool(action.abs().max().item() < epsilon)

        try:
            await self._face_controller.update(
                valence=valence,
                arousal=arousal,
                is_emergency=safety_ctx.is_emergency,
                is_idle=is_idle,
            )
        except Exception as exc:
            _log.warning(
                "face_controller_update_failed",
                exc_type=type(exc).__name__,
                exc_info=True,
            )
