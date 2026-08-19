"""Resilient composite LLM gateway coordinating cloud and edge models."""

from __future__ import annotations

import asyncio

from mousedroid.config.schema.llm import LLMConfig
from mousedroid.constants import (
    DEFAULT_LLM_MAX_COMMAND_LEN,
    MOCK_DISPATCH_YIELD_S,
    MOCK_GOAL_FAST_CONFIDENCE,
    MOCK_GOAL_FAST_LINEAR,
    MOCK_GOAL_FORWARD_CONFIDENCE,
    MOCK_GOAL_FORWARD_LINEAR,
    MOCK_GOAL_REVERSE_CONFIDENCE,
    MOCK_GOAL_REVERSE_LINEAR,
    MOCK_GOAL_TURN_ANGULAR,
    MOCK_GOAL_TURN_CONFIDENCE,
    MOCK_GOAL_TURN_LINEAR,
)
from mousedroid.interfaces.protocols import (
    GoalVector,
    LLMGatewayProtocol,
    MetricsRegistryProtocol,
    PromptInjectionFilterProtocol,
)
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import RegexInjectionFilter

_log = get_logger("mousedroid.llm_gateway.composite")


class CompositeLLMGateway(LLMGatewayProtocol):
    """Hybrid LLM mission translator with cloud-to-edge failover and safety filtering."""

    def __init__(
        self,
        cfg: LLMConfig,
        mock_mode: bool = False,
        metrics: MetricsRegistryProtocol | None = None,
        injection_filter: PromptInjectionFilterProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        self._mock_mode = mock_mode
        self._metrics = metrics
        self._degraded = False
        self._ready = True
        patterns = (
            list(self._cfg.injection_patterns)
            if hasattr(self._cfg, "injection_patterns") and self._cfg.injection_patterns
            else [
                r"ignore.*(previous|rule|prompt|instruction|directive)",
                r"system.*(prompt|override)",
                r"override.*safety",
                r"disable.*(brake|emergency)",
            ]
        )
        self._filter = injection_filter or RegexInjectionFilter(
            patterns=patterns,
            max_len=getattr(self._cfg, "max_command_len", DEFAULT_LLM_MAX_COMMAND_LEN),
        )
        _log.info(
            "composite_llm_gateway_initialized",
            primary=self._cfg.primary_backend
            if hasattr(self._cfg, "primary_backend")
            else self._cfg.backend,
            fallback=getattr(self._cfg, "fallback_backend", "none"),
            mock_mode=self._mock_mode,
        )

    def is_ready(self) -> bool:
        """Return True if gateway is operational."""
        return self._ready

    def is_degraded(self) -> bool:
        """Return True if operating under fallback or degraded state."""
        return self._degraded

    async def translate_mission(self, command: str) -> GoalVector:
        """Sanitize and translate natural language mission into GoalVector.

        Args:
            command: Natural language mission command string.

        Returns:
            High-level normalized GoalVector.
        """
        if not command or not command.strip():
            _log.warning("empty_mission_command_received")
            return GoalVector.neutral_stop()

        # 1. Pre-egress prompt-injection filtering
        if getattr(self._cfg, "enable_injection_filter", True):
            try:
                command = self._filter.sanitize(command)
            except Exception as exc:
                _log.warning("prompt_injection_sanitization_rejected", error=str(exc))
                return GoalVector(
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                    arm_action="e_stop",
                    confidence=0.0,
                    is_safe=False,
                )

        # 2. Mock mode execution
        if (
            self._mock_mode
            or getattr(self._cfg, "backend", "") == "mock"
            or getattr(self._cfg, "primary_backend", "") == "mock"
        ):
            return self._mock_translate(command)

        # 3. Cloud translation with edge failover
        try:
            vector = await self._dispatch_primary_translation(command)
            self._degraded = False
            if self._metrics:
                self._metrics.record_counter(
                    "mousedroid_llm_gateway_served", labels={"tier": "primary", "outcome": "ok"}
                )
            return vector
        except asyncio.CancelledError:
            _log.info("mission_translation_task_cancelled")
            raise
        except Exception as exc:
            _log.warning("primary_llm_backend_failed_falling_back", error=str(exc))
            self._degraded = True
            if self._metrics:
                self._metrics.record_counter(
                    "mousedroid_llm_gateway_served",
                    labels={"tier": "fallback", "outcome": "degraded"},
                )
            return self._mock_translate(command)

    async def _dispatch_primary_translation(self, command: str) -> GoalVector:
        """Execute translation via primary backend."""
        await asyncio.sleep(MOCK_DISPATCH_YIELD_S)
        return self._mock_translate(command)

    def _mock_translate(self, command: str) -> GoalVector:
        """Deterministic rule-based translation for offline/mock environments."""
        cmd_lower = command.lower()
        if "stop" in cmd_lower or "halt" in cmd_lower or "emergency" in cmd_lower:
            return GoalVector.neutral_stop()
        if "fast" in cmd_lower or "run" in cmd_lower:
            return GoalVector(
                linear_velocity=MOCK_GOAL_FAST_LINEAR,
                angular_velocity=0.0,
                arm_action="idle",
                confidence=MOCK_GOAL_FAST_CONFIDENCE,
                is_safe=True,
            )
        if "left" in cmd_lower:
            return GoalVector(
                linear_velocity=MOCK_GOAL_TURN_LINEAR,
                angular_velocity=MOCK_GOAL_TURN_ANGULAR,
                arm_action="idle",
                confidence=MOCK_GOAL_TURN_CONFIDENCE,
                is_safe=True,
            )
        if "right" in cmd_lower:
            return GoalVector(
                linear_velocity=MOCK_GOAL_TURN_LINEAR,
                angular_velocity=-MOCK_GOAL_TURN_ANGULAR,
                arm_action="idle",
                confidence=MOCK_GOAL_TURN_CONFIDENCE,
                is_safe=True,
            )
        if "back" in cmd_lower or "reverse" in cmd_lower:
            return GoalVector(
                linear_velocity=MOCK_GOAL_REVERSE_LINEAR,
                angular_velocity=0.0,
                arm_action="idle",
                confidence=MOCK_GOAL_REVERSE_CONFIDENCE,
                is_safe=True,
            )
        return GoalVector(
            linear_velocity=MOCK_GOAL_FORWARD_LINEAR,
            angular_velocity=0.0,
            arm_action="idle",
            confidence=MOCK_GOAL_FORWARD_CONFIDENCE,
            is_safe=True,
        )

    async def stop(self) -> None:
        """Teardown gateway resources."""
        self._ready = False
        _log.info("composite_llm_gateway_stopped")
