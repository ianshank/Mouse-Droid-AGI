"""Tier C2.3: LLM-backed mission replanner adapter.

Implements :class:`MissionReplannerProtocol` on top of any
:class:`LLMGatewayProtocol`-conforming gateway (in-process llama-cpp
or HTTP-backed Ollama/LM Studio/OpenAI). Built by
:func:`mousedroid.factory.build_mission_replanner` when
``cfg.mission.llm_replanner_enabled`` is True.

Pure adapter — no state beyond injected dependencies. Every public path
emits a structured log event and increments
``mission_replan_llm_calls_total{outcome=...}`` so operators monitor
replan health out-of-band.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MissionReplannerConfig
    from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class LLMGatewayMissionReplanner:
    """Adapter from :class:`LLMGatewayProtocol` to ``MissionReplannerProtocol``.

    Conforms structurally to the protocol defined at
    :class:`mousedroid.orchestrator.mission_lifecycle.MissionReplannerProtocol`.
    Returns ``None`` whenever the gateway is degraded or raises so the
    lifecycle falls back to its safe ``llm_replan_unavailable`` failure
    path — never partial state.
    """

    def __init__(
        self,
        *,
        gateway: LLMGatewayProtocol,
        cfg: MissionReplannerConfig,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._gateway = gateway
        self._cfg = cfg
        self._metrics = metrics

    async def submit_replan_request(
        self,
        *,
        mission_id: str,
        goal_text: str,
        last_progress: float,
    ) -> GoalVector | None:
        """Request a replan from the LLM gateway."""
        if not self._gateway.is_ready:
            _log.warning(
                "mission_replan_gateway_degraded",
                mission_id=mission_id,
                last_progress=last_progress,
            )
            if self._metrics is not None:
                self._metrics.inc_mission_replan_llm("degraded")
            return None

        prompt = self._build_prompt(goal_text=goal_text, last_progress=last_progress)
        try:
            goal = await self._gateway.translate_mission(prompt)
        except Exception as exc:
            _log.warning(
                "mission_replan_llm_exception",
                mission_id=mission_id,
                error=f"{type(exc).__name__}:{exc}",
            )
            if self._metrics is not None:
                self._metrics.inc_mission_replan_llm("exception")
            return None

        _log.info(
            "mission_replan_llm_ok",
            mission_id=mission_id,
            last_progress=last_progress,
            vx=goal.vx_target,
            vy=goal.vy_target,
            omega=goal.omega_target,
        )
        if self._metrics is not None:
            self._metrics.inc_mission_replan_llm("ok")
        return goal

    def _build_prompt(self, *, goal_text: str, last_progress: float) -> str:
        """Construct the augmented prompt, clipped to ``max_prompt_chars``."""
        if self._cfg.include_progress_in_prompt:
            prompt = f"{goal_text} (last_progress={last_progress:.2f})"
        else:
            prompt = goal_text
        return prompt[: self._cfg.max_prompt_chars]


__all__ = ["LLMGatewayMissionReplanner"]
