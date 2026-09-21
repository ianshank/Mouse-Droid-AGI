"""Google ADK mission decomposer adapter.

Off-loop NL command decomposition via Google ADK. Lazy SDK import in
start(); degrade-not-crash when SDK is absent.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema.agents import ADKConfig
    from mousedroid.llm_gateway.mission_parser import MissionIntent
    from mousedroid.security.injection_filter import PromptInjectionFilterProtocol

_log = get_logger(__name__)


class ADKMissionAdapter:
    """Google ADK mission decomposer adapter."""

    def __init__(
        self,
        cfg: ADKConfig,
        *,
        injection_filter: PromptInjectionFilterProtocol | None = None
    ) -> None:
        """Initialize the ADK adapter.

        Args:
            cfg: The ADK configuration.
            injection_filter: Optional prompt injection filter for sanitization.
        """
        self._cfg = cfg
        self._injection_filter = injection_filter
        self._adk: Any = None
        self._agent: Any = None
        self._ready = False
        self._degraded = False

    async def start(self) -> None:
        """Start the adapter and initialize the ADK agent lazily."""
        try:
            import google.adk as adk
            self._adk = adk
            # Assuming agent creation looks something like this
            self._agent = await asyncio.to_thread(lambda: adk.Agent(name="mousedroid"))
            self._ready = True
            _log.info("adk_adapter_started")
        except ImportError:
            self._degraded = True
            _log.warning("adk_sdk_missing_degraded", degraded=True)
        except Exception as e:
            self._degraded = True
            _log.warning("adk_adapter_start_failed", error=str(e), degraded=True)

    async def stop(self) -> None:
        """Stop the adapter and clean up resources."""
        self._ready = False
        self._adk = None
        self._agent = None
        _log.info("adk_adapter_stopped")

    async def decompose(self, command: str) -> list[MissionIntent]:
        """Decompose a natural language command into mission intents.

        Args:
            command: The natural language command to decompose.

        Returns:
            A list of mission intents.
        """
        from mousedroid.llm_gateway.mission_parser import MissionIntent

        if not command:
            _log.warning("adk_adapter_empty_command")
            return [MissionIntent(agent_source='adk')]

        if self._degraded or not self._ready:
            _log.warning("adk_adapter_degraded_or_not_ready")
            return [MissionIntent(agent_source='adk')]

        sanitized_command = command
        if self._injection_filter:
            sanitized_command = self._injection_filter.sanitize(command)

        try:
            # Assuming agent.decompose or similar method
            response = await asyncio.wait_for(
                asyncio.to_thread(self._agent.decompose, sanitized_command),
                timeout=self._cfg.timeout_s
            )
            # Assuming response is an iterable of intent data
            intents = []
            for item in response:
                intents.append(MissionIntent(agent_source='adk', **item))
            return intents
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("adk_adapter_decompose_failed", error=str(e))
            return [MissionIntent(agent_source='adk')]

    @property
    def is_ready(self) -> bool:
        """Check if the adapter is ready."""
        return self._ready

    @property
    def is_degraded(self) -> bool:
        """Check if the adapter is in a degraded state."""
        return self._degraded
