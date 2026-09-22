"""Google ADK mission decomposer adapter.

Off-loop NL command decomposition via Google ADK. Lazy SDK import in
start(); degrade-not-crash when SDK is absent.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
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
        self, cfg: ADKConfig, *, injection_filter: PromptInjectionFilterProtocol | None = None
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
        self._adk = None
        self._agent = None
        self._ready = False
        self._degraded = False
        try:
            import importlib

            adk = importlib.import_module("google.adk")
            agent_factory = getattr(adk, "Agent", None)
            if adk is None or agent_factory is None:
                raise ImportError("google.adk.Agent unavailable")
            self._adk = adk
            model_name = self._cfg.model_name
            self._agent = await asyncio.to_thread(
                lambda: agent_factory(name="mousedroid", model=model_name)
            )
            if self._agent is None:
                raise RuntimeError("google.adk.Agent returned no agent")
            self._ready = True
            self._degraded = False
            _log.info("adk_adapter_started", model=model_name)
        except (ImportError, ModuleNotFoundError):
            self._ready = False
            self._adk = None
            self._agent = None
            self._degraded = True
            _log.warning("adk_sdk_missing_degraded", degraded=True)
        except Exception as e:
            self._ready = False
            self._adk = None
            self._agent = None
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
        from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent

        if not command:
            _log.warning("adk_adapter_empty_command")
            return [MissionIntent(agent_source="adk")]

        if self._degraded or not self._ready:
            _log.warning("adk_adapter_degraded_or_not_ready")
            return [MissionIntent(agent_source="adk", raw_command=command)]

        sanitized_command = command
        if self._injection_filter:
            sanitized_command = self._injection_filter.sanitize(command)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._decompose_with_config,
                    sanitized_command,
                ),
                timeout=self._cfg.timeout_s,
            )
            intents: list[MissionIntent] = []
            valid_fields = {
                "intent_type",
                "goal_vector",
                "confidence",
                "raw_command",
                "parameters",
                "sub_tasks",
            }
            for item in list(response)[: self._cfg.max_sub_tasks]:
                if isinstance(item, dict):
                    filtered = {k: v for k, v in item.items() if k in valid_fields}
                    if "raw_command" not in filtered or not filtered["raw_command"]:
                        filtered["raw_command"] = command
                    if "intent_type" in filtered and isinstance(filtered["intent_type"], str):
                        try:
                            filtered["intent_type"] = IntentType(filtered["intent_type"])
                        except ValueError:
                            filtered["intent_type"] = IntentType.UNKNOWN
                    if "sub_tasks" in filtered and isinstance(filtered["sub_tasks"], (list, tuple)):
                        filtered["sub_tasks"] = tuple(str(st) for st in filtered["sub_tasks"])
                    intents.append(MissionIntent(agent_source="adk", **filtered))
                elif isinstance(item, MissionIntent):
                    intents.append(item)
                else:
                    intents.append(
                        MissionIntent(
                            agent_source="adk",
                            raw_command=command,
                            sub_tasks=(str(item),),
                        )
                    )
            return intents if intents else [MissionIntent(agent_source="adk", raw_command=command)]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("adk_adapter_decompose_failed", error=str(e))
            return [MissionIntent(agent_source="adk", raw_command=command)]

    @property
    def is_ready(self) -> bool:
        """Check if the adapter is ready."""
        return self._ready

    @property
    def is_degraded(self) -> bool:
        """Check if the adapter is in a degraded state."""
        return self._degraded

    def _decompose_with_config(self, command: str) -> Any:
        """Call the ADK agent with supported configuration kwargs."""
        if self._agent is None:
            raise RuntimeError("ADK agent is not initialized")

        decompose = self._agent.decompose
        kwargs: dict[str, Any] = {"max_sub_tasks": self._cfg.max_sub_tasks}

        params: Mapping[str, inspect.Parameter]
        try:
            params = inspect.signature(decompose).parameters
        except (TypeError, ValueError):
            params = {}

        supports_var_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

        if supports_var_kwargs or "model_name" in params:
            kwargs["model_name"] = self._cfg.model_name
        elif "model" in params:
            kwargs["model"] = self._cfg.model_name

        return decompose(command, **kwargs)
