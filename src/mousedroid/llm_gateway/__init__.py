"""LLM gateway for natural-language goal translation."""

from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.config import GatewayConfig
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway
from mousedroid.llm_gateway.gateway import LLMGateway
from mousedroid.llm_gateway.mission_parser import (
    IntentType,
    MissionIntent,
    MissionParserProtocol,
    RuleBasedMissionParser,
)
from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol

# NOTE: ``OpenAICompatibleLLMGateway`` is intentionally NOT re-exported here.
# It imports ``aiohttp`` (an optional ``[telemetry]`` dependency) at module
# load time; eagerly importing it would force that dep on every
# ``import mousedroid.llm_gateway``. The factory imports it lazily instead.
# ``AnthropicLLMGateway`` is safe to export — its SDK import is deferred to
# ``start()`` — and ``FallbackLLMGateway`` has no heavy imports.

__all__ = [
    "AnthropicLLMGateway",
    "FallbackLLMGateway",
    "GatewayConfig",
    "GoalVector",
    "IntentType",
    "LLMGateway",
    "LLMGatewayProtocol",
    "MissionIntent",
    "MissionParserProtocol",
    "RuleBasedMissionParser",
]
