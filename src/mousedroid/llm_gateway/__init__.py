"""LLM gateway for natural-language goal translation."""

from __future__ import annotations

from mousedroid.llm_gateway.config import GatewayConfig
from mousedroid.llm_gateway.gateway import LLMGateway
from mousedroid.llm_gateway.mission_parser import (
    IntentType,
    MissionIntent,
    MissionParserProtocol,
    RuleBasedMissionParser,
)
from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol

# NOTE: the concrete cloud/composite gateways (``AnthropicLLMGateway``,
# ``FallbackLLMGateway``) and ``OpenAICompatibleLLMGateway`` are intentionally
# NOT re-exported here. Per the CLAUDE.md DI invariant, concrete gateway types
# are imported only inside the :mod:`mousedroid.factory` builders; application
# code depends on :class:`LLMGatewayProtocol`. (``OpenAICompatibleLLMGateway``
# additionally imports ``aiohttp`` — an optional ``[telemetry]`` dep — at module
# load time, so eager re-export would force that dependency.) Import the
# concrete classes from their submodules directly when needed (e.g. tests).

__all__ = [
    "GatewayConfig",
    "GoalVector",
    "IntentType",
    "LLMGateway",
    "LLMGatewayProtocol",
    "MissionIntent",
    "MissionParserProtocol",
    "RuleBasedMissionParser",
]
