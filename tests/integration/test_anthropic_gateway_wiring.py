"""Tier C-rover integration: Anthropic gateway + failover through the factory.

Exercises the real request path the rover uses — ``build_llm_gateway`` builds
the concrete :class:`AnthropicLLMGateway` (and the :class:`FallbackLLMGateway`
composite), and the orchestrator's ``process_mission`` rule-based -> LLM
fallback chain drives it — with the ``anthropic`` SDK faked end-to-end so no
network or API key is required.
"""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.constants import DEFAULT_AUDIO_CHUNK_SIZE, DEFAULT_BATTERY_VOLTAGE
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway
from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


# --------------------------------------------------------------------------- #
# Fake anthropic SDK
# --------------------------------------------------------------------------- #
class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    async def create(self, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._text)


def _make_sdk(reply: dict[str, float]) -> types.SimpleNamespace:
    text = json.dumps(reply)

    class _FakeAsyncClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.messages = _FakeMessages(text)

    sdk = types.SimpleNamespace()
    sdk.AsyncAnthropic = _FakeAsyncClient  # type: ignore[attr-defined]
    return sdk


# --------------------------------------------------------------------------- #
# Orchestrator harness (mirrors tests/integration/test_llm_gateway_wiring.py)
# --------------------------------------------------------------------------- #
def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, DEFAULT_BATTERY_VOLTAGE], dtype=np.float32),
        _audio_chunk=np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator(
    cfg: Settings,
    llm_gateway: Any,
    mission_parser: Any,
) -> MouseDroidOrchestrator:
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0
    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        llm_gateway=llm_gateway,
        mission_parser=mission_parser,
    )


def _unknown_parser() -> MagicMock:
    """A parser that defers everything to the LLM (UNKNOWN intent)."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.UNKNOWN,
        confidence=0.0,
        raw_command="navigate to the cantina",
    )
    return parser


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_anthropic_gateway_drives_process_mission() -> None:
    """build_llm_gateway(anthropic) → translate → orchestrator goal vector."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"

    gateway = build_llm_gateway(cfg)
    assert isinstance(gateway, AnthropicLLMGateway)
    gateway._sdk = _make_sdk({"vx": 0.6, "vy": 0.0, "omega": 0.2})  # type: ignore[attr-defined]
    await gateway.start()

    orch = _make_orchestrator(cfg, llm_gateway=gateway, mission_parser=_unknown_parser())
    goal = await orch.process_mission("navigate to the cantina")

    assert goal == GoalVector(vx_target=0.6, vy_target=0.0, omega_target=0.2)
    await gateway.stop()


@pytest.mark.asyncio
async def test_fallback_composite_uses_local_when_cloud_unavailable() -> None:
    """Cloud primary fails to start (no SDK) → local llama_cpp secondary serves."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"

    gateway = build_llm_gateway(cfg)
    assert isinstance(gateway, FallbackLLMGateway)

    # Primary: no SDK injected and start will look for the real one. Force the
    # degraded path deterministically by giving it an SDK without AsyncAnthropic.
    gateway._primary._sdk = types.SimpleNamespace()  # type: ignore[attr-defined]
    await gateway.start()

    # Primary is degraded (no async client); secondary is the llama_cpp gateway
    # which is degraded too in this env (no GGUF model) → composite returns a
    # neutral GoalVector WITHOUT raising. The point of this test is that the
    # control path never crashes when the cloud is unreachable.
    assert gateway.is_ready is False
    goal = await gateway.translate_mission("explore the corridor")
    assert isinstance(goal, GoalVector)
    await gateway.stop()


@pytest.mark.asyncio
async def test_fallback_prefers_healthy_cloud_primary() -> None:
    """When the cloud primary is healthy, the composite serves from it."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"

    gateway = build_llm_gateway(cfg)
    assert isinstance(gateway, FallbackLLMGateway)
    gateway._primary._sdk = _make_sdk({"vx": 0.9, "vy": 0.0, "omega": 0.0})  # type: ignore[attr-defined]
    await gateway.start()

    assert gateway.is_ready is True
    goal = await gateway.translate_mission("dash forward")
    assert goal == GoalVector(vx_target=0.9, vy_target=0.0, omega_target=0.0)
    await gateway.stop()
