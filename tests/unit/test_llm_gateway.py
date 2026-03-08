"""Tests for LLM Gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mousedroid.llm_gateway.config import GatewayConfig
from mousedroid.llm_gateway.gateway import LLMGateway
from mousedroid.llm_gateway.protocol import GoalVector

# -- GatewayConfig tests --


def test_gateway_config_construction():
    cfg = GatewayConfig(model_path="/tmp/model.gguf")
    assert cfg.n_threads == 4
    assert cfg.n_gpu_layers == -1
    assert cfg.max_tokens == 256
    assert cfg.temperature == 0.1
    assert len(cfg.stop_tokens) == 2


def test_gateway_config_custom_values():
    cfg = GatewayConfig(
        model_path="/tmp/model.gguf",
        n_threads=8,
        max_tokens=128,
    )
    assert cfg.n_threads == 8
    assert cfg.max_tokens == 128


# -- LLMGateway tests --


@pytest.fixture
def gateway() -> LLMGateway:
    cfg = GatewayConfig(model_path="/tmp/fake.gguf")
    return LLMGateway(cfg)


def test_gateway_constructor(gateway: LLMGateway):
    assert gateway._model is None


async def test_translate_mission_empty_raises(gateway: LLMGateway):
    with pytest.raises(ValueError, match="non-empty"):
        await gateway.translate_mission("")


async def test_translate_mission_whitespace_raises(gateway: LLMGateway):
    with pytest.raises(ValueError, match="non-empty"):
        await gateway.translate_mission("   ")


async def test_translate_mission_without_start_returns_default(gateway: LLMGateway):
    result = await gateway.translate_mission("go forward")
    assert result == GoalVector()


def test_parse_response_valid_json(gateway: LLMGateway):
    raw = '{"vx": 0.5, "vy": -0.3, "omega": 0.8}'
    result = gateway._parse_response(raw)
    assert result.vx_target == 0.5
    assert result.vy_target == -0.3
    assert result.omega_target == 0.8


def test_parse_response_clamps_values(gateway: LLMGateway):
    raw = '{"vx": 5.0, "vy": -5.0, "omega": 0.0}'
    result = gateway._parse_response(raw)
    assert result.vx_target == 1.0
    assert result.vy_target == -1.0


def test_parse_response_invalid_json_returns_default(gateway: LLMGateway):
    result = gateway._parse_response("not json at all")
    assert result == GoalVector()


def test_parse_response_partial_json(gateway: LLMGateway):
    raw = '{"vx": 0.2}'
    result = gateway._parse_response(raw)
    assert result.vx_target == 0.2
    assert result.vy_target == 0.0
    assert result.omega_target == 0.0


async def test_stop_sets_model_to_none(gateway: LLMGateway):
    gateway._model = "fake_model"
    await gateway.stop()
    assert gateway._model is None


async def test_start_raises_without_llama_cpp(gateway: LLMGateway):
    with (
        patch.object(gateway, "_load_model", side_effect=ImportError("no llama")),
        pytest.raises(RuntimeError, match="llama-cpp-python"),
    ):
        await gateway.start()


async def test_translate_mission_with_model(gateway: LLMGateway):
    gateway._model = MagicMock()
    raw_json = '{"vx": 0.7, "vy": 0.0, "omega": -0.2}'
    with patch.object(gateway, "_infer_sync", return_value=raw_json):
        result = await gateway.translate_mission("go forward fast")
    assert result.vx_target == 0.7
    assert result.omega_target == -0.2
