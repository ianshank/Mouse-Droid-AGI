"""Tier C2.3: build_llm_gateway dispatches on cfg.llm.backend."""

from __future__ import annotations

from mousedroid.config.schema import Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.gateway import LLMGateway
from mousedroid.llm_gateway.openai_compatible import OpenAICompatibleLLMGateway


def test_dispatches_to_llama_cpp_when_backend_default() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.llm.backend == "llama_cpp"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, LLMGateway)


def test_dispatches_to_openai_compatible_when_configured() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "openai_compatible"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, OpenAICompatibleLLMGateway)


def test_returns_protocol_conforming_gateway_when_llm_disabled() -> None:
    """``cfg.llm.enabled=False`` still returns a protocol-conforming gateway."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.enabled = False
    gw = build_llm_gateway(cfg)
    assert hasattr(gw, "is_ready")
