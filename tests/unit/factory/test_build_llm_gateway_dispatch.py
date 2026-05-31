"""Tier C2.3 / C-rover: build_llm_gateway dispatches on cfg.llm.backend."""

from __future__ import annotations

from mousedroid.config.schema import Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway
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


def test_dispatches_to_anthropic_when_configured() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, AnthropicLLMGateway)


def test_returns_protocol_conforming_gateway_when_llm_disabled() -> None:
    """``cfg.llm.enabled=False`` still returns a protocol-conforming gateway."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.enabled = False
    gw = build_llm_gateway(cfg)
    assert hasattr(gw, "is_ready")


def test_no_fallback_wrap_by_default() -> None:
    """Default ``fallback_backend='none'`` returns the bare primary gateway."""
    cfg = Settings(mock_hardware=True)
    assert cfg.llm.fallback_backend == "none"
    gw = build_llm_gateway(cfg)
    assert not isinstance(gw, FallbackLLMGateway)


def test_wraps_in_fallback_composite_when_fallback_backend_set() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, FallbackLLMGateway)
    assert isinstance(gw._primary, AnthropicLLMGateway)
    assert isinstance(gw._secondary, LLMGateway)


def test_fallback_model_name_override_applies_to_secondary_only() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "openai_compatible"
    cfg.llm.fallback_model_name = "gemma-local"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, FallbackLLMGateway)
    assert isinstance(gw._secondary, OpenAICompatibleLLMGateway)
    assert gw._secondary._cfg.model_name == "gemma-local"
    assert gw._secondary._cfg.backend == "openai_compatible"
    # Primary keeps its own model id — override is secondary-scoped.
    assert gw._primary._cfg.model_name == "claude-haiku-4-5"


def test_same_backend_fallback_is_noop() -> None:
    """``fallback_backend == backend`` skips the composite (pointless)."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "llama_cpp"
    cfg.llm.fallback_backend = "llama_cpp"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, LLMGateway)
    assert not isinstance(gw, FallbackLLMGateway)
