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


def test_fallback_retry_cooldown_s_threaded_through_to_composite() -> None:
    """Regression — code-reviewer PR #107 round-3 finding 2.

    The composite's cooldown is operator-tunable via
    ``LLMConfig.fallback_retry_cooldown_s``. The factory MUST pass the
    configured value into ``FallbackLLMGateway`` rather than letting the
    constructor default mask a misconfiguration. Without this assertion,
    a future refactor that swapped the kwarg for a hardcoded literal
    would pass both the schema test (which checks the default) and the
    composite tests (which use their own custom values) while the rover
    silently used the wrong cooldown.
    """
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"
    cfg.llm.fallback_retry_cooldown_s = 75.0
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, FallbackLLMGateway)
    assert gw._retry_cooldown_s == 75.0


def test_default_fallback_retry_cooldown_s_propagates() -> None:
    """Default 30 s also propagates correctly — no silent override anywhere."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, FallbackLLMGateway)
    assert gw._retry_cooldown_s == 30.0


# --------------------------------------------------------------------------- #
# Observability — MetricsRegistry threading (keyword-only, None default)
# --------------------------------------------------------------------------- #
def _registry() -> object:
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

    return MetricsRegistry(MetricsConfig())


def test_metrics_threaded_into_anthropic_primary() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    reg = _registry()
    gw = build_llm_gateway(cfg, metrics=reg)
    assert isinstance(gw, AnthropicLLMGateway)
    assert gw._metrics is reg


def test_metrics_threaded_into_composite_and_both_tiers() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    cfg.llm.fallback_backend = "llama_cpp"
    reg = _registry()
    gw = build_llm_gateway(cfg, metrics=reg)
    assert isinstance(gw, FallbackLLMGateway)
    assert gw._metrics is reg
    assert gw._primary._metrics is reg  # anthropic primary instrumented


def test_metrics_none_default_keeps_legacy_construction() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    gw = build_llm_gateway(cfg)  # no metrics kwarg
    assert isinstance(gw, AnthropicLLMGateway)
    assert gw._metrics is None


def test_metrics_kwarg_is_keyword_only_with_none_default() -> None:
    """Pin the threading contract: metrics is keyword-only, defaults None."""
    import inspect

    sig = inspect.signature(build_llm_gateway)
    param = sig.parameters["metrics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None
