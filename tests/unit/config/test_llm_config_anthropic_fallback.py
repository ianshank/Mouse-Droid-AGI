"""Tier C-rover: LLMConfig schema fields for anthropic + failover backends.

Backwards-compatibility AQA: existing YAML / env must load unchanged, the new
fields must default to disabled, and the ``backend`` Literal must accept the
new ``anthropic`` value while still rejecting unknown values.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import LLMConfig, Settings


def test_backend_literal_accepts_anthropic() -> None:
    assert LLMConfig(backend="anthropic").backend == "anthropic"


def test_backend_literal_still_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="backend"):
        LLMConfig(backend="gpt5-turbo")  # type: ignore[arg-type]


def test_fallback_backend_defaults_to_none_for_backcompat() -> None:
    """Existing single-backend deployments stay byte-identical."""
    assert LLMConfig().fallback_backend == "none"


def test_fallback_model_name_defaults_to_none() -> None:
    assert LLMConfig().fallback_model_name is None


def test_fallback_backend_accepts_local_backends() -> None:
    assert LLMConfig(fallback_backend="llama_cpp").fallback_backend == "llama_cpp"
    assert LLMConfig(fallback_backend="openai_compatible").fallback_backend == "openai_compatible"


def test_fallback_backend_rejects_anthropic_as_secondary() -> None:
    """Failover must be to a LOCAL backend — cloud->cloud is disallowed."""
    with pytest.raises(ValueError, match="fallback_backend"):
        LLMConfig(fallback_backend="anthropic")  # type: ignore[arg-type]


def test_fallback_retry_cooldown_defaults_to_30s() -> None:
    assert LLMConfig().fallback_retry_cooldown_s == 30.0


def test_fallback_retry_cooldown_must_be_positive() -> None:
    with pytest.raises(ValueError, match="fallback_retry_cooldown_s"):
        LLMConfig(fallback_retry_cooldown_s=0.0)


def test_existing_minimal_yaml_loads_unchanged() -> None:
    """A config that predates these fields still validates with defaults."""
    cfg = LLMConfig(enabled=True, model_name="gemma-4-e4b")
    assert cfg.backend == "llama_cpp"
    assert cfg.fallback_backend == "none"
    assert cfg.fallback_model_name is None


def test_env_overrides_for_anthropic_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can wire cloud-primary + local-fallback purely by env."""
    monkeypatch.setenv("MOUSEDROID_LLM__BACKEND", "anthropic")
    monkeypatch.setenv("MOUSEDROID_LLM__MODEL_NAME", "claude-haiku-4-5")
    monkeypatch.setenv("MOUSEDROID_LLM__API_KEY", "sk-ant-xyz")
    monkeypatch.setenv("MOUSEDROID_LLM__FALLBACK_BACKEND", "llama_cpp")
    monkeypatch.setenv("MOUSEDROID_LLM__REQUEST_TIMEOUT_S", "20")
    settings = Settings()
    assert settings.llm.backend == "anthropic"
    assert settings.llm.model_name == "claude-haiku-4-5"
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "sk-ant-xyz"
    assert settings.llm.fallback_backend == "llama_cpp"
    assert settings.llm.request_timeout_s == 20.0
    # Secret must not leak through repr.
    assert "sk-ant-xyz" not in repr(settings.llm)
