"""Tier C2.3: LLMConfig fields for the OpenAI-compatible HTTP backend."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from mousedroid.config.schema import LLMConfig, Settings


def test_backend_defaults_to_llama_cpp_for_backcompat() -> None:
    """Existing deployments running the GGUF backend keep working unchanged."""
    assert LLMConfig().backend == "llama_cpp"


def test_base_url_default_targets_local_ollama() -> None:
    """Default base_url points at the canonical Ollama port."""
    assert LLMConfig().base_url == "http://127.0.0.1:11434"


def test_model_name_default_is_gemma_4_e4b() -> None:
    assert LLMConfig().model_name == "gemma-4-e4b"


def test_api_key_defaults_to_none() -> None:
    """Anonymous local Ollama needs no key; OpenAI deployments override."""
    assert LLMConfig().api_key is None


def test_request_timeout_default_above_latency_target() -> None:
    cfg = LLMConfig()
    # request_timeout_s must be >= latency_target_ms/1000 so the HTTP
    # client doesn't kill a request the gateway would tolerate.
    assert cfg.request_timeout_s >= cfg.latency_target_ms / 1000.0


def test_backend_rejects_unknown_literal() -> None:
    with pytest.raises(ValueError):
        LLMConfig(backend="bogus")  # type: ignore[arg-type]


def test_settings_env_overrides_via_mousedroid_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MOUSEDROID_LLM__BASE_URL`` env var overrides the schema default.

    Pins the existing Pydantic ``env_nested_delimiter='__'`` contract on
    :class:`Settings` so operators can wire LM Studio / Ollama / OpenAI
    by env without YAML edits.
    """
    monkeypatch.setenv("MOUSEDROID_LLM__BASE_URL", "http://192.168.55.1:11434")
    monkeypatch.setenv("MOUSEDROID_LLM__MODEL_NAME", "llama-3.1-8b")
    settings = Settings()
    assert settings.llm.base_url == "http://192.168.55.1:11434"
    assert settings.llm.model_name == "llama-3.1-8b"


def test_api_key_loads_as_secret_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key must be a ``SecretStr`` so it doesn't leak through repr."""
    monkeypatch.setenv("MOUSEDROID_LLM__API_KEY", "sk-test-xyz")
    settings = Settings()
    assert isinstance(settings.llm.api_key, SecretStr)
    assert settings.llm.api_key.get_secret_value() == "sk-test-xyz"
    # Default repr must NOT include the secret.
    assert "sk-test-xyz" not in repr(settings.llm)
