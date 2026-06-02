"""Regression: pin config/jetson_production.yaml's PR #107 LLM wiring.

Guards against silent drift of the deployed cloud-primary / local-fallback
wiring. (Backwards-compat of the LLMConfig DEFAULTS is covered separately by
tests/regression/test_pr107_backwards_compat.py — this file pins the PRODUCTION
overlay's explicit values.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"


def _load() -> Settings:
    return load_settings(_PROD, config_dir=_CONFIG_DIR)


def test_production_uses_anthropic_primary() -> None:
    assert _load().llm.backend == "anthropic"


def test_production_falls_back_to_llama_cpp() -> None:
    assert _load().llm.fallback_backend == "llama_cpp"


def test_production_model_name_is_a_claude_id() -> None:
    assert _load().llm.model_name.startswith("claude-")


def test_production_fallback_model_path_is_phi3() -> None:
    # The off-network fallback reuses the already-staged Phi-3-mini GGUF.
    assert "Phi-3-mini" in str(_load().llm.model_path)


def test_production_cooldown_positive() -> None:
    assert _load().llm.fallback_retry_cooldown_s > 0.0


def test_production_loads_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load must not require a key (the gateway degrades, not parse)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOUSEDROID_LLM__API_KEY", raising=False)
    assert isinstance(_load(), Settings)
