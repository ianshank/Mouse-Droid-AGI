"""Integration: build_llm_gateway on the production config yields the composite.

Asserts that the deployed jetson_production.yaml wiring (anthropic primary +
llama_cpp fallback) produces a FallbackLLMGateway through the real factory.
No network / API key / GGUF load — we only inspect the composite's structure,
not run inference.
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


def test_production_config_builds_fallback_composite() -> None:
    settings = load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
    gateway = build_llm_gateway(settings)
    assert isinstance(gateway, FallbackLLMGateway)
