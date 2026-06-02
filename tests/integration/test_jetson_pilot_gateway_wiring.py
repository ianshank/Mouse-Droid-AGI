"""Integration: build_llm_gateway on the production config yields the composite.

Asserts that the deployed jetson_production.yaml wiring (anthropic primary +
llama_cpp fallback) produces a FallbackLLMGateway through the real factory.
No network / API key / GGUF load — we only inspect the composite's structure,
not run inference.

This test deliberately verifies that ``build_llm_gateway`` constructs the
composite WITHOUT requiring the heavy optional ``llama-cpp-python`` (or the
``anthropic`` SDK) to be installed: the ``llama_cpp`` import is lazy, deferred
to ``LLMGateway.start()`` -> ``_load_model()`` (``from llama_cpp import Llama``
at src/mousedroid/llm_gateway/gateway.py:106), and the ``anthropic`` SDK is
imported only inside the Anthropic gateway's client init. Construction touches
neither. Because this test never calls ``gateway.start()``, it is the proof
that the wiring builds dependency-free — so it must NOT be guarded with
``pytest.importorskip("llama_cpp")``, which would silently skip the very case
it exists to cover. Type identity is asserted via ``type(...).__name__``
string checks rather than ``isinstance`` against the concrete classes, so the
assertions never trigger an import of the optional backends' runtime deps.
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

    # Inspect the composite's tiers by class NAME (not isinstance) so we prove
    # the production wiring (anthropic primary + llama_cpp secondary) without
    # importing either backend's optional runtime dependency. The factory has
    # already constructed both child gateways at this point — their heavy deps
    # (anthropic SDK, llama-cpp-python) remain unimported because those imports
    # are deferred to each gateway's start()/client init.
    assert type(gateway._primary).__name__ == "AnthropicLLMGateway"
    assert type(gateway._secondary).__name__ == "LLMGateway"
