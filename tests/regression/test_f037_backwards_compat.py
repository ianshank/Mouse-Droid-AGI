"""F-037 backwards-compat: existing YAML and factory call sites still load.

Self-building a filter when ``None`` is additive. The factory signature,
default ``injection_patterns``, and shipped overlays must stay unchanged
so a ``git pull`` does not alter sanitisation envelopes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import LLMConfig, Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.openai_compatible import OpenAICompatibleLLMGateway
from mousedroid.security.injection_filter import RegexInjectionFilter

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_PATTERNS = (
    r"ignore (previous|above|all) instructions?",
    r"system prompt",
    r"you are now",
)


def test_llm_injection_pattern_defaults_unchanged() -> None:
    cfg = LLMConfig()
    assert cfg.injection_patterns == list(_DEFAULT_PATTERNS)
    assert cfg.max_command_len == 512


def test_explicit_filter_instance_is_still_stored() -> None:
    """Factory-threaded instance must win over the self-built default."""
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "openai_compatible"
    cfg.llm.base_url = "http://127.0.0.1:11434"
    filt = RegexInjectionFilter(("forbidden",), max_len=64)
    gw = build_llm_gateway(cfg, injection_filter=filt)
    assert isinstance(gw, OpenAICompatibleLLMGateway)
    assert gw._injection_filter is filt


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    cfg = Settings.model_validate({**raw, "mock_hardware": True})
    assert cfg.llm.injection_patterns == list(_DEFAULT_PATTERNS)
