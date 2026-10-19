"""F-059 WS-A backwards-compatibility regression tests.

F-059 WS-A adds one optional ``LLMConfig`` field, ``anthropic``, holding an
``AnthropicRequestConfig``. Per CLAUDE.md invariant 6 a new config field must
default so that existing YAML loads unchanged after a ``git pull``; here that
means the block defaults to ``None`` and ``None`` means the request the gateway
has always sent (``temperature`` included). A failure in this file means a rover
that pulled this change would refuse to start - or would silently change the
request it sends to the Claude API.

Not covered here (see the sibling AQA file): field descriptions, validators and
the expected-failure overlay-retarget pins.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import AnthropicRequestConfig, LLMConfig, Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_OVERLAYS = ("jetson_production.yaml", "jetson_claude_pilot.yaml")


# --------------------------------------------------------------------------- #
# Default-value invariants - these must never silently change
# --------------------------------------------------------------------------- #
def test_anthropic_block_defaults_and_absent_key() -> None:
    assert LLMConfig().anthropic is None
    assert AnthropicRequestConfig().omit_sampling_params is True
    assert AnthropicRequestConfig().thinking is None
    assert AnthropicRequestConfig().effort is None


def test_existing_llm_defaults_are_untouched() -> None:
    """The new field must not have moved any pre-existing default."""
    cfg = LLMConfig()
    assert cfg.backend == "llama_cpp"
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 256
    assert cfg.fallback_backend == "none"


# --------------------------------------------------------------------------- #
# Legacy YAML keeps loading
# --------------------------------------------------------------------------- #
def test_legacy_llm_yaml_without_the_key_loads_unchanged(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "platform: mouse_droid\n"
        "mock_hardware: true\n"
        "llm:\n"
        "  enabled: true\n"
        "  backend: anthropic\n"
        "  model_name: claude-haiku-4-5\n"
        "  temperature: 0.1\n"
        "  max_tokens: 128\n",
        encoding="utf-8",
    )
    settings = load_settings(legacy, config_dir=_CONFIG_DIR)
    assert settings.llm.anthropic is None
    assert settings.llm.model_name == "claude-haiku-4-5"
    assert settings.llm.temperature == 0.1
    assert settings.llm.max_tokens == 128


def test_default_yaml_still_loads_with_no_block() -> None:
    settings = load_settings(_CONFIG_DIR / "default.yaml", config_dir=_CONFIG_DIR)
    assert isinstance(settings, Settings)
    assert settings.llm.anthropic is None


@pytest.mark.parametrize("overlay", _OVERLAYS)
def test_shipped_overlays_are_unchanged_by_the_profile_pr(overlay: str) -> None:
    """THIS PR lands the block only; the overlays still target Haiku 4.5, block absent.

    The retarget PR (``claude-haiku-5-5`` + the block) deletes this test together
    with the ``xfail`` marker on the AQA overlay pins - both flip in the same
    change, by design.
    """
    llm = load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR).llm
    assert llm.backend == "anthropic"
    assert llm.model_name == "claude-haiku-4-5"
    assert llm.temperature == 0.1
    assert llm.anthropic is None


def test_block_round_trips_from_yaml(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "mock_hardware: true\n"
        "llm:\n"
        "  backend: anthropic\n"
        "  model_name: claude-haiku-5-5\n"
        "  anthropic:\n"
        "    omit_sampling_params: true\n"
        "    thinking: disabled\n"
        "    effort: low\n",
        encoding="utf-8",
    )
    block = load_settings(overlay, config_dir=_CONFIG_DIR).llm.anthropic
    assert block is not None
    assert block.omit_sampling_params is True
    assert block.thinking == "disabled"
    assert block.effort == "low"


# --------------------------------------------------------------------------- #
# Operator env lever - only the real loader can prove an env var reaches the field
# --------------------------------------------------------------------------- #
def test_env_override_reaches_the_nested_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "mock_hardware: true\nllm:\n  backend: anthropic\n  model_name: claude-haiku-5-5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MOUSEDROID_LLM__ANTHROPIC__THINKING", "disabled")
    block = load_settings(overlay, config_dir=_CONFIG_DIR).llm.anthropic
    assert block is not None
    assert block.thinking == "disabled"
    assert block.omit_sampling_params is True  # untouched sibling keeps its default


def test_env_override_wins_over_the_yaml_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "mock_hardware: true\n"
        "llm:\n"
        "  backend: anthropic\n"
        "  anthropic:\n"
        "    thinking: adaptive\n"
        "    effort: low\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MOUSEDROID_LLM__ANTHROPIC__THINKING", "disabled")
    block = load_settings(overlay, config_dir=_CONFIG_DIR).llm.anthropic
    assert block is not None
    assert block.thinking == "disabled"
    assert block.effort == "low"  # only the overridden key moved


def test_env_null_removes_the_yaml_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``MOUSEDROID_LLM__ANTHROPIC=null`` restores the legacy request without editing YAML.

    This is the switch the paired before/after probe run relies on (same overlay,
    ``MOUSEDROID_LLM__MODEL_NAME=claude-haiku-4-5``, block nulled): only the real
    loader can show an env var reaches - and here, removes - a nested block.
    """
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "mock_hardware: true\n"
        "llm:\n"
        "  backend: anthropic\n"
        "  model_name: claude-haiku-5-5\n"
        "  anthropic:\n"
        "    omit_sampling_params: true\n"
        "    thinking: disabled\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MOUSEDROID_LLM__ANTHROPIC", "null")
    assert load_settings(overlay, config_dir=_CONFIG_DIR).llm.anthropic is None


# --------------------------------------------------------------------------- #
# The request an unchanged config produces
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("overlay", _OVERLAYS)
async def test_shipped_overlays_still_send_the_pre_f059_request(overlay: str) -> None:
    """With the block absent the gateway builds exactly the legacy kwargs."""
    from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway

    calls: list[dict[str, Any]] = []

    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="{}")])

    class _Client:
        def __init__(self, **_kw: Any) -> None:
            self.messages = _Messages()

    llm = load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR).llm
    gw = AnthropicLLMGateway(llm, sdk=types.SimpleNamespace(AsyncAnthropic=_Client))
    await gw.start()
    await gw.translate_mission("turn left slowly")
    assert calls[-1] == {
        "model": llm.model_name,
        "max_tokens": llm.max_tokens,
        "temperature": llm.temperature,
        "system": llm.system_prompt,
        "messages": [{"role": "user", "content": "turn left slowly"}],
        "timeout": llm.request_timeout_s,
    }
