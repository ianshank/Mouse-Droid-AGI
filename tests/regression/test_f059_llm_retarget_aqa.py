"""Automated Quality Assurance (AQA) - schema hygiene for the F-059 WS-A request profile.

F-059 WS-A adds ``AnthropicRequestConfig`` and the optional ``LLMConfig.anthropic``
block: a per-model request shape for the Anthropic backend. Claude Haiku 5.5 and
Sonnet 5.5 answer HTTP 400 to any non-default ``temperature`` / ``top_p`` /
``top_k``, Haiku 5.5 thinks by default, and ``output_config.effort`` is the
cost/latency lever - so a bare model-id swap in an overlay does not work. The
block lets an overlay say what the new model needs, and its absence keeps the
request byte-identical to the one the gateway has always sent.

Pinned off ``model_fields`` (the ``FieldInfo``), never off a live instance: an
instance only proves the default is *legal*, not that it is *declared* the way
this file claims.

The two overlay pins at the bottom are EXPECTED FAILURES in the PR that
introduces the block. The overlays are retargeted to ``claude-haiku-5-5`` in a
separate, tiny PR after the on-Jetson probe proves ``served: primary`` with the
rotated key - a bare id swap would 400 and fall back to Phi-3 silently (plan risk
R10). ``strict=True`` makes the retarget PR fail loudly until it removes the
marker, so the pin cannot be forgotten, and ``raises=AssertionError`` keeps a
loader crash from masquerading as the expected failure.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import AnthropicRequestConfig, LLMConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"

_REQUEST_FIELDS = ("omit_sampling_params", "thinking", "effort")

# A description is only useful to an operator if it says something; the project
# convention (regression-pair-scaffold skill) is "longer than 20 characters".
_MIN_DESCRIPTION_CHARS = 20


# --------------------------------------------------------------------------- #
# Field hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _REQUEST_FIELDS)
def test_request_field_has_an_explanatory_description(name: str) -> None:
    info = AnthropicRequestConfig.model_fields[name]
    assert info.description, f"AnthropicRequestConfig.{name} has no description"
    assert len(info.description) > _MIN_DESCRIPTION_CHARS, info.description


def test_request_profile_defaults_are_pinned_off_field_info() -> None:
    fields = AnthropicRequestConfig.model_fields
    assert fields["omit_sampling_params"].default is True
    assert fields["thinking"].default is None
    assert fields["effort"].default is None


def test_llm_config_anthropic_field_is_declared_optional_and_described() -> None:
    info = LLMConfig.model_fields["anthropic"]
    assert info.default is None, "absent block must stay the default (byte-identical request)"
    assert info.description, "LLMConfig.anthropic has no description"
    assert len(info.description) > _MIN_DESCRIPTION_CHARS, info.description
    annotation = str(info.annotation)
    assert "AnthropicRequestConfig" in annotation
    assert "None" in annotation, "the block must be optional"


# --------------------------------------------------------------------------- #
# Validation - a misconfigured block is rejected at load time, not no-op'd
# --------------------------------------------------------------------------- #
def test_unknown_key_in_the_block_is_rejected() -> None:
    """``StrictBaseModel``: a typo must not silently fall back to the defaults."""
    with pytest.raises(ValidationError, match="thinkng"):
        AnthropicRequestConfig(thinkng="disabled")  # type: ignore[call-arg]


def test_sampling_values_do_not_live_in_the_block() -> None:
    """The block can only OMIT sampling parameters; the value stays ``LLMConfig.temperature``."""
    with pytest.raises(ValidationError, match="temperature"):
        AnthropicRequestConfig(temperature=0.5)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["enabled", "between_tools", "ADAPTIVE", ""])
def test_thinking_rejects_modes_haiku_5_5_does_not_take(bad: str) -> None:
    """``enabled`` (a thinking budget) 400s on Haiku 5.5; ``between_tools`` is Sonnet 5.5 only."""
    with pytest.raises(ValidationError, match="thinking"):
        AnthropicRequestConfig(thinking=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("good", ["adaptive", "disabled", None])
def test_thinking_accepts_the_documented_modes(good: str | None) -> None:
    assert AnthropicRequestConfig(thinking=good).thinking == good  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["xhigh", "max", "extreme", ""])
def test_effort_stops_at_high(bad: str) -> None:
    """``thinking: disabled`` 400s at ``xhigh``/``max`` on Haiku 5.5.

    Capping the schema at ``high`` keeps every combination the schema can express
    a combination the model accepts.
    """
    with pytest.raises(ValidationError, match="effort"):
        AnthropicRequestConfig(effort=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("good", ["low", "medium", "high", None])
def test_effort_accepts_the_documented_levels(good: str | None) -> None:
    assert AnthropicRequestConfig(effort=good).effort == good  # type: ignore[arg-type]


def test_llm_config_accepts_the_block_as_a_mapping() -> None:
    cfg = LLMConfig(anthropic={"thinking": "disabled", "effort": "low"})  # type: ignore[arg-type]
    assert isinstance(cfg.anthropic, AnthropicRequestConfig)
    assert cfg.anthropic.omit_sampling_params is True  # default survives a partial block


# --------------------------------------------------------------------------- #
# What the gateway may put on the wire (verified facts, not preferences)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    [None, {}, {"omit_sampling_params": False}, {"thinking": "disabled", "effort": "low"}],
)
async def test_gateway_never_sends_top_p_or_top_k(block: dict[str, Any] | None) -> None:
    """Haiku 5.5 / Sonnet 5.5 400 on any ``top_k`` and any non-default ``top_p``.

    ``LLMConfig`` has no such field, and this pins that no request shape grows one.
    """
    from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway

    calls: list[dict[str, Any]] = []

    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="{}")])

    class _Client:
        def __init__(self, **_kw: Any) -> None:
            self.messages = _Messages()

    cfg = LLMConfig(backend="anthropic", model_name="claude-haiku-4-5", anthropic=block)  # type: ignore[arg-type]
    gw = AnthropicLLMGateway(cfg, sdk=types.SimpleNamespace(AsyncAnthropic=_Client))
    await gw.start()
    await gw.translate_mission("go")
    assert calls, "the request never reached the SDK"
    assert not ({"top_p", "top_k"} & set(calls[-1]))
    extra = calls[-1].get("extra_body") or {}
    assert not ({"top_p", "top_k"} & set(extra))


# --------------------------------------------------------------------------- #
# Overlay retarget pins - EXPECTED FAILURES until the retarget PR (see docstring)
# --------------------------------------------------------------------------- #
def _llm(overlay: str) -> LLMConfig:
    return load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR).llm


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="overlay retarget lands with the probe evidence",
)
def test_production_overlay_targets_haiku_5_5_with_sampling_omitted() -> None:
    cfg = _llm("jetson_production.yaml")
    assert cfg.model_name == "claude-haiku-5-5"
    assert cfg.anthropic is not None
    assert cfg.anthropic.omit_sampling_params is True
    assert cfg.anthropic.thinking == "disabled"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="overlay retarget lands with the probe evidence",
)
def test_pilot_overlay_targets_haiku_5_5_with_sampling_omitted() -> None:
    cfg = _llm("jetson_claude_pilot.yaml")
    assert cfg.model_name == "claude-haiku-5-5"
    assert cfg.anthropic is not None
    assert cfg.anthropic.omit_sampling_params is True
    assert cfg.anthropic.thinking == "disabled"
