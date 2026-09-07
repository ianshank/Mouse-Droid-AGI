"""AQA pins for F-037 HTTP-gateway constructor sanitisation symmetry.

Pins the constructor / factory signatures rather than a YAML field: the
HTTP backend must always store a ``PromptInjectionFilterProtocol``, and
``build_llm_gateway`` must still accept ``injection_filter=None`` so the
self-build path stays the documented default for forgetful callers.
"""

from __future__ import annotations

import inspect

from mousedroid.config.schema import LLMConfig, Settings
from mousedroid.factory import build_injection_filter, build_llm_gateway
from mousedroid.llm_gateway.openai_compatible import OpenAICompatibleLLMGateway
from mousedroid.security.injection_filter import (
    PromptInjectionFilterProtocol,
    RegexInjectionFilter,
)


def test_openai_compatible_ctor_injection_filter_defaults_none() -> None:
    """Callers may still omit the kwarg; the gateway must not require it."""
    param = inspect.signature(OpenAICompatibleLLMGateway.__init__).parameters["injection_filter"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_openai_compatible_ctor_always_stores_a_protocol_filter() -> None:
    gw = OpenAICompatibleLLMGateway(LLMConfig(backend="openai_compatible"))
    assert isinstance(gw._injection_filter, RegexInjectionFilter)
    assert isinstance(gw._injection_filter, PromptInjectionFilterProtocol)


def test_build_llm_gateway_injection_filter_still_optional() -> None:
    param = inspect.signature(build_llm_gateway).parameters["injection_filter"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_factory_none_filter_http_backend_self_builds() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "openai_compatible"
    cfg.llm.base_url = "http://127.0.0.1:11434"
    gw = build_llm_gateway(cfg)
    assert isinstance(gw, OpenAICompatibleLLMGateway)
    assert isinstance(gw._injection_filter, RegexInjectionFilter)


def test_build_injection_filter_is_the_cli_shared_instance_constructor() -> None:
    """CLIs must keep calling this — not a one-off RegexInjectionFilter()."""
    sig = inspect.signature(build_injection_filter)
    assert list(sig.parameters) == ["cfg"]
