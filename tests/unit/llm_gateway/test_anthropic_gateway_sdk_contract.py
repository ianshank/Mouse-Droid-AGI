"""F-059 WS-A: what the gateway asks of the SDK, and what the REAL SDK does with it.

``test_anthropic_gateway.py`` fakes the SDK, which proves what the gateway *asks
for*. Only the real SDK can prove it *accepts* it - and the two have already
diverged once: ``anthropic>=1.0`` removed the ``temperature`` / ``top_p`` /
``top_k`` keywords from ``messages.create`` (passing one is a ``TypeError``),
which the fakes (``**kwargs``) could never have shown.

Two groups:

* ``_accepts_keyword`` - the capability probe the gateway uses to ask the
  installed SDK what ``messages.create`` takes. Pure Python, always runs.
* Wire tests - the gateway driven through a REAL ``anthropic.AsyncAnthropic``
  whose HTTP transport is a mock, so the JSON body the SDK would POST can be
  asserted. No network, no key. They skip when the optional ``anthropic`` extra
  is absent (CI's ``test`` job installs ``[dev,telemetry,mcp]`` only); install it
  with ``pip install -e ".[anthropic]"`` to run them.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from typing import Any

import pytest

from mousedroid.config.schema import LLMConfig
from mousedroid.llm_gateway import anthropic_gateway as gateway_module
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.protocol import GoalVector


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    # Resolved at call time, not import time, so this module still COLLECTS against a
    # gateway that predates the helper and the wire tests below fail there (rather than
    # the whole file erroring out) - which is what lets prove_pin_fails.sh get a verdict.
    return gateway_module._accepts_keyword(func, name)


# --------------------------------------------------------------------------- #
# _accepts_keyword - the capability probe
# --------------------------------------------------------------------------- #
def _closed(*, model: str, thinking: Any = None) -> None:
    """A signature with a closed keyword list (like the real SDK)."""


def _open(**kwargs: Any) -> None:
    """A ``**kwargs`` signature (accepts anything)."""


def _positional_only(model: str, /, thinking: Any = None) -> None:
    """``model`` is positional-only, so the KEYWORD ``model`` is not accepted."""


class _Opaque:
    """A callable whose signature cannot be read (a C extension, some mocks)."""

    @property
    def __signature__(self) -> Any:
        raise ValueError("no signature available")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


def test_accepts_keyword_true_for_a_declared_keyword() -> None:
    assert _accepts_keyword(_closed, "thinking") is True
    assert _accepts_keyword(_closed, "model") is True


def test_accepts_keyword_false_for_an_undeclared_keyword_on_a_closed_signature() -> None:
    assert _accepts_keyword(_closed, "temperature") is False
    assert _accepts_keyword(_closed, "output_config") is False


def test_accepts_keyword_true_for_anything_when_the_callable_takes_kwargs() -> None:
    assert _accepts_keyword(_open, "temperature") is True
    assert _accepts_keyword(_open, "anything_at_all") is True


def test_accepts_keyword_false_for_a_positional_only_parameter() -> None:
    assert _accepts_keyword(_positional_only, "model") is False
    assert _accepts_keyword(_positional_only, "thinking") is True


def test_accepts_keyword_assumes_true_when_the_signature_is_unreadable() -> None:
    """A capability probe must never raise out of the request path."""
    assert _accepts_keyword(_Opaque(), "temperature") is True


def test_accepts_keyword_works_on_bound_async_methods() -> None:
    class _Messages:
        async def create(self, *, model: str, max_tokens: int) -> None:
            return None

    assert _accepts_keyword(_Messages().create, "max_tokens") is True
    assert _accepts_keyword(_Messages().create, "temperature") is False


# --------------------------------------------------------------------------- #
# Wire tests against the real SDK (skipped when the extra is not installed)
# --------------------------------------------------------------------------- #
_MESSAGE_JSON: dict[str, Any] = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": '{"vx": 0.0, "vy": 0.0, "omega": 0.5}'}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 9},
}


@pytest.fixture
def wire() -> Callable[[LLMConfig], tuple[AnthropicLLMGateway, list[dict[str, Any]]]]:
    """Build a gateway over a real ``AsyncAnthropic`` with a mock transport.

    Returns a factory ``cfg -> (gateway, bodies)``; ``bodies`` collects the JSON
    document of every request the SDK would have sent.
    """
    anthropic = pytest.importorskip(
        "anthropic", reason="optional extra: pip install -e '.[anthropic]'"
    )
    try:  # anthropic>=1.0 is built on httpx2; older releases on httpx
        import httpx2 as httpx_lib
    except ImportError:  # pragma: no cover - only on anthropic<1.0
        httpx_lib = pytest.importorskip("httpx")

    def _build(cfg: LLMConfig) -> tuple[AnthropicLLMGateway, list[dict[str, Any]]]:
        bodies: list[dict[str, Any]] = []

        def _handler(request: Any) -> Any:
            bodies.append(json.loads(request.content))
            return httpx_lib.Response(200, json=_MESSAGE_JSON)

        transport = httpx_lib.MockTransport(_handler)

        def _client(**_kwargs: Any) -> Any:
            return anthropic.AsyncAnthropic(
                api_key="test-key",
                base_url="http://mock.invalid",
                http_client=anthropic.DefaultAsyncHttpxClient(transport=transport),
                max_retries=0,
            )

        sdk = types.SimpleNamespace(AsyncAnthropic=_client)
        return AnthropicLLMGateway(cfg, sdk=sdk), bodies

    return _build


def _cfg(**overrides: Any) -> LLMConfig:
    base: dict[str, Any] = {"backend": "anthropic", "model_name": "claude-haiku-4-5"}
    base.update(overrides)
    return LLMConfig(**base)


@pytest.mark.asyncio
async def test_legacy_request_is_accepted_by_the_real_sdk_and_carries_temperature(
    wire: Callable[[LLMConfig], tuple[AnthropicLLMGateway, list[dict[str, Any]]]],
) -> None:
    """No block: the temperature still reaches the wire, whichever SDK is installed.

    On ``anthropic>=1.0`` a raw ``temperature=`` keyword is a ``TypeError`` that
    degrades the gateway before any request is made; this proves the legacy
    request survives that, and on ``anthropic<1.0`` that it is unchanged.
    """
    gw, bodies = wire(_cfg(temperature=0.2))
    await gw.start()
    goal = await gw.translate_mission("turn left")

    assert gw.is_degraded is False
    assert goal == GoalVector(vx_target=0.0, vy_target=0.0, omega_target=0.5)
    body = bodies[-1]
    assert body["temperature"] == 0.2
    assert body["model"] == "claude-haiku-4-5"
    assert "thinking" not in body
    assert "output_config" not in body


@pytest.mark.asyncio
async def test_profile_request_is_accepted_by_the_real_sdk_without_sampling(
    wire: Callable[[LLMConfig], tuple[AnthropicLLMGateway, list[dict[str, Any]]]],
) -> None:
    """With the block: no sampling parameter, ``thinking`` and ``effort`` on the wire."""
    cfg = _cfg(
        model_name="claude-haiku-5-5",
        anthropic={"omit_sampling_params": True, "thinking": "disabled", "effort": "low"},
    )
    gw, bodies = wire(cfg)
    await gw.start()
    goal = await gw.translate_mission("turn left")

    assert gw.is_degraded is False
    assert goal == GoalVector(vx_target=0.0, vy_target=0.0, omega_target=0.5)
    body = bodies[-1]
    assert body["model"] == "claude-haiku-5-5"
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == cfg.max_tokens
    assert body["messages"] == [{"role": "user", "content": "turn left"}]
