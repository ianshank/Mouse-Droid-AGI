"""Tests for the arm LLM-replanner backends."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.arm.planning.llm_replanners.anthropic_backend import (
    AnthropicReplanner,
    AnthropicSDKMissingError,
)
from mousedroid.arm.planning.llm_replanners.base import LLMReplannerProtocol
from mousedroid.arm.planning.llm_replanners.null_backend import NullLLMReplanner
from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.config.schema import LLMReplannerConfig


def _state(predicates: set[str] | None = None) -> SymbolicState:
    return SymbolicState(
        predicates=frozenset(predicates or {"on(disk_1, peg_A)"}),
        objects={"disk_1": "disk", "peg_A": "peg"},
    )


def _fake_sdk(response_text: str, *, with_async: bool = False) -> Any:
    """Build a minimal stand-in for the ``anthropic`` SDK.

    ``with_async=True`` also exposes an ``AsyncAnthropic`` class whose
    ``messages.create`` is awaitable, used by the ``areplan`` tests.
    """

    class _Block:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> _Response:
            self.calls.append(kwargs)
            return _Response(response_text)

    class _Client:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.messages = _Messages()

    class _AsyncMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> _Response:
            self.calls.append(kwargs)
            return _Response(response_text)

    class _AsyncClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.messages = _AsyncMessages()

    members: dict[str, Any] = {"Anthropic": _Client}
    if with_async:
        members["AsyncAnthropic"] = _AsyncClient
    sdk = SimpleNamespace(**members)
    return sdk


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------


def test_null_implements_protocol() -> None:
    r = NullLLMReplanner()
    assert isinstance(r, LLMReplannerProtocol)


def test_null_returns_empty_list() -> None:
    r = NullLLMReplanner()
    assert r.replan(_state(), _state(), "any error") == []


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


def test_anthropic_disabled_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=False)
    sdk = _fake_sdk('[{"action": "noop", "args": []}]')
    r = AnthropicReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "fail") == []


def test_anthropic_parses_json_response() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk(
        '[{"action": "move", "args": ["disk_1", "peg_A", "peg_C"]},'
        ' {"action": "open_gripper", "args": []}]'
    )
    r = AnthropicReplanner(cfg, sdk=sdk)
    steps = r.replan(_state(), _state({"on(disk_1, peg_C)"}), "grasp failed")
    assert len(steps) == 2
    assert isinstance(steps[0], PlanStep)
    assert steps[0].action == "move"
    assert steps[0].args == ["disk_1", "peg_A", "peg_C"]
    assert steps[1].action == "open_gripper"


def test_anthropic_non_json_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk("Sorry, I cannot help with that.")
    r = AnthropicReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "x") == []


def test_anthropic_non_list_response_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk('{"action": "move"}')
    r = AnthropicReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "x") == []


def test_anthropic_skips_malformed_steps() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk(
        '[{"action": "move", "args": ["a"]}, '
        '"not_a_dict", '
        '{"args": ["x"]}, '  # missing action
        '{"action": "ok", "args": "not_a_list"}]'
    )
    r = AnthropicReplanner(cfg, sdk=sdk)
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1
    assert steps[0].action == "move"


def test_anthropic_retries_on_exception() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic", max_retries=2)
    sdk = _fake_sdk('[{"action": "ok", "args": []}]')
    # Patch the messages.create on the constructed client to fail once,
    # then succeed on retry.
    r = AnthropicReplanner(cfg, sdk=sdk)
    real_create = r._client.messages.create
    failures = {"left": 1}

    def flaky(**kwargs: Any) -> Any:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("network")
        return real_create(**kwargs)

    r._client.messages.create = flaky  # type: ignore[assignment]
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1


def test_anthropic_exhausts_retries_and_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic", max_retries=1)
    sdk = _fake_sdk("does not matter")
    r = AnthropicReplanner(cfg, sdk=sdk)

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("always fails")

    r._client.messages.create = boom  # type: ignore[assignment]
    assert r.replan(_state(), _state(), "x") == []


def test_anthropic_passes_config_to_sdk() -> None:
    cfg = LLMReplannerConfig(
        enabled=True,
        backend="anthropic",
        model="custom-model",
        max_tokens=42,
        temperature=0.7,
        system_prompt="You are a robot.",
        request_timeout_s=12.5,
    )
    sdk = _fake_sdk("[]")
    r = AnthropicReplanner(cfg, sdk=sdk)
    r.replan(_state(), _state(), "fail")
    call = r._client.messages.calls[0]
    assert call["model"] == "custom-model"
    assert call["max_tokens"] == 42
    assert call["temperature"] == 0.7
    assert call["system"] == "You are a robot."
    assert call["timeout"] == 12.5


@pytest.mark.asyncio
async def test_anthropic_areplan_uses_async_client() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk('[{"action": "ok", "args": []}]', with_async=True)
    r = AnthropicReplanner(cfg, sdk=sdk)
    steps = await r.areplan(_state(), _state(), "fail")
    assert len(steps) == 1
    assert steps[0].action == "ok"
    # Verify the async client was actually used (not the sync one).
    async_client = r._async_client_cache
    assert async_client is not None
    assert len(async_client.messages.calls) == 1


@pytest.mark.asyncio
async def test_anthropic_areplan_disabled_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=False, backend="anthropic")
    sdk = _fake_sdk("[]", with_async=True)
    r = AnthropicReplanner(cfg, sdk=sdk)
    assert await r.areplan(_state(), _state(), "x") == []


@pytest.mark.asyncio
async def test_anthropic_areplan_raises_when_async_missing() -> None:
    """When the SDK lacks ``AsyncAnthropic``, ``areplan`` raises a clear error."""
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    sdk = _fake_sdk("[]", with_async=False)  # no AsyncAnthropic exposed
    r = AnthropicReplanner(cfg, sdk=sdk)
    with pytest.raises(AnthropicSDKMissingError, match="AsyncAnthropic"):
        await r.areplan(_state(), _state(), "x")


def test_anthropic_sdk_missing_raises() -> None:
    """Construction must raise when ``anthropic`` is unavailable."""
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")

    class _BadReplanner(AnthropicReplanner):
        def _import_sdk(self) -> Any:  # type: ignore[override]
            raise AnthropicSDKMissingError("sdk missing in test")

    with pytest.raises(AnthropicSDKMissingError):
        _BadReplanner(cfg)


# ---------------------------------------------------------------------------
# Replanner delegates to the injected backend
# ---------------------------------------------------------------------------


def test_replanner_delegates_to_injected_llm() -> None:
    from mousedroid.arm.planning.replanner import Replanner
    from mousedroid.config.schema import ArmPlanningConfig

    captured: list[Any] = []

    class _StubLLM:
        name = "stub"

        def replan(self, current_state: Any, goal_state: Any, error: str) -> list[PlanStep]:
            captured.append((current_state, goal_state, error))
            return [PlanStep(action="custom", args=[])]

    planner = MagicMock()
    planner.plan.return_value = [PlanStep(action="symbolic_fallback", args=[])]
    cfg = ArmPlanningConfig(llm_replanner_enabled=True, max_replan_attempts=3)
    replanner = Replanner(cfg, planner, llm_replanner=_StubLLM())
    out = replanner.replan(_state(), _state({"on(d, p)"}), "boom")
    assert out[0].action == "custom"
    assert len(captured) == 1


def test_replanner_falls_back_when_llm_returns_empty() -> None:
    from mousedroid.arm.planning.replanner import Replanner
    from mousedroid.config.schema import ArmPlanningConfig

    class _EmptyLLM:
        name = "empty"

        def replan(self, *_args: Any, **_kwargs: Any) -> list[PlanStep]:
            return []

    planner = MagicMock()
    planner.plan.return_value = [PlanStep(action="symbolic_fallback", args=[])]
    cfg = ArmPlanningConfig(llm_replanner_enabled=True, max_replan_attempts=3)
    replanner = Replanner(cfg, planner, llm_replanner=_EmptyLLM())
    out = replanner.replan(_state(), _state(), "boom")
    assert out[0].action == "symbolic_fallback"


def test_replanner_default_uses_null_llm_when_enabled_flag_set() -> None:
    """When ``llm_replanner_enabled`` is True but no explicit backend is
    supplied, the default :class:`NullLLMReplanner` returns ``[]`` and the
    outer ``Replanner`` falls back to the symbolic planner — preserving
    legacy behaviour without requiring the Anthropic SDK."""
    from mousedroid.arm.planning.replanner import Replanner
    from mousedroid.config.schema import ArmPlanningConfig

    planner = MagicMock()
    planner.plan.return_value = [PlanStep(action="legacy", args=[])]
    cfg = ArmPlanningConfig(llm_replanner_enabled=True, max_replan_attempts=3)
    replanner = Replanner(cfg, planner)
    out = replanner.replan(_state(), _state(), "boom")
    assert out[0].action == "legacy"
