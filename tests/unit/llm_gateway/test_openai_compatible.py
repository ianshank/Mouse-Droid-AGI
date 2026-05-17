"""Tier C2.3: OpenAICompatibleLLMGateway HTTP backend tests."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from mousedroid.config.schema import LLMConfig
from mousedroid.llm_gateway.openai_compatible import OpenAICompatibleLLMGateway
from mousedroid.llm_gateway.protocol import GoalVector


def _config(**overrides: object) -> LLMConfig:
    base: dict[str, object] = {
        "backend": "openai_compatible",
        "base_url": "http://127.0.0.1:11434",
    }
    base.update(overrides)
    return LLMConfig(**base)  # type: ignore[arg-type]


def _async_context_manager(value: object) -> MagicMock:
    """Build a MagicMock that behaves like ``async with x: ...`` yielding ``value``."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=value)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_start_marks_ready_when_models_endpoint_returns_200() -> None:
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_response = MagicMock()
    fake_response.status = 200
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=_async_context_manager(fake_response))
    with patch.object(gw, "_build_session", return_value=fake_session):
        await gw.start()
    assert gw.is_ready is True
    assert gw.is_degraded is False


@pytest.mark.asyncio
async def test_start_marks_degraded_on_connection_error() -> None:
    import aiohttp

    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_session = MagicMock()
    fake_session.get = MagicMock(
        side_effect=aiohttp.ClientConnectionError("connection refused")
    )
    with patch.object(gw, "_build_session", return_value=fake_session):
        await gw.start()
    assert gw.is_ready is False
    assert gw.is_degraded is True


@pytest.mark.asyncio
async def test_start_no_op_when_disabled() -> None:
    cfg = _config(enabled=False)
    gw = OpenAICompatibleLLMGateway(cfg)
    await gw.start()  # must not raise even though no patches are in place
    assert gw.is_ready is False
    assert gw.is_degraded is False


@pytest.mark.asyncio
async def test_translate_mission_parses_goal_vector_from_chat_response() -> None:
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"vx_target": 0.5, "vy_target": 0.0, "omega_target": 0.1}
                    )
                }
            }
        ]
    }
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.json = AsyncMock(return_value=payload)
    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=_async_context_manager(fake_response))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    goal = await gw.translate_mission("navigate to charger")
    assert goal == GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.1)


@pytest.mark.asyncio
async def test_translate_mission_returns_neutral_goal_on_timeout() -> None:
    cfg = _config(request_timeout_s=0.01)
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_session = MagicMock()
    fake_session.post = MagicMock(side_effect=asyncio.TimeoutError("LLM slow"))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    goal = await gw.translate_mission("explore")
    assert goal == GoalVector()  # neutral fallback, never raises


@pytest.mark.asyncio
async def test_translate_mission_returns_neutral_goal_on_non_json_content() -> None:
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    payload = {"choices": [{"message": {"content": "I'm a chatty LLM not JSON"}}]}
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.json = AsyncMock(return_value=payload)
    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=_async_context_manager(fake_response))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    goal = await gw.translate_mission("explore")
    assert goal == GoalVector()


@pytest.mark.asyncio
async def test_translate_mission_returns_neutral_goal_when_not_ready() -> None:
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    # No session, _ready stays False — adapter should short-circuit.
    goal = await gw.translate_mission("explore")
    assert goal == GoalVector()


@pytest.mark.asyncio
async def test_api_key_forwarded_as_bearer_header() -> None:
    cfg = _config(api_key=SecretStr("sk-test"))
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.json = AsyncMock(
        return_value={"choices": [{"message": {"content": "{}"}}]}
    )
    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=_async_context_manager(fake_response))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    await gw.translate_mission("explore")
    _args, kwargs = fake_session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_stop_closes_session() -> None:
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_session = MagicMock()
    fake_session.close = AsyncMock()
    gw._session = fake_session  # type: ignore[attr-defined]
    await gw.stop()
    fake_session.close.assert_awaited_once()
    assert gw.is_ready is False


# ---------------------------------------------------------------------------
# Coverage-gap closure — non-200 + timeout + malformed body branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_marks_degraded_on_non_200_health_response() -> None:
    """``GET /v1/models`` returning 503 marks the gateway degraded."""
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_response = MagicMock()
    fake_response.status = 503
    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=_async_context_manager(fake_response))
    with patch.object(gw, "_build_session", return_value=fake_session):
        await gw.start()
    assert gw.is_ready is False
    assert gw.is_degraded is True


@pytest.mark.asyncio
async def test_start_marks_degraded_on_health_check_timeout() -> None:
    """``GET /v1/models`` timing out marks the gateway degraded."""
    cfg = _config(request_timeout_s=0.01)
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_session = MagicMock()
    fake_session.get = MagicMock(side_effect=asyncio.TimeoutError("slow"))
    with patch.object(gw, "_build_session", return_value=fake_session):
        await gw.start()
    assert gw.is_ready is False
    assert gw.is_degraded is True


@pytest.mark.asyncio
async def test_translate_mission_returns_neutral_goal_on_non_200_response() -> None:
    """Server returning 500 → translate_mission yields a neutral GoalVector."""
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_response = MagicMock()
    fake_response.status = 500
    fake_response.json = AsyncMock(return_value={})
    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=_async_context_manager(fake_response))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    goal = await gw.translate_mission("explore")
    assert goal == GoalVector()


@pytest.mark.asyncio
async def test_translate_mission_returns_neutral_goal_on_malformed_body() -> None:
    """Missing ``choices`` key → neutral GoalVector, no exception bubble."""
    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.json = AsyncMock(return_value={"error": "no choices field"})
    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=_async_context_manager(fake_response))
    gw._session = fake_session  # type: ignore[attr-defined]
    gw._ready = True  # type: ignore[attr-defined]

    goal = await gw.translate_mission("explore")
    assert goal == GoalVector()


@pytest.mark.asyncio
async def test_build_session_constructs_real_aiohttp_session() -> None:
    """``_build_session`` returns a live ``aiohttp.ClientSession`` (covers L117-119).

    ``aiohttp.ClientSession()`` requires a running event loop, so this
    test must be ``async``. We close the session immediately to avoid
    leaking sockets in the test process.
    """
    import aiohttp

    cfg = _config()
    gw = OpenAICompatibleLLMGateway(cfg)
    session = gw._build_session()  # noqa: SLF001
    try:
        assert isinstance(session, aiohttp.ClientSession)
    finally:
        await session.close()
