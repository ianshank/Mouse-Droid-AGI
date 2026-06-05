"""Unit tests for LLMCommentaryComposer (grounded prompt + degrade-safe)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mousedroid.commentary.composers import LLMCommentaryComposer
from mousedroid.commentary.protocol import CommentaryFacts
from mousedroid.config.schema import CommentaryConfig
from mousedroid.llm_gateway.protocol import QueryCapableLLMProtocol
from mousedroid.security.injection_filter import InjectionRejected

# A denylist of object nouns that MUST NOT appear in a grounded prompt — the
# mouse-droid loop has no object labels, so the prompt must stay label-free.
_OBJECT_NOUNS = ("chair", "table", "person", "dog", "wall", "door", "box")


def _facts(*, lidar_valid: bool = True, audio_valid: bool = True) -> CommentaryFacts:
    return CommentaryFacts(
        min_clearance_m=0.3,
        forward_distance_m=0.3,
        audio_rms=0.4,
        speed_mps=0.05,
        turn_rate=0.0,
        battery_v=10.8,
        novelty=1.23,
        is_emergency=False,
        lidar_valid=lidar_valid,
        audio_valid=audio_valid,
        timestamp=0.0,
    )


@pytest.fixture
def cfg() -> CommentaryConfig:
    return CommentaryConfig(enabled=True, composer="llm", max_words=5)


@pytest.mark.asyncio
async def test_happy_path_grounded_prompt_no_labels(cfg: CommentaryConfig) -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    gw.answer_query = AsyncMock(return_value="Tight squeeze here, Rocky careful now")
    out = await LLMCommentaryComposer(gw, cfg).compose(_facts())
    # Truncated to max_words=5.
    assert out == "Tight squeeze here, Rocky careful"
    prompt = gw.answer_query.call_args.args[0]
    assert "{facts}" not in prompt  # placeholder filled
    for noun in _OBJECT_NOUNS:
        assert noun not in prompt.lower()


@pytest.mark.asyncio
async def test_empty_answer_returns_empty(cfg: CommentaryConfig) -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    gw.answer_query = AsyncMock(return_value="")
    assert await LLMCommentaryComposer(gw, cfg).compose(_facts()) == ""


@pytest.mark.asyncio
async def test_injection_rejected_returns_empty(cfg: CommentaryConfig) -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    gw.answer_query = AsyncMock(side_effect=InjectionRejected("nope"))
    assert await LLMCommentaryComposer(gw, cfg).compose(_facts()) == ""


@pytest.mark.asyncio
async def test_value_error_returns_empty(cfg: CommentaryConfig) -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    gw.answer_query = AsyncMock(side_effect=ValueError("empty"))
    assert await LLMCommentaryComposer(gw, cfg).compose(_facts()) == ""


@pytest.mark.asyncio
async def test_facts_string_uses_forward_distance_when_lidar_invalid(cfg: CommentaryConfig) -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    gw.answer_query = AsyncMock(return_value="ok")
    await LLMCommentaryComposer(gw, cfg).compose(_facts(lidar_valid=False, audio_valid=False))
    prompt = gw.answer_query.call_args.args[0]
    assert "forward distance" in prompt
    assert "sound level" not in prompt  # audio invalid -> omitted
