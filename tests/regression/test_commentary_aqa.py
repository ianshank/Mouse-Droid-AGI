"""AQA: commentary protocol conformance + metric label hygiene."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mousedroid.commentary.composers import LLMCommentaryComposer, TemplateCommentaryComposer
from mousedroid.commentary.engine import SUPPRESSION_REASONS, CommentaryEngine
from mousedroid.commentary.protocol import (
    CommentaryComposerProtocol,
    CommentaryEngineProtocol,
)
from mousedroid.config.schema import CommentaryConfig
from mousedroid.llm_gateway.protocol import QueryCapableLLMProtocol
from mousedroid.telemetry import metrics as metrics_module
from mousedroid.voice.protocol import VoiceEngineProtocol


def _cfg() -> CommentaryConfig:
    return CommentaryConfig(enabled=True)


def test_template_composer_satisfies_protocol() -> None:
    assert isinstance(TemplateCommentaryComposer(_cfg()), CommentaryComposerProtocol)


def test_llm_composer_satisfies_protocol() -> None:
    gw = AsyncMock(spec=QueryCapableLLMProtocol)
    assert isinstance(LLMCommentaryComposer(gw, _cfg()), CommentaryComposerProtocol)


def test_engine_satisfies_protocol() -> None:
    eng = CommentaryEngine(
        _cfg(),
        voice_engine=AsyncMock(spec=VoiceEngineProtocol),
        composer=TemplateCommentaryComposer(_cfg()),
    )
    assert isinstance(eng, CommentaryEngineProtocol)


def test_suppression_reason_sets_match() -> None:
    """The engine's reasons and the metrics drop-guard set are identical."""
    assert SUPPRESSION_REASONS == metrics_module._COMMENTARY_SUPPRESS_REASONS


def test_no_object_label_vocabulary_in_default_templates() -> None:
    """Default templates phrase over the situation, never naming objects."""
    nouns = ("chair", "table", "person", "dog", "wall", "door", "box", "ball")
    cfg = CommentaryConfig()
    for phrase in cfg.templates.values():
        lowered = phrase.lower()
        assert not any(noun in lowered for noun in nouns), phrase
