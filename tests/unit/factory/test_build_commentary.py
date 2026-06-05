"""Unit tests for build_commentary + build_commentary_composer."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mousedroid.commentary.composers import LLMCommentaryComposer, TemplateCommentaryComposer
from mousedroid.commentary.protocol import CommentaryEngineProtocol
from mousedroid.config.schema import CommentaryConfig, Settings, VoiceConfig
from mousedroid.factory import build_commentary, build_commentary_composer
from mousedroid.llm_gateway.protocol import QueryCapableLLMProtocol
from mousedroid.voice.protocol import VoiceEngineProtocol


def _settings(**commentary: object) -> Settings:
    return Settings(mock_hardware=True, commentary=CommentaryConfig(enabled=True, **commentary))  # type: ignore[arg-type]


def _voice() -> AsyncMock:
    return AsyncMock(spec=VoiceEngineProtocol)


def _query_gateway() -> AsyncMock:
    return AsyncMock(spec=QueryCapableLLMProtocol)


# --------------------------------------------------------------------------- #
# build_commentary
# --------------------------------------------------------------------------- #
def test_none_when_commentary_absent() -> None:
    assert build_commentary(Settings(mock_hardware=True)) is None


def test_none_when_disabled() -> None:
    cfg = Settings(mock_hardware=True, commentary=CommentaryConfig(enabled=False))
    assert build_commentary(cfg) is None


def test_template_engine_built() -> None:
    eng = build_commentary(_settings(composer="template"), voice_engine=_voice())
    assert isinstance(eng, CommentaryEngineProtocol)


def test_none_when_voice_unavailable() -> None:
    cfg = Settings(
        mock_hardware=True,
        voice=VoiceConfig(enabled=False),
        commentary=CommentaryConfig(enabled=True, composer="template"),
    )
    assert build_commentary(cfg, voice_engine=None) is None


def test_none_when_llm_composer_has_no_gateway() -> None:
    eng = build_commentary(_settings(composer="llm"), voice_engine=_voice(), gateway=None)
    assert eng is None


# --------------------------------------------------------------------------- #
# build_commentary_composer
# --------------------------------------------------------------------------- #
def test_composer_none_when_disabled() -> None:
    assert build_commentary_composer(Settings(mock_hardware=True)) is None


def test_composer_template() -> None:
    comp = build_commentary_composer(_settings(composer="template"))
    assert isinstance(comp, TemplateCommentaryComposer)


def test_composer_llm_with_gateway() -> None:
    comp = build_commentary_composer(_settings(composer="llm"), gateway=_query_gateway())
    assert isinstance(comp, LLMCommentaryComposer)


def test_composer_llm_without_gateway_returns_none() -> None:
    assert build_commentary_composer(_settings(composer="llm"), gateway=None) is None


def test_composer_auto_with_query_capable_gateway_is_llm() -> None:
    comp = build_commentary_composer(_settings(composer="auto"), gateway=_query_gateway())
    assert isinstance(comp, LLMCommentaryComposer)


def test_composer_auto_without_gateway_is_template() -> None:
    comp = build_commentary_composer(_settings(composer="auto"), gateway=None)
    assert isinstance(comp, TemplateCommentaryComposer)


def test_dependency_warning_when_no_novelty_source() -> None:
    """P3: enabled + statistical gate + memory disabled -> warning logged."""
    from structlog.testing import capture_logs

    cfg = _settings(composer="template", allow_without_novelty=False)
    # memory.enabled defaults False -> curiosity module won't be built.
    assert cfg.memory.enabled is False
    with capture_logs() as logs:
        build_commentary(cfg, voice_engine=_voice())
    assert any(e.get("event") == "commentary_enabled_without_novelty_source" for e in logs)
