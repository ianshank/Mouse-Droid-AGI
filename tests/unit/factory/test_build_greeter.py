"""Factory tests for ``build_greeter``.

Opt-in semantics: raises ``ValueError`` when greeting is unconfigured /
disabled OR when the voice engine cannot be built. Test seam (a
pre-built ``voice_engine``) verified so the unit suite never depends
on the real Piper/USB stack.
"""

from __future__ import annotations

import pytest
from tests.unit.voice._fakes import FakeVoiceEngine

from mousedroid.config.schema import GreetingConfig, Settings
from mousedroid.factory import build_greeter
from mousedroid.voice.greeting import Greeter


def _enabled_settings() -> Settings:
    """Build an enabled greeting Settings via the Pydantic constructor.

    Constructs through ``Settings(...)`` so every Pydantic validator
    runs (including any future ``@model_validator`` added to
    ``Settings`` itself). Direct attribute mutation after construction
    would silently bypass those validators — a hygiene gap flagged by
    the round-2 code review.
    """
    return Settings(
        mock_hardware=True,
        greeting=GreetingConfig(enabled=True, names=["A", "B"]),
    )


def _enabled_settings_with_voice_disabled() -> Settings:
    """Enabled greeting + voice disabled — the no-voice-engine path.

    Constructed via :meth:`Settings.model_validate` so the nested
    voice override flows through the canonical validation path.
    """
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "greeting": {"enabled": True, "names": ["A", "B"]},
            "voice": {"enabled": False},
        }
    )


def test_raises_when_greeting_is_none() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.greeting is None
    with pytest.raises(ValueError, match="GreetingConfig with enabled=True"):
        build_greeter(cfg)


def test_raises_when_greeting_disabled() -> None:
    cfg = Settings(mock_hardware=True, greeting=GreetingConfig())  # enabled defaults False
    with pytest.raises(ValueError, match="enabled=True"):
        build_greeter(cfg)


def test_returns_greeter_when_enabled_and_voice_engine_provided() -> None:
    cfg = _enabled_settings()
    greeter = build_greeter(cfg, voice_engine=FakeVoiceEngine())  # type: ignore[arg-type]
    assert isinstance(greeter, Greeter)


def test_raises_when_voice_engine_cannot_be_built() -> None:
    """No voice engine path → operator-actionable error, not a None return."""
    cfg = _enabled_settings_with_voice_disabled()
    with pytest.raises(ValueError, match="voice engine"):
        build_greeter(cfg)  # no voice_engine override either


def test_voice_engine_test_seam_used_when_provided() -> None:
    """The test seam bypasses the build_voice_engine factory chain."""
    # Even if voice were disabled, an explicit voice_engine override wins.
    cfg = _enabled_settings_with_voice_disabled()
    engine = FakeVoiceEngine()
    greeter = build_greeter(cfg, voice_engine=engine)  # type: ignore[arg-type]
    # The greeter must hold the *exact* engine we passed in. Read via the
    # public ``voice_engine`` property (round-1 finding #3) rather than
    # piercing the private attribute.
    assert greeter.voice_engine is engine
