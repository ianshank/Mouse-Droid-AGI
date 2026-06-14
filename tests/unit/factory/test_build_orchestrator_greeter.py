"""Factory wiring tests for the orchestrator startup greeter (Issue #109).

``build_orchestrator`` must wire a :class:`Greeter` onto the orchestrator
when ``cfg.greeting`` is enabled (reusing the orchestrator's own voice
engine), and ``None`` otherwise. The standalone
``_build_orchestrator_greeter`` helper is unit-tested directly so the
wiring contract is pinned without standing up the full orchestrator.
"""

from __future__ import annotations

from tests.unit.voice._fakes import FakeVoiceEngine

from mousedroid.config.schema import GreetingConfig, Settings
from mousedroid.factory import _build_orchestrator_greeter
from mousedroid.voice.greeting import Greeter


def test_greeter_wired_when_greeting_enabled() -> None:
    cfg = Settings(
        mock_hardware=True,
        greeting=GreetingConfig(enabled=True, names=["A", "B"]),
    )
    engine = FakeVoiceEngine()
    greeter = _build_orchestrator_greeter(cfg, engine)  # type: ignore[arg-type]
    assert isinstance(greeter, Greeter)
    # Reuses the orchestrator's already-built engine — no second engine.
    assert greeter.voice_engine is engine


def test_greeter_none_when_greeting_config_none() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.greeting is None
    assert _build_orchestrator_greeter(cfg, FakeVoiceEngine()) is None  # type: ignore[arg-type]


def test_greeter_none_when_greeting_disabled() -> None:
    cfg = Settings(mock_hardware=True, greeting=GreetingConfig(enabled=False))
    assert _build_orchestrator_greeter(cfg, FakeVoiceEngine()) is None  # type: ignore[arg-type]


def test_greeter_none_when_no_voice_engine() -> None:
    """No voice engine (voice disabled) → no greeter, no crash."""
    cfg = Settings(
        mock_hardware=True,
        greeting=GreetingConfig(enabled=True, names=["A"]),
    )
    assert _build_orchestrator_greeter(cfg, None) is None
