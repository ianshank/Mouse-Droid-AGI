"""Verify ``build_builtin_skills`` and ``build_skill_registry`` honour openclaw gating."""

from __future__ import annotations

from mousedroid.config.schema import OpenClawConfig, Settings
from mousedroid.factory import build_builtin_skills, build_skill_registry


def _settings(*, openclaw: OpenClawConfig | None) -> Settings:
    return Settings.model_validate(
        {"mock_hardware": True, "openclaw": openclaw.model_dump() if openclaw else None}
    )


def test_returns_empty_tuple_when_openclaw_disabled() -> None:
    cfg = _settings(openclaw=None)
    assert build_builtin_skills(cfg) == ()


def test_returns_empty_tuple_when_openclaw_enabled_false() -> None:
    cfg = _settings(openclaw=OpenClawConfig(enabled=False))
    assert build_builtin_skills(cfg) == ()


def test_returns_four_specs_when_openclaw_enabled_true() -> None:
    cfg = _settings(openclaw=OpenClawConfig(enabled=True))
    specs = build_builtin_skills(cfg)
    assert len(specs) == 4
    assert {s.name for s in specs} == {
        "mousedroid-navigate",
        "mousedroid-sensor-report",
        "mousedroid-voice",
        "mousedroid-world-model",
    }


def test_skill_registry_picks_up_builtins_when_enabled() -> None:
    cfg = _settings(openclaw=OpenClawConfig(enabled=True))
    registry = build_skill_registry(cfg)
    assert len(registry) == 4
    assert "mousedroid-navigate" in registry.names()


def test_skill_registry_empty_when_openclaw_disabled() -> None:
    cfg = _settings(openclaw=None)
    registry = build_skill_registry(cfg)
    assert len(registry) == 0
