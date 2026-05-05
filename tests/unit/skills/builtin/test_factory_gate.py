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


# ---------------------------------------------------------------------------
# Regression: build_injection_filter must check ``enabled``, not just presence
# ---------------------------------------------------------------------------


def test_build_injection_filter_uses_llm_cap_when_openclaw_disabled_with_block() -> None:
    """REGRESSION: Devin Review #BUG_..._0002.

    A YAML block like ``openclaw: {enabled: false, max_command_len: 128}``
    must NOT make the LLM gateway's injection filter use 128 — it must
    keep the historical ``cfg.llm.max_command_len`` (default 512).
    """
    from mousedroid.factory import build_injection_filter

    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "openclaw": {"enabled": False, "max_command_len": 128},
        }
    )
    f = build_injection_filter(s)
    # Disabled openclaw → fall back to llm.max_command_len (default 512).
    assert f.max_len == s.llm.max_command_len


def test_build_injection_filter_uses_openclaw_cap_when_enabled() -> None:
    """When OpenClaw IS enabled, the dispatcher's cap takes precedence."""
    from mousedroid.factory import build_injection_filter

    s = Settings.model_validate(
        {
            "mock_hardware": True,
            "openclaw": {"enabled": True, "max_command_len": 128},
        }
    )
    f = build_injection_filter(s)
    assert f.max_len == 128


def test_build_injection_filter_falls_back_when_openclaw_absent() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    from mousedroid.factory import build_injection_filter

    assert s.openclaw is None
    f = build_injection_filter(s)
    assert f.max_len == s.llm.max_command_len
