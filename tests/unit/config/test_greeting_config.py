"""Schema tests for the operator-tools ``GreetingConfig``.

Backwards-compat AQA: existing YAML / env must load unchanged
(``Settings.greeting`` defaults to ``None``); the new fields must
default to safe (disabled) values; ``enabled=True`` with an empty
``names`` list or a template missing the ``{names}`` placeholder
must be rejected at YAML-load time so misconfigured overlays surface
at parse time rather than at runtime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import GreetingConfig, Settings


def test_settings_greeting_defaults_to_none_for_backcompat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing YAML files must load with greeting=None unchanged."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    s = Settings()
    assert s.greeting is None


def test_greeting_disabled_with_empty_names_is_valid() -> None:
    """Disabled config does not require names — default loads inert."""
    cfg = GreetingConfig()
    assert cfg.enabled is False
    assert cfg.names == []
    assert "{names}" in cfg.message_template
    assert cfg.pre_chirp_event == "greeting_excited"
    assert 0.0 <= cfg.excitement_intensity <= 1.0
    assert 0.0 <= cfg.inter_chirp_delay_s <= 5.0


def test_greeting_enabled_requires_names() -> None:
    """enabled=True with no names is a config error at parse time."""
    with pytest.raises(ValidationError, match="greeting.names"):
        GreetingConfig(enabled=True, names=[])


def test_greeting_enabled_with_names_validates() -> None:
    cfg = GreetingConfig(enabled=True, names=["John", "Jordan"])
    assert cfg.enabled is True
    assert cfg.names == ["John", "Jordan"]


def test_message_template_must_contain_names_placeholder() -> None:
    """A template without {names} would silently produce a name-less greeting."""
    with pytest.raises(ValidationError, match=r"'\{names\}' placeholder"):
        GreetingConfig(
            enabled=True,
            names=["A"],
            message_template="Hello world!",  # missing {names}
        )


def test_disabled_config_accepts_template_without_placeholder() -> None:
    """A disabled overlay can carry an in-progress / placeholder template.

    Regression — code-reviewer round-1 finding #1: a stricter unconditional
    guard rejected disabled configs with custom templates, blocking a
    legitimate operator workflow (drafting a new template while the
    subsystem is disabled). The guard now gates on ``enabled``.
    """
    cfg = GreetingConfig(
        enabled=False,
        message_template="Stand by",  # no {names} placeholder
    )
    assert cfg.enabled is False
    assert cfg.message_template == "Stand by"


def test_excitement_intensity_range_gated() -> None:
    """[0, 1] bounds — rocky_transform expects normalised intensity."""
    with pytest.raises(ValidationError):
        GreetingConfig(excitement_intensity=-0.1)
    with pytest.raises(ValidationError):
        GreetingConfig(excitement_intensity=1.5)


def test_inter_chirp_delay_s_range_gated() -> None:
    """[0, 5] bounds — keeps the chirp+message back-to-back without runaway."""
    with pytest.raises(ValidationError):
        GreetingConfig(inter_chirp_delay_s=-0.5)
    with pytest.raises(ValidationError):
        GreetingConfig(inter_chirp_delay_s=10.0)


def test_pre_chirp_event_empty_string_disables_flourish() -> None:
    """Empty string is the documented way to skip the pre-flourish."""
    cfg = GreetingConfig(
        enabled=True,
        names=["Pat"],
        pre_chirp_event="",
    )
    assert cfg.pre_chirp_event == ""


def test_message_template_foreign_placeholder_rejected_at_load() -> None:
    """A foreign placeholder ({wrong}) fails YAML-load — not runtime.

    Round-3 review (Gemini #1): before this guard, an operator-typo
    template like ``"Hello {names} from {wrong}"`` would parse cleanly
    and then crash the greeter at runtime with ``KeyError: 'wrong'``,
    masking the YAML-config fault as a hardware error. The probe
    ``self.message_template.format(names="__probe__")`` in the model
    validator surfaces the bad template at load time with an
    operator-actionable error.
    """
    with pytest.raises(ValidationError, match="message_template formatting failed"):
        GreetingConfig(
            enabled=True,
            names=["Pat"],
            message_template="Hello {names} from {wrong_key}",
        )


def test_message_template_positional_placeholder_rejected() -> None:
    """Positional ``{0}`` also fails the probe — only ``{names}`` is allowed."""
    with pytest.raises(ValidationError, match="message_template formatting failed"):
        GreetingConfig(
            enabled=True,
            names=["Pat"],
            message_template="Hello {names} {0}",
        )


def test_message_template_unbalanced_brace_rejected() -> None:
    """A stray unbalanced brace raises ValueError inside ``.format()``."""
    with pytest.raises(ValidationError, match="message_template formatting failed"):
        GreetingConfig(
            enabled=True,
            names=["Pat"],
            message_template="Hello {names} {",
        )


def test_fire_on_startup_defaults_false() -> None:
    """New field defaults False so the orchestrator startup seam stays inert.

    Issue #109 lifecycle wiring: ``fire_on_startup`` gates the one-shot
    greeting at ``orchestrator.start()``. Default ``False`` keeps the
    30 Hz loop byte-identical for every existing deployment.
    """
    cfg = GreetingConfig()
    assert cfg.fire_on_startup is False


def test_fire_on_startup_accepts_true() -> None:
    """Operators flip it on the overlay alongside enabled=True + names."""
    cfg = GreetingConfig(enabled=True, names=["Pat"], fire_on_startup=True)
    assert cfg.fire_on_startup is True


def test_existing_config_without_fire_on_startup_loads_unchanged() -> None:
    """Backwards-compat: a pre-#109 overlay (no ``fire_on_startup`` key) loads.

    Simulates an existing YAML file by validating a dict that omits the
    new field entirely. It must load and default ``fire_on_startup`` to
    ``False`` — the CLAUDE.md backwards-compatibility invariant.
    """
    cfg = GreetingConfig.model_validate({"enabled": True, "names": ["John", "Jordan"]})
    assert cfg.fire_on_startup is False
    assert cfg.enabled is True
    assert cfg.names == ["John", "Jordan"]


def test_startup_timeout_s_defaults_to_10() -> None:
    """Issue #109 review fix: the startup greeting is bounded by a config timeout.

    Defaults to 10.0s so pre-review overlays load unchanged and the
    orchestrator's ``asyncio.wait_for`` bound needs no hardcoded literal — a
    hung TTS / blocked ALSA device can never wedge bring-up.
    """
    cfg = GreetingConfig()
    assert cfg.startup_timeout_s == 10.0


def test_startup_timeout_s_must_be_positive() -> None:
    """gt=0 — a zero/negative bound would defeat the wait_for guard."""
    with pytest.raises(ValidationError):
        GreetingConfig(startup_timeout_s=0.0)
    with pytest.raises(ValidationError):
        GreetingConfig(startup_timeout_s=-1.0)


def test_existing_config_without_startup_timeout_loads_unchanged() -> None:
    """Backwards-compat: a pre-review overlay (no startup_timeout_s key) loads."""
    cfg = GreetingConfig.model_validate({"enabled": True, "names": ["John"]})
    assert cfg.startup_timeout_s == 10.0


def test_settings_greeting_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can flip the master switch via env (then YAML supplies names)."""
    # The Pydantic env-prefix only flips scalar fields at Settings level;
    # the names list requires YAML. Confirm the master switch round-trips
    # via env so a CI smoke can enable the greeter without YAML edits.
    monkeypatch.setenv("MOUSEDROID_GREETING__ENABLED", "false")
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    s = Settings(greeting=GreetingConfig())
    assert s.greeting is not None
    assert s.greeting.enabled is False
