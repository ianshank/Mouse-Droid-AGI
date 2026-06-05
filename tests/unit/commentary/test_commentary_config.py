"""Unit tests for CommentaryConfig (backwards-compat + validators)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import CommentaryConfig, Settings


def test_settings_commentary_defaults_to_none() -> None:
    """Existing YAML / env load with commentary absent (backwards-compat)."""
    assert Settings(mock_hardware=True).commentary is None


def test_disabled_defaults_in_range() -> None:
    cfg = CommentaryConfig()
    assert cfg.enabled is False
    assert cfg.composer == "auto"
    assert cfg.novelty_sigma == 2.5
    assert cfg.observe_stride >= 1
    assert "default" in cfg.templates


def test_disabled_overlay_skips_cross_field_validation() -> None:
    """A disabled overlay may carry in-progress / invalid values."""
    cfg = CommentaryConfig(enabled=False, open_space_m=0.1, tight_space_m=0.4)
    assert cfg.enabled is False  # no raise


def test_enabled_rejects_open_le_tight() -> None:
    with pytest.raises(ValidationError, match="open_space_m must exceed"):
        CommentaryConfig(enabled=True, open_space_m=0.4, tight_space_m=0.4)


def test_enabled_requires_facts_placeholder() -> None:
    with pytest.raises(ValidationError, match=r"\{facts\}"):
        CommentaryConfig(enabled=True, composer="llm", llm_prompt_template="no placeholder")


def test_enabled_template_composer_skips_facts_placeholder_check() -> None:
    """composer='template' never uses the LLM template, so {facts} not required."""
    cfg = CommentaryConfig(enabled=True, composer="template", llm_prompt_template="plain text")
    assert cfg.composer == "template"


def test_enabled_requires_default_template_key() -> None:
    with pytest.raises(ValidationError, match="'default' key"):
        CommentaryConfig(enabled=True, templates={"tight_space": "x"})


@pytest.mark.parametrize("bad_template", ["{wrong}", "{0}", "{facts} {unbalanced"])
def test_enabled_rejects_bad_llm_template_formatting(bad_template: str) -> None:
    with pytest.raises(ValidationError):
        CommentaryConfig(enabled=True, composer="llm", llm_prompt_template=bad_template)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("novelty_sigma", -0.1),
        ("novelty_sigma", 11.0),
        ("novelty_gate_alpha", 0.0),
        ("novelty_gate_alpha", 1.5),
        ("cadence_s", 0.0),
        ("observe_stride", 0),
        ("max_words", 0),
        ("excitement_intensity", 1.5),
        ("idle_min_clearance_m", 0.0),
    ],
)
def test_range_gates(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        CommentaryConfig(**{field: value})  # type: ignore[arg-type]


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env alone constructs the Optional nested model (no YAML / init-arg)."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    monkeypatch.setenv("MOUSEDROID_COMMENTARY__ENABLED", "true")
    monkeypatch.setenv("MOUSEDROID_COMMENTARY__NOVELTY_SIGMA", "3.0")
    s = Settings()
    assert s.commentary is not None
    assert s.commentary.enabled is True
    assert s.commentary.novelty_sigma == 3.0
