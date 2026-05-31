"""PR #107 backwards-compatibility regression tests.

PR #107 added three new ``LLMConfig`` fields for the Anthropic Claude gateway
+ cloud/local failover composite. They MUST inherit safe defaults so existing
YAML configs (rover deployments, CI fixtures, the operator's
``~/.config/mousedroid/`` overlay) keep loading byte-identically after a
``git pull``:

* ``LLMConfig.backend`` — Literal gains ``"anthropic"`` but still defaults to
  ``"llama_cpp"`` (the pre-PR value).
* ``LLMConfig.fallback_backend`` — failover secondary; default ``"none"``
  (disabled, single-backend behaviour preserved).
* ``LLMConfig.fallback_model_name`` — secondary model override; default
  ``None`` (reuse ``model_name``).
* ``LLMConfig.fallback_retry_cooldown_s`` — degraded-primary re-probe window;
  default ``30.0``.

These tests pin the invariant from the project CLAUDE.md:

    > **9. Backwards compatibility**: New config fields MUST have defaults.
    > Existing YAML files must load unchanged.

A failure here means a rover that ``git pull``ed would refuse to start or
silently switch behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import LLMConfig, Settings


def _repo_root() -> Path:
    """Locate the worktree root for loading ``config/*.yaml`` fixtures."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Default-value invariants — these must NEVER silently change
# ---------------------------------------------------------------------------
def test_backend_default_unchanged() -> None:
    """The default backend stays ``llama_cpp`` (pre-PR-107 behaviour)."""
    assert LLMConfig().backend == "llama_cpp"


def test_fallback_backend_defaults_to_none() -> None:
    assert LLMConfig().fallback_backend == "none"


def test_fallback_model_name_defaults_to_none() -> None:
    assert LLMConfig().fallback_model_name is None


def test_fallback_retry_cooldown_defaults_to_30s() -> None:
    assert LLMConfig().fallback_retry_cooldown_s == 30.0


# ---------------------------------------------------------------------------
# Legacy-YAML load semantics — a pre-PR-107 llm block must load unchanged and
# pick up the new fields at their documented defaults
# ---------------------------------------------------------------------------
def test_legacy_llm_yaml_loads_with_new_field_defaults(tmp_path: Path) -> None:
    """A pre-PR-107 ``llm:`` block (no fallback keys) loads with safe defaults."""
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "platform: mouse_droid\n"
        "mock_hardware: true\n"
        "llm:\n"
        "  enabled: true\n"
        "  model_path: /opt/mousedroid/models/llama-3-8b-instruct.Q4_K_M.gguf\n"
        "  context_length: 2048\n"
        "  max_tokens: 256\n"
        "  temperature: 0.1\n",
    )
    settings = load_settings(legacy)
    # Pre-existing fields preserved.
    assert settings.llm.backend == "llama_cpp"
    assert settings.llm.enabled is True
    assert settings.llm.max_tokens == 256
    # New fields default to their safe (disabled) values.
    assert settings.llm.fallback_backend == "none"
    assert settings.llm.fallback_model_name is None
    assert settings.llm.fallback_retry_cooldown_s == 30.0


def test_default_yaml_overlay_still_loads() -> None:
    """The shipped ``config/default.yaml`` loads unchanged after PR #107."""
    settings = load_settings(_repo_root() / "config" / "default.yaml")
    assert isinstance(settings, Settings)
    # default.yaml predates the failover fields → they take schema defaults.
    assert settings.llm.fallback_backend == "none"
    assert settings.llm.fallback_retry_cooldown_s == 30.0


def test_new_anthropic_pilot_overlay_loads() -> None:
    """The new pilot overlay loads and round-trips the failover wiring."""
    settings = load_settings(_repo_root() / "config" / "jetson_claude_pilot.yaml")
    assert settings.llm.backend == "anthropic"
    assert settings.llm.fallback_backend == "llama_cpp"
    assert settings.llm.fallback_retry_cooldown_s == 30.0


def test_cooldown_must_stay_positive() -> None:
    """Guard the ``gt=0`` constraint so a 0/negative cooldown is rejected."""
    with pytest.raises(ValueError, match="fallback_retry_cooldown_s"):
        LLMConfig(fallback_retry_cooldown_s=0.0)
