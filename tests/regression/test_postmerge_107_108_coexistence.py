"""Post-merge cross-feature backwards-compatibility regression tests.

PRs #107 (Anthropic Claude LLM gateway + cloud/local failover) and #108/#110
(MSE-6 named-greeting subsystem) landed independently and then merged into the
same trunk. Each shipped its own per-PR back-compat test
(``test_pr107_backwards_compat.py`` / ``test_pr108_backwards_compat.py``);
this module pins the *combined* invariant that the two opt-in subsystems
coexist on a single ``Settings`` object without interfering:

* A legacy YAML that predates BOTH features must still load, with each new
  block defaulting to its disabled/safe value.
* Enabling one subsystem must not perturb the other's defaults.
* The two features touch disjoint config blocks (``llm`` vs ``greeting``),
  so a change to one must never silently flip the other.

This is the regression that catches a future schema refactor which, say,
moves a shared default or reorders validators in a way that couples the two
previously-independent blocks. Per CLAUDE.md invariant 9 (backwards
compatibility): existing YAML must load unchanged after a ``git pull``.
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings


def _repo_root() -> Path:
    """Locate the worktree root for loading ``config/*.yaml`` fixtures."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Combined-default surface — both subsystems opt-in and independent
# ---------------------------------------------------------------------------
def test_both_subsystems_default_to_disabled_and_safe() -> None:
    """A bare ``Settings`` leaves greeting unset and the LLM on its legacy path."""
    settings = Settings(mock_hardware=True)
    # Greeting subsystem (PR #108) — opt-in, absent by default.
    assert settings.greeting is None
    # LLM gateway (PR #107) — legacy llama_cpp, failover disabled.
    assert settings.llm.backend == "llama_cpp"
    assert settings.llm.fallback_backend == "none"
    assert settings.llm.fallback_model_name is None
    assert settings.llm.fallback_retry_cooldown_s == 30.0


def test_legacy_yaml_predating_both_features_loads_unchanged(tmp_path: Path) -> None:
    """A pre-#107/#108 overlay validates with both new blocks at safe defaults."""
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "platform: mouse_droid\n"
        "mock_hardware: true\n"
        "llm:\n"
        "  enabled: true\n"
        "  max_tokens: 256\n",
    )
    settings = load_settings(legacy)
    # Pre-existing llm field preserved.
    assert settings.llm.max_tokens == 256
    # Both new surfaces defaulted.
    assert settings.greeting is None
    assert settings.llm.fallback_backend == "none"
    assert settings.llm.fallback_retry_cooldown_s == 30.0


def test_enabling_greeting_does_not_perturb_llm_defaults(tmp_path: Path) -> None:
    """Opting into the greeter leaves the LLM gateway block untouched."""
    overlay = tmp_path / "greet.yaml"
    overlay.write_text(
        "platform: mouse_droid\n"
        "mock_hardware: true\n"
        "greeting:\n"
        "  enabled: true\n"
        "  names: [Artoo, Threepio]\n",
    )
    settings = load_settings(overlay)
    assert settings.greeting is not None
    assert settings.greeting.enabled is True
    # LLM block must keep its independent defaults.
    assert settings.llm.backend == "llama_cpp"
    assert settings.llm.fallback_backend == "none"


def test_enabling_llm_failover_does_not_enable_greeting(tmp_path: Path) -> None:
    """Opting into cloud Claude + failover leaves the greeter disabled."""
    overlay = tmp_path / "llm.yaml"
    overlay.write_text(
        "platform: mouse_droid\n"
        "mock_hardware: true\n"
        "llm:\n"
        "  enabled: true\n"
        "  backend: anthropic\n"
        "  model_name: claude-haiku-4-5\n"
        "  fallback_backend: llama_cpp\n",
    )
    settings = load_settings(overlay)
    assert settings.llm.backend == "anthropic"
    assert settings.llm.fallback_backend == "llama_cpp"
    # Greeting must stay opt-out.
    assert settings.greeting is None


# ---------------------------------------------------------------------------
# Shipped overlays still load post-merge
# ---------------------------------------------------------------------------
def test_jetson_claude_pilot_overlay_loads() -> None:
    """PR #107/#111 pilot overlay loads with the failover wiring intact."""
    settings = load_settings(_repo_root() / "config" / "jetson_claude_pilot.yaml")
    assert settings.llm.backend == "anthropic"
    assert settings.llm.fallback_backend == "llama_cpp"


def test_default_overlay_loads_with_both_features_defaulted() -> None:
    """``config/default.yaml`` predates both features → safe defaults hold."""
    settings = load_settings(_repo_root() / "config" / "default.yaml")
    assert isinstance(settings, Settings)
    assert settings.greeting is None
    assert settings.llm.fallback_backend == "none"
