"""Regression: cap inline type:ignore / noqa debt in src/.

Update the budgets DOWN as the purge lands; never up without justification.

Also budgets the ``[tool.ruff.lint.per-file-ignores]`` entries whose glob
targets ``src/`` — pyproject.toml documents that at least one suppression
(harness/spec.py S602) was moved to a file-level ignore precisely "to keep
the src/ noqa budget flat", so file-level waivers must count toward a budget
too or the inline budgets under-report real suppression debt.
"""

from __future__ import annotations

from pathlib import Path

from tools.claude_hooks.config import RatchetBudgetsConfig, load_config
from tools.ratchet_budgets import count_marker_occurrences

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "mousedroid"
_SCOPE_GLOB = "src/mousedroid/**/*.py"
# Budgets = the MEASURED post-purge residual of load-bearing suppressions
# (untyped 3rd-party boundaries, Pydantic A00x field shadows) PLUS the handful
# of prose mentions of the literal tokens inside explanatory comments/docstrings.
# Expected NON-ZERO. May only ratchet DOWN, never up without a documented reason.
#
# The ignore budget (8) = 7 live torch untyped-call directives
#   (Tensor.backward x5 — growth/distillation.py, learning/ewc.py,
#   learning/offline_rl.py, meta/maml.py, training/rssm_pretrainer.py;
#   torch.jit.trace/.load x2 — efficiency/tensorrt.py) + 1 prose mention
#   (voice/greeting.py docstring documenting a *removed* private-access ignore).
#   (The torch.amp.GradScaler attr-defined ignore was eliminated by importing
#   from torch.amp.grad_scaler directly — runtime-valid AND mypy-clean.)
# The lint-waiver budget (19) = 18 live directives (torch.nn.functional N812 x4
#   — curiosity/icm.py, growth/distillation.py, learning/offline_rl.py,
#   scaling/moe.py; InjectionRejected N818 — security/injection_filter.py;
#   non-crypto random S311 x3 — resilience/retry.py, voice/rocky.py,
#   telemetry/mock_source.py; fixed-http urllib S310 x4 — comms/wifi_driver.py;
#   Pydantic Field default paths S108/S104 x3 — config/schema.py;
#   Argus socket S108 — validation/runtime/_shared.py; watchdog systemd-notify
#   S603/S607 — health/watchdog.py) + 1 prose mention
#   (common/imports.py docstring).
#   (The 10 stale BLE001/RUF100 waivers in training/observability/
#   mlflow_logger.py were removed — BLE001 is not enabled in the ruff config,
#   so the directive was entirely dead.)
#
# The ceilings themselves now live in .claude/workforce.yaml
# (ratchet_budgets.items), read here via tools.claude_hooks.config.load_config
# so this hard-fail gate and the ratchet_budget_check hook's early warning can
# never drift apart on what the ceiling actually is. Counting logic is the
# same relocated (not changed) tools.ratchet_budgets.count_marker_occurrences.


def _ceiling(name: str) -> int:
    items = {item.name: item for item in load_config(repo_root=_REPO_ROOT).ratchet_budgets.items}
    return items[name].ceiling


def test_type_ignore_within_budget() -> None:
    count = count_marker_occurrences(_REPO_ROOT, _SCOPE_GLOB, "type: ignore")
    assert count <= _ceiling("type_ignore")


def test_noqa_within_budget() -> None:
    count = count_marker_occurrences(_REPO_ROOT, _SCOPE_GLOB, "noqa")
    assert count <= _ceiling("noqa")


def test_hook_fallback_budgets_match_workforce_yaml() -> None:
    """The hook's built-in fallback budgets must equal the YAML's, item for item.

    This module's header says the ceilings live in ``.claude/workforce.yaml``
    "so this hard-fail gate and the ratchet_budget_check hook's early warning
    can never drift apart on what the ceiling actually is". That holds only
    while the YAML is readable. ``RatchetBudgetsConfig`` also carries a
    hard-coded fallback list, and its docstring claims those defaults
    "reproduce today's hard-fail-only regression budgets ... exactly" while
    naming this very file — but nothing asserted it, and it had in fact
    drifted: the ``hardcoded_ok`` fallback sat at 24/22 against a YAML of
    26/24 for the whole life of the F-030 bump, so any run with a missing or
    unreadable YAML reported a phantom breach at 26 > 24. F-052 Slice A
    ratcheted the YAML to 24/22, which made the fallback accidentally correct;
    this pin is what keeps it correct on the next ratchet, when nobody
    remembers the second file.
    """
    live = {item.name: item for item in load_config(repo_root=_REPO_ROOT).ratchet_budgets.items}
    fallback = {item.name: item for item in RatchetBudgetsConfig().items}

    assert set(fallback) == set(live), (
        "budget names differ between the hook fallback and .claude/workforce.yaml: "
        f"fallback={sorted(fallback)} yaml={sorted(live)}"
    )
    for name, yaml_item in live.items():
        got = fallback[name]
        assert (got.marker, got.ceiling, got.warn_threshold) == (
            yaml_item.marker,
            yaml_item.ceiling,
            yaml_item.warn_threshold,
        ), (
            f"{name}: hook fallback (marker={got.marker!r}, ceiling={got.ceiling}, "
            f"warn={got.warn_threshold}) != .claude/workforce.yaml "
            f"(marker={yaml_item.marker!r}, ceiling={yaml_item.ceiling}, "
            f"warn={yaml_item.warn_threshold}). Ratchet both, or the fallback lies "
            "whenever the YAML cannot be read."
        )


# The measured post-hygiene-sprint set of file-level ruff waivers targeting
# src/, as (glob-pattern, rule) pairs. Each is justified inline in
# pyproject.toml. Ratchet-down-only: removing an entry here is free; adding
# one requires the same documented-reason bar as raising the inline budgets.
_ALLOWED_SRC_PER_FILE_IGNORES = frozenset(
    {
        ("src/mousedroid/harness/**/*.py", "D102"),
        ("src/mousedroid/harness/**/*.py", "D105"),
        ("src/mousedroid/skills/**/*.py", "D102"),
        ("src/mousedroid/skills/**/*.py", "D105"),
        ("src/mousedroid/arm/planning/llm_replanners/**/*.py", "D102"),
        ("src/mousedroid/validation/runtime/_storage.py", "S603"),
        ("src/mousedroid/validation/runtime/_storage.py", "S607"),
        ("src/mousedroid/harness/spec.py", "S602"),
        ("src/mousedroid/harness/spec.py", "S603"),
        ("src/mousedroid/harness/spec.py", "S607"),
    }
)


def test_src_per_file_ignores_within_budget() -> None:
    """No new file-level ruff waiver may target src/ without updating this pin.

    Closes the documented budget bypass: a file-level ignore is invisible to
    the inline noqa counter above, so it gets its own ratchet. Tier-wide
    ``tests/**`` / ``scripts/**`` / ``tools/**`` waivers are deliberately out
    of scope — they cover generated/procedural code, not the runtime package.
    """
    ruff_lint = load_pyproject()["tool"]["ruff"]["lint"]  # type: ignore[index]
    per_file = ruff_lint.get("per-file-ignores", {})
    actual = {
        (pattern, rule)
        for pattern, rules in per_file.items()
        if pattern.startswith("src/")
        for rule in rules
    }
    new_waivers = actual - _ALLOWED_SRC_PER_FILE_IGNORES
    assert not new_waivers, (
        "New file-level ruff waivers target src/ — either remove them or "
        f"update _ALLOWED_SRC_PER_FILE_IGNORES with a documented reason: {sorted(new_waivers)}"
    )
