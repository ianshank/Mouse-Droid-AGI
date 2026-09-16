"""Early-warning checker for this repo's ratchet-down-only numeric budgets.

Several hard-fail regression tests cap inline suppression/marker counts and
may only ratchet the cap *down*: ``noqa`` and ``type: ignore`` occurrences
(``tests/regression/test_suppression_budget.py``) and the ``# hardcoded-ok``
waiver marker (``tests/regression/test_hardcoded_value_marker_budget.py``).
All three previously had zero advance signal — a change either stayed under
the ceiling or went hard red on it, with nothing in between.

This module is the shared, pure counting logic behind both an edit-time hook
(``tools/claude_hooks/ratchet_budget_check.py``) and the regression tests
themselves (re-pointed here rather than each keeping its own private
``_count()`` helper) — one counting implementation, two consumers, mirroring
``tools/doc_hygiene.py``'s shape. Budget definitions (marker, scope, ceiling,
warn threshold) live in ``.claude/workforce.yaml`` via
``tools.claude_hooks.config.RatchetBudgetsConfig`` — nothing here restates a
threshold.

Report-only by convention: ``main()`` always exits 0 unless ``--strict``,
matching ``tools/doc_hygiene.py`` and ``scripts/check_advisory_promotions.py``.

``--strict`` fails on a **ceiling breach** only, not on an approaching-budget
warning. The distinction is load-bearing rather than cosmetic: a
ratchet-down-only budget is, by its own discipline, ratcheted to the current
count whenever a waiver is resolved, so a healthy budget sits *at* its ceiling
and therefore permanently above its ``warn_threshold``. A ``--strict`` that
failed on any warning would be red on a correctly-maintained repo, which is why
the flag existed but was never wired into CI. Failing on the breach — the
condition the budget actually contracts — makes it wireable. Every pre-existing
``--strict`` test exercised a real breach (25 occurrences against a ceiling of
19), so no behaviour a test depended on changed here; only the docstring and
help text, which described the conflated form, did.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.claude_hooks.config import ConfigError, RatchetBudgetItem, load_config
from tools.claude_hooks.logging_setup import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class BudgetFinding:
    """A single budget's health, with the breach/approaching split made explicit.

    ``check_budget_item`` and ``check_all_budgets`` still return ``list[str]``
    for their pre-existing consumers (the ``PostToolUse`` hook and the
    regression tests); this type is the severity-aware view those two now
    delegate to, so the message text has exactly one source.

    Attributes:
        name: Budget name from ``.claude/workforce.yaml``.
        count: Measured marker occurrences in scope.
        ceiling: The ratchet-down-only cap this budget contracts.
        breached: ``True`` when ``count`` exceeds ``ceiling`` — a contract
            violation. ``False`` for an approaching-budget warning, which is
            advisory by design and must not fail a gate.
        message: Human-readable report line, without the ``WARN:`` prefix.
    """

    name: str
    count: int
    ceiling: int
    breached: bool
    message: str


def count_marker_occurrences(repo_root: Path, scope_glob: str, marker: str) -> int:
    """Count ``marker`` occurrences per source line across ``scope_glob``.

    Relocates (does not change) the inline counting logic the pre-existing
    hard-fail regression tests used: ``sum(line.count(marker) for p in ... for
    line in p.read_text().splitlines())``. ``scope_glob`` is resolved with
    :meth:`Path.glob`, whose ``**`` component crosses directory separators the
    same way the original tests' ``rglob("*.py")`` did, so re-pointing a test
    at this function must not change its count.

    Args:
        repo_root: Repository root the glob is resolved against.
        scope_glob: A glob pattern relative to ``repo_root``, e.g.
            ``"src/mousedroid/**/*.py"``.
        marker: Literal substring counted per line.

    Returns:
        The total occurrence count across every matched file.
    """
    return sum(
        line.count(marker)
        for path in sorted(repo_root.glob(scope_glob))
        if path.is_file()
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def classify_budget_item(repo_root: Path, item: RatchetBudgetItem) -> BudgetFinding | None:
    """Return this item's finding, or ``None`` when healthy.

    A count over ``ceiling`` is a breach; a count over ``warn_threshold`` but
    still at or under ``ceiling`` is an approaching-budget warning. Only one of
    the two fires per item — a ceiling breach already implies the warn threshold
    was crossed, so reporting both would be redundant noise.

    Args:
        repo_root: Repository root the budget's ``scope_glob`` resolves against.
        item: The budget definition, sourced from ``.claude/workforce.yaml``.

    Returns:
        A :class:`BudgetFinding` when the budget is breached or approaching,
        otherwise ``None``.
    """
    count = count_marker_occurrences(repo_root, item.scope_glob, item.marker)
    _logger.debug(
        "ratchet_budget_measured",
        extra={
            "budget": item.name,
            "count": count,
            "ceiling": item.ceiling,
            "warn_threshold": item.warn_threshold,
            "scope_glob": item.scope_glob,
        },
    )
    if count > item.ceiling:
        return BudgetFinding(
            name=item.name,
            count=count,
            ceiling=item.ceiling,
            breached=True,
            message=(
                f"{item.name}: {count} occurrences exceeds the ratchet-down-only "
                f"ceiling of {item.ceiling} (marker={item.marker!r}, scope={item.scope_glob})"
            ),
        )
    if item.warn_threshold is not None and count > item.warn_threshold:
        return BudgetFinding(
            name=item.name,
            count=count,
            ceiling=item.ceiling,
            breached=False,
            message=(
                f"{item.name}: {count} occurrences crossed the early-warning "
                f"threshold of {item.warn_threshold} (ceiling: {item.ceiling}) - "
                "approaching the ratchet-down-only budget"
            ),
        )
    return None


def classify_all_budgets(
    repo_root: Path, items: Sequence[RatchetBudgetItem]
) -> list[BudgetFinding]:
    """Return findings across every budget item (empty == all healthy)."""
    findings: list[BudgetFinding] = []
    for item in items:
        finding = classify_budget_item(repo_root, item)
        if finding is not None:
            findings.append(finding)
    return findings


def check_budget_item(repo_root: Path, item: RatchetBudgetItem) -> list[str]:
    """Return WARN strings for a single budget item (empty == healthy).

    Retained at its original ``list[str]`` signature for the ``PostToolUse``
    hook and the regression tests; delegates to :func:`classify_budget_item` so
    the message text has one source.
    """
    finding = classify_budget_item(repo_root, item)
    return [] if finding is None else [finding.message]


def check_all_budgets(repo_root: Path, items: Sequence[RatchetBudgetItem]) -> list[str]:
    """Return WARN strings across every budget item (empty == all healthy)."""
    return [finding.message for finding in classify_all_budgets(repo_root, items)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python tools/ratchet_budgets.py",
        description=(
            "Early-warning check for this repo's ratchet-down-only suppression/"
            "marker budgets (WARN-only unless --strict)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 1 when a budget exceeds its ceiling (default: advisory, always "
            "exit 0). Approaching-budget warnings still print but never fail: a "
            "correctly ratcheted budget sits at its ceiling by design."
        ),
    )
    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()

    try:
        cfg = load_config(repo_root=repo_root)
    except ConfigError as exc:
        # Advisory tool: a broken workforce config must degrade to a WARN like
        # any other finding, not crash the script out from under --strict's
        # documented "exit 0 unless --strict" contract.
        print(f"WARN: could not load workforce config ({exc}) - ratchet budgets not checked")
        return 1 if args.strict else 0

    findings = (
        classify_all_budgets(repo_root, cfg.ratchet_budgets.items)
        if cfg.ratchet_budgets.enabled
        else []
    )

    for finding in findings:
        print(f"WARN: {finding.message}")
    if not findings:
        print("ratchet budgets: all tracked budgets within range")

    breaches = [finding for finding in findings if finding.breached]
    if breaches:
        _logger.warning(
            "ratchet_budget_breached",
            extra={"budgets": [finding.name for finding in breaches], "strict": args.strict},
        )
    if args.strict and breaches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
