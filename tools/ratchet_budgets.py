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
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.claude_hooks.config import RatchetBudgetItem, load_config


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


def check_budget_item(repo_root: Path, item: RatchetBudgetItem) -> list[str]:
    """Return WARN strings for a single budget item (empty == healthy).

    A count over ``ceiling`` is reported as an over-budget warning; a count
    over ``warn_threshold`` (but still at or under ``ceiling``) is reported as
    an approaching-budget warning. Only one of the two fires per item — a
    ceiling breach already implies the warn threshold was crossed, so
    reporting both would be redundant noise.
    """
    count = count_marker_occurrences(repo_root, item.scope_glob, item.marker)
    if count > item.ceiling:
        return [
            f"{item.name}: {count} occurrences exceeds the ratchet-down-only "
            f"ceiling of {item.ceiling} (marker={item.marker!r}, scope={item.scope_glob})"
        ]
    if item.warn_threshold is not None and count > item.warn_threshold:
        return [
            f"{item.name}: {count} occurrences crossed the early-warning "
            f"threshold of {item.warn_threshold} (ceiling: {item.ceiling}) - "
            "approaching the ratchet-down-only budget"
        ]
    return []


def check_all_budgets(repo_root: Path, items: Sequence[RatchetBudgetItem]) -> list[str]:
    """Return WARN strings across every budget item (empty == all healthy)."""
    warnings: list[str] = []
    for item in items:
        warnings.extend(check_budget_item(repo_root, item))
    return warnings


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
        help="Exit 1 when any warning fires (default: advisory, always exit 0).",
    )
    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()

    cfg = load_config(repo_root=repo_root)
    warnings = (
        check_all_budgets(repo_root, cfg.ratchet_budgets.items)
        if cfg.ratchet_budgets.enabled
        else []
    )

    for warning in warnings:
        print(f"WARN: {warning}")
    if not warnings:
        print("ratchet budgets: all tracked budgets within range")
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
