"""PostToolUse hook: ratchet-budget early-warning check.

Report-only by platform contract — ``PostToolUse`` fires *after* the write, so
it cannot block. Its value is latency: surfacing a ratchet-down-only budget
(``noqa``, ``type: ignore``, ``# hardcoded-ok``, or any operator-added item)
crossing its early-warning threshold at edit time, well before the hard-fail
regression tests (``test_suppression_budget.py``,
``test_hardcoded_value_marker_budget.py``) turn a PR red on it.

Gated on the edited file falling inside any tracked budget item's
``scope_glob`` — narrower than ``post_edit_check.py``'s per-suffix gate, since
a ratchet budget's scope is a repo-relative glob, not a bare file extension.
Reuses :func:`tools.claude_hooks.paths.path_matches_any`, the same primitive
``freeze_gate.py`` uses for its frozen-paths glob. Nothing here raises: an
unreadable or invalid workforce config degrades to a silent no-op (advisory
hook, never fails the turn).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO, Any

from tools.claude_hooks import hookio
from tools.claude_hooks.config import load_config
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.paths import path_matches_any, resolve_repo_root, to_repo_relative
from tools.ratchet_budgets import check_all_budgets

_logger = get_logger(__name__)


def main(
    argv: list[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Entry point for the PostToolUse ratchet-budget early warning.

    Always returns success: this hook reports, it never blocks.

    Args:
        argv: Unused; accepted for signature parity with the other hooks.
        stdin: Payload stream. Defaults to :data:`sys.stdin`.
        stdout: Unused; the hook writes no decision payload.
        stderr: Report stream. Defaults to :data:`sys.stderr`.
        env: Environment mapping. Defaults to :data:`os.environ`.

    Returns:
        :data:`~tools.claude_hooks.hookio.EXIT_OK`.
    """
    del argv, stdout
    environ = dict(os.environ) if env is None else env
    report_stream = sys.stderr if stderr is None else stderr
    payload: dict[str, Any] = hookio.read_payload(stdin)

    try:
        repo_root = resolve_repo_root(env=environ)
        cfg = load_config(repo_root=repo_root)
    except Exception as exc:  # Advisory hook: never fail the turn.
        _logger.error("ratchet_budget_check_config_error", error=str(exc))
        return hookio.EXIT_OK

    if not cfg.ratchet_budgets.enabled:
        _logger.debug("ratchet_budget_check_disabled")
        return hookio.EXIT_OK

    target_raw = hookio.extract_target_path(payload)
    if target_raw is None:
        return hookio.EXIT_OK

    target = Path(target_raw)
    if not target.is_absolute():
        target = repo_root / target
    if not target.is_file():
        _logger.debug("ratchet_budget_check_target_missing", target=str(target))
        return hookio.EXIT_OK
    rel_path = to_repo_relative(target, repo_root)
    if rel_path is None:
        _logger.debug("ratchet_budget_check_outside_repo", target=str(target))
        return hookio.EXIT_OK

    scope_globs = [item.scope_glob for item in cfg.ratchet_budgets.items]
    if path_matches_any(rel_path, scope_globs) is None:
        _logger.debug("ratchet_budget_check_out_of_scope", rel_path=rel_path)
        return hookio.EXIT_OK

    warnings = check_all_budgets(repo_root, cfg.ratchet_budgets.items)
    for warning in warnings:
        report_stream.write(f"[ratchet-budget] {warning}\n")
    if warnings:
        _logger.info("ratchet_budget_check_findings", rel_path=rel_path, count=len(warnings))
    else:
        _logger.debug("ratchet_budget_check_clean", rel_path=rel_path)
    report_stream.flush()
    return hookio.EXIT_OK


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
