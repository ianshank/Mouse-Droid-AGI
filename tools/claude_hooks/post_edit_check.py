"""PostToolUse hook: advisory lint/type checks on the file just edited.

Report-only by platform contract — ``PostToolUse`` fires *after* the write, so it
cannot block. Its value is latency: surfacing a lint or type error at edit time
rather than at the end of a long CI run.

The checker set is config-driven (``post_edit.checks``), so adding or removing a
checker is a configuration edit rather than a code change. Nothing here raises:

* an **unknown** checker name is a configuration mistake that will never work,
  so it is skipped and logged at WARNING (``post_edit_checker_unknown``);
* a **known but uninstalled** checker is an environment condition, so it is
  skipped at DEBUG (``post_edit_checker_unavailable``) — this hook runs on every
  edit, and warning each time mypy is absent would be pure noise.

Both keep forward-compatibility with configs written for a newer tool set.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from tools.claude_hooks import hookio
from tools.claude_hooks.config import WorkforceConfig, load_config
from tools.claude_hooks.logging_setup import debug_enabled, get_logger
from tools.claude_hooks.paths import resolve_repo_root, to_repo_relative

_logger = get_logger(__name__)

#: Supported checkers -> the config accessor for their extra arguments. A name
#: absent from this mapping is a configuration error, not a missing install.
_KNOWN_CHECKERS: dict[str, Callable[[WorkforceConfig], list[str]]] = {
    "ruff": lambda cfg: cfg.post_edit.ruff_args,
    "mypy": lambda cfg: cfg.post_edit.mypy_args,
}


def _checker_base_argv(name: str) -> list[str] | None:
    """Return the interpreter-anchored base argv for checker ``name``.

    Invokes the checker as ``sys.executable -m <name>`` rather than the first
    binary on ``PATH``. This is the repository's stated convention (AGENTS.md:
    *"Invoke the linter as ``python -m ruff`` … never bare ``ruff``"*) and it is
    what `scripts/ci.sh` does, so an edit-time check reports the same findings
    as the local gate instead of whatever stray global install shadows it.

    Args:
        name: Checker module name.

    Returns:
        The base argv, or ``None`` when the module is not importable in this
        interpreter (the checker is then skipped, never guessed at).
    """
    if importlib.util.find_spec(name) is None:
        return None
    return [sys.executable, "-m", name]


def _checker_argv(name: str, cfg: WorkforceConfig, target: Path) -> list[str] | None:
    """Build the full argv for checker ``name``, or ``None`` when unavailable.

    Args:
        name: Checker name from ``post_edit.checks``.
        cfg: Workforce configuration.
        target: Absolute path of the file to check.

    Returns:
        The argv list, or ``None`` when the checker is unknown or not installed.
    """
    extra_args = _KNOWN_CHECKERS.get(name)
    if extra_args is None:
        # Unknown name: a config mistake, surfaced by the caller at WARNING.
        return None
    base = _checker_base_argv(name)
    if base is None:
        return None
    return [*base, *extra_args(cfg), str(target)]


def run_checks(
    target: Path,
    cfg: WorkforceConfig,
    *,
    repo_root: Path,
) -> list[tuple[str, str]]:
    """Run the configured checkers against ``target``.

    Args:
        target: Absolute path of the edited file.
        cfg: Workforce configuration.
        repo_root: Repository root, used as the subprocess working directory so
            each checker picks up the repository's own configuration.

    Returns:
        A list of ``(checker_name, report)`` pairs for checkers that reported
        findings. An empty list means everything the hook could run was clean.
    """
    findings: list[tuple[str, str]] = []
    for name in cfg.post_edit.checks:
        argv = _checker_argv(name, cfg, target)
        if argv is None:
            if name not in _KNOWN_CHECKERS:
                # Config names a checker this hook cannot ever run.
                _logger.warning("post_edit_checker_unknown", checker=name)
            else:
                _logger.debug("post_edit_checker_unavailable", checker=name)
            continue
        if debug_enabled():
            _logger.debug("post_edit_invoking", checker=name, argv=argv)
        try:
            # S603: argv[0] is sys.executable and every arg is a list element
            # (no shell), so the call carries no injection surface.
            completed = subprocess.run(  # noqa: S603
                argv,
                capture_output=True,
                text=True,
                timeout=cfg.post_edit.timeout_s,
                check=False,
                cwd=str(repo_root),
            )
        except subprocess.TimeoutExpired:
            _logger.warning("post_edit_check_timeout", checker=name)
            continue
        except OSError as exc:
            _logger.warning("post_edit_check_failed", checker=name, error=str(exc))
            continue

        if completed.returncode != 0:
            report = (completed.stdout or completed.stderr or "").strip()
            findings.append((name, report))
            _logger.info("post_edit_check_findings", checker=name, target=str(target))
        else:
            _logger.debug("post_edit_check_clean", checker=name, target=str(target))
    return findings


def main(
    argv: list[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Entry point for the PostToolUse advisory checks.

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
        _logger.error("post_edit_config_error", error=str(exc))
        return hookio.EXIT_OK

    if not cfg.post_edit.enabled:
        _logger.debug("post_edit_disabled")
        return hookio.EXIT_OK

    target_raw = hookio.extract_target_path(payload)
    if target_raw is None:
        return hookio.EXIT_OK

    target = Path(target_raw)
    if not target.is_absolute():
        target = repo_root / target
    if not target.is_file():
        _logger.debug("post_edit_target_missing", target=str(target))
        return hookio.EXIT_OK
    if target.suffix not in cfg.post_edit.suffixes:
        _logger.debug("post_edit_suffix_skipped", target=str(target), suffix=target.suffix)
        return hookio.EXIT_OK
    if to_repo_relative(target, repo_root) is None:
        _logger.debug("post_edit_outside_repo", target=str(target))
        return hookio.EXIT_OK

    findings = run_checks(target, cfg, repo_root=repo_root)
    for name, report in findings:
        report_stream.write(f"[post-edit:{name}] {target}\n{report}\n")
    report_stream.flush()
    return hookio.EXIT_OK


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
