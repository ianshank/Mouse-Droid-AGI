"""PreToolUse hook: edit-time secret scan.

Closes the gap the CI secret-scan gate leaves open: CI catches a leaked
credential *after* it is committed and pushed, and only advisorily. This hook
scans the pending buffer *before* it reaches disk, using the repository's own
scanner and its regex-only allowlist so there is exactly one secret policy.

Posture when the scanner is unavailable (not installed, or it timed out) is
config-driven via ``secret_scan.strict``: warn-and-allow by default, mirroring
the advisory CI job, or deny when an operator wants the stricter stance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Any

from tools.claude_hooks import hookio
from tools.claude_hooks.config import ConfigError, WorkforceConfig, load_config
from tools.claude_hooks.logging_setup import debug_enabled, get_logger
from tools.claude_hooks.paths import resolve_repo_root

_logger = get_logger(__name__)

#: gitleaks exits 1 when it finds leaks, 0 when clean; anything else is an error.
_EXIT_CLEAN = 0
_EXIT_LEAKS_FOUND = 1

DENY_TEMPLATE = (
    "Edit-time secret scan blocked this write to {target}.\n"
    "{scanner} reported a finding in the pending content.\n"
    "{detail}\n"
    "If this is a documented placeholder, add its literal regex to {config_file} "
    "(regex only — never allowlist by path)."
)

UNAVAILABLE_TEMPLATE = (
    "Edit-time secret scan could not run ({problem}) and strict mode is enabled.\n"
    "Install '{scanner}' or set secret_scan.strict=false in the workforce config."
)


class ScanOutcome:
    """Result of a scan attempt.

    Attributes:
        clean: The content passed the scan.
        available: The scanner ran to completion.
        detail: Operator-facing detail (findings summary or failure reason).
    """

    __slots__ = ("available", "clean", "detail")

    def __init__(self, *, clean: bool, available: bool, detail: str = "") -> None:
        """Initialise the outcome."""
        self.clean = clean
        self.available = available
        self.detail = detail


def _suffix_for(target: str | None) -> str:
    """Return a temp-file suffix mirroring the target's extension."""
    if not target:
        return ".txt"
    suffix = Path(target).suffix
    return suffix if suffix and len(suffix) <= 16 else ".txt"


def scan_content(
    content: str,
    cfg: WorkforceConfig,
    *,
    repo_root: Path,
    target: str | None = None,
) -> ScanOutcome:
    """Scan ``content`` with the configured scanner.

    The content is written to a temporary file outside the repository (so the
    scan never picks up neighbouring files) and scanned in no-git mode.

    Args:
        content: Pending content to scan.
        cfg: Workforce configuration.
        repo_root: Repository root, used to resolve the allowlist config.
        target: The edit's target path, used only to pick a temp-file suffix.

    Returns:
        The :class:`ScanOutcome`.
    """
    executable = shutil.which(cfg.secret_scan.command)
    if executable is None:
        return ScanOutcome(
            clean=True,
            available=False,
            detail=f"'{cfg.secret_scan.command}' not found on PATH",
        )

    config_path = repo_root / cfg.secret_scan.config_file
    with tempfile.TemporaryDirectory(prefix="workforce-secret-scan-") as tmpdir:
        probe = Path(tmpdir) / f"pending{_suffix_for(target)}"
        probe.write_text(content, encoding="utf-8")

        argv = [
            executable,
            "detect",
            "--no-git",
            "--source",
            str(probe),
            "--redact",
            "--no-banner",
        ]
        if config_path.is_file():
            argv.extend(["--config", str(config_path)])
        argv.extend(cfg.secret_scan.extra_args)

        if debug_enabled():
            _logger.debug("secret_scan_invoking", argv=argv, bytes=len(content))

        try:
            # S603: argv[0] is resolved via shutil.which and every element is a
            # list item (no shell), so there is no injection surface here.
            completed = subprocess.run(  # noqa: S603
                argv,
                capture_output=True,
                text=True,
                timeout=cfg.secret_scan.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ScanOutcome(
                clean=True,
                available=False,
                detail=f"scan timed out after {cfg.secret_scan.timeout_s}s",
            )
        except OSError as exc:
            return ScanOutcome(clean=True, available=False, detail=f"scanner failed: {exc}")

    if completed.returncode == _EXIT_CLEAN:
        return ScanOutcome(clean=True, available=True)
    if completed.returncode == _EXIT_LEAKS_FOUND:
        detail = (completed.stdout or completed.stderr or "").strip()
        return ScanOutcome(clean=False, available=True, detail=detail)
    return ScanOutcome(
        clean=True,
        available=False,
        detail=(
            f"scanner exited {completed.returncode}: "
            f"{(completed.stderr or '').strip() or 'no stderr'}"
        ),
    )


def evaluate(
    payload: dict[str, Any],
    cfg: WorkforceConfig,
    *,
    repo_root: Path,
) -> tuple[bool, str]:
    """Decide whether the pending write is allowed.

    Args:
        payload: Parsed hook payload.
        cfg: Workforce configuration.
        repo_root: Repository root.

    Returns:
        ``(allowed, reason)``. ``reason`` is empty when allowed.
    """
    if not cfg.secret_scan.enabled:
        _logger.debug("secret_scan_disabled")
        return True, ""

    content = hookio.extract_pending_content(payload)
    if not content:
        _logger.debug("secret_scan_no_content", tool=hookio.tool_name(payload))
        return True, ""

    target = hookio.extract_target_path(payload)

    if len(content.encode("utf-8", errors="ignore")) > cfg.secret_scan.max_bytes:
        _logger.warning(
            "secret_scan_skipped_oversized",
            target=target,
            max_bytes=cfg.secret_scan.max_bytes,
        )
        return True, ""

    outcome = scan_content(content, cfg, repo_root=repo_root, target=target)

    if not outcome.available:
        if cfg.secret_scan.strict:
            _logger.error("secret_scan_unavailable_strict", target=target, detail=outcome.detail)
            return False, UNAVAILABLE_TEMPLATE.format(
                problem=outcome.detail,
                scanner=cfg.secret_scan.command,
            )
        _logger.warning("secret_scan_unavailable", target=target, detail=outcome.detail)
        return True, ""

    if outcome.clean:
        _logger.debug("secret_scan_clean", target=target)
        return True, ""

    _logger.error("secret_scan_denied", target=target)
    return False, DENY_TEMPLATE.format(
        target=target or "the pending file",
        scanner=cfg.secret_scan.command,
        detail=outcome.detail or "(finding details redacted by the scanner)",
        config_file=cfg.secret_scan.config_file,
    )


def main(
    argv: list[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Entry point for the PreToolUse secret scan.

    Args:
        argv: Unused; accepted for signature parity with the other hooks.
        stdin: Payload stream. Defaults to :data:`sys.stdin`.
        stdout: Decision stream. Defaults to :data:`sys.stdout`.
        env: Environment mapping. Defaults to :data:`os.environ`.

    Returns:
        Process exit code.
    """
    del argv
    environ = dict(os.environ) if env is None else env
    payload = hookio.read_payload(stdin)

    try:
        repo_root = resolve_repo_root(env=environ)
        cfg = load_config(repo_root=repo_root)
    except ConfigError as exc:
        _logger.error("secret_scan_config_invalid", error=str(exc))
        return hookio.emit_deny(
            f"Edit-time secret scan could not load its configuration: {exc}",
            stream=stdout,
        )
    except Exception as exc:  # Environment failure must not brick every edit.
        _logger.error("secret_scan_environment_error", error=str(exc))
        return hookio.emit_allow()

    allowed, reason = evaluate(payload, cfg, repo_root=repo_root)
    if allowed:
        return hookio.emit_allow()
    return hookio.emit_deny(reason, stream=stdout)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
