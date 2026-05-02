"""Validate every YAML overlay under ``config/`` against the Pydantic schema.

This catches three classes of regressions before they reach the Jetson:

1. **Schema drift** — a YAML key was removed/renamed but the overlay still
   references it (Pydantic ``ValidationError``).
2. **Type drift** — a field's type tightened (e.g. ``int`` → ``Literal[...]``).
3. **Cross-field invariants** — model validators (e.g.
   ``ModelConfig._validate_optional_modalities``) that fire only after a
   full overlay merge.

Usage::

    python scripts/validate_configs.py
    python scripts/validate_configs.py --fail-fast
    python scripts/validate_configs.py --include-default
    python scripts/validate_configs.py --config-dir path/to/configs

The script returns:

* ``0`` — every overlay loaded.
* ``1`` — at least one overlay raised a ``ValidationError`` (or other error).
* ``2`` — usage / IO error (no YAMLs found, missing default).

Designed to be safe to run from a pre-commit hook *and* from CI: it does
not import any platform-specific drivers (``mousedroid.config.loader``
only touches Pydantic + YAML).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

# Resolve the repo root so the script is invocable from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402

_log = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2

DEFAULT_FILENAME = "default.yaml"
SKIP_MARKER = "# config-validator: skip"
"""Marker comment that excludes a YAML from validation.

Place this in the first 10 lines of any YAML in ``config/`` that is not
a runtime overlay (e.g. deployment descriptors). Keeps the validator
opt-in-skip and self-documenting on the YAML side.
"""
_MARKER_HEAD_LINES = 10


@dataclass(frozen=True)
class OverlayResult:
    """One overlay's load outcome."""

    path: Path
    ok: bool
    error: str | None = None


def _has_skip_marker(path: Path) -> bool:
    """Return True if ``path``'s first ``_MARKER_HEAD_LINES`` carry the skip marker."""
    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(_MARKER_HEAD_LINES):
                line = fh.readline()
                if not line:
                    return False
                if SKIP_MARKER in line:
                    return True
    except OSError:
        return False
    return False


def discover_overlays(config_dir: Path, *, include_default: bool = False) -> list[Path]:
    """Return every ``*.yaml`` under ``config_dir`` (lexicographic order).

    Files carrying the :data:`SKIP_MARKER` comment in their first
    ``_MARKER_HEAD_LINES`` are excluded — they are deployment descriptors
    or other non-runtime YAMLs that share the ``config/`` directory.

    Args:
        config_dir: Directory containing YAML overlays.
        include_default: When ``True``, include ``default.yaml`` itself
            (validated by loading with no overlays). Default ``False``
            because ``load_settings()`` already loads it as the base.

    Returns:
        Sorted list of YAML paths to validate.
    """
    if not config_dir.is_dir():
        msg = f"config_dir does not exist or is not a directory: {config_dir}"
        raise NotADirectoryError(msg)
    overlays = sorted(config_dir.glob("*.yaml"))
    if not include_default:
        overlays = [p for p in overlays if p.name != DEFAULT_FILENAME]
    skipped = [p for p in overlays if _has_skip_marker(p)]
    for p in skipped:
        _log.info("config_overlay_skip", overlay=str(p), reason="skip_marker")
    return [p for p in overlays if p not in skipped]


def validate_overlay(overlay: Path, *, config_dir: Path) -> OverlayResult:
    """Load ``default.yaml`` + ``overlay`` through the full Pydantic schema.

    Catches every Pydantic ``ValidationError`` plus any ``Exception`` raised
    by the loader (file system, YAML parse, etc.).
    """
    try:
        load_settings(overlay, config_dir=config_dir)
    except ValidationError as exc:
        _log.error(
            "config_overlay_invalid",
            overlay=str(overlay),
            error_count=len(exc.errors()),
        )
        return OverlayResult(path=overlay, ok=False, error=str(exc))
    except Exception as exc:
        _log.error("config_overlay_load_error", overlay=str(overlay), error=str(exc))
        return OverlayResult(path=overlay, ok=False, error=f"{type(exc).__name__}: {exc}")
    _log.info("config_overlay_ok", overlay=str(overlay))
    return OverlayResult(path=overlay, ok=True)


def validate_default(config_dir: Path) -> OverlayResult:
    """Validate ``default.yaml`` standalone (no overlays)."""
    default_path = config_dir / DEFAULT_FILENAME
    try:
        load_settings(config_dir=config_dir)
    except ValidationError as exc:
        return OverlayResult(path=default_path, ok=False, error=str(exc))
    except Exception as exc:
        return OverlayResult(
            path=default_path,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return OverlayResult(path=default_path, ok=True)


def run(
    config_dir: Path,
    *,
    fail_fast: bool = False,
    include_default: bool = False,
) -> tuple[list[OverlayResult], int]:
    """Drive validation; returns ``(results, exit_code)``."""
    if not config_dir.exists():
        _log.error("config_dir_missing", config_dir=str(config_dir))
        return [], EXIT_USAGE

    results: list[OverlayResult] = []

    if include_default:
        result = validate_default(config_dir)
        results.append(result)
        if not result.ok and fail_fast:
            return results, EXIT_FAILURES

    try:
        overlays = discover_overlays(config_dir, include_default=False)
    except NotADirectoryError:
        _log.error("config_dir_not_a_directory", config_dir=str(config_dir))
        return results, EXIT_USAGE

    if not overlays and not include_default:
        _log.warning("config_no_overlays_found", config_dir=str(config_dir))
        return results, EXIT_USAGE

    for overlay in overlays:
        result = validate_overlay(overlay, config_dir=config_dir)
        results.append(result)
        if not result.ok and fail_fast:
            return results, EXIT_FAILURES

    failures = [r for r in results if not r.ok]
    return results, EXIT_FAILURES if failures else EXIT_OK


def _print_report(results: list[OverlayResult]) -> None:
    """Emit per-overlay structured log lines summarising the run.

    Uses :data:`_log` (structlog) rather than ``print()`` so the script
    obeys CLAUDE.md invariant 4 ("Never use ``print()``") and so CI runs
    can ingest the output as JSON instead of free-form lines.
    """
    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    _log.info(
        "config_validate_summary",
        passed=passed,
        failed=failed,
        total=len(results),
    )
    for result in results:
        if result.ok:
            _log.info(
                "config_validate_overlay_ok",
                path=str(result.path),
            )
        else:
            _log.error(
                "config_validate_overlay_failed",
                path=str(result.path),
                error=result.error,
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate every YAML overlay under config/ against the schema.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_REPO_ROOT / "config",
        help="Directory containing YAML overlays (default: repo config/).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing overlay.",
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Also validate default.yaml standalone.",
    )
    args = parser.parse_args(argv)

    results, exit_code = run(
        args.config_dir,
        fail_fast=args.fail_fast,
        include_default=args.include_default,
    )
    _print_report(results)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
