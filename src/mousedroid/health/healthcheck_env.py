"""Derive Docker healthcheck environment variables from runtime Settings.

The container entrypoint calls :func:`derive_healthcheck_env` once at
startup to produce the env-var mapping consumed by
``scripts/mousedroid_healthcheck.sh``. This is the single source of
truth — no values are duplicated between Python and shell. The shell
script reads only env vars; this module defines them.

The module also re-applies the shell-safety whitelist that the
``LoopConfig`` field validator enforces at YAML load time. The
duplication is defense in depth: if a value somehow reaches this
function via an unvalidated code path (mocking in tests, programmatic
``Settings`` construction, etc.), the env file is still safe to
dot-source.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ``Settings`` is only used as a type annotation. ``from __future__
    # import annotations`` defers annotation evaluation, so this import
    # never executes at runtime. Removing the ``from __future__`` import
    # at the top would silently break this — keep it.
    from mousedroid.config.schema import Settings


# Reject any character that could break out of single-quoted shell
# strings or enable command substitution. Whitelist forward slashes,
# alphanumerics, dot, dash, underscore, plus colon (for paths like
# C:/...). Anything else is unsafe for shell-source contexts.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._/\-:]+$")


def _validate_path(value: str, field: str) -> str:
    """Reject paths that could break shell-source quoting.

    Args:
        value: Path string to validate.
        field: Name of the originating config field, used in the error
            message so operators can find the offending YAML line.

    Returns:
        ``value`` unchanged when it passes the whitelist.

    Raises:
        ValueError: When ``value`` contains characters outside the
            shell-safe whitelist.
    """
    if not _SAFE_PATH_RE.fullmatch(value):
        msg = (
            f"{field}={value!r} contains characters unsafe for shell-source "
            f"env files; allowed: [A-Za-z0-9._/-:]"
        )
        raise ValueError(msg)
    return value


def derive_healthcheck_env(cfg: Settings) -> dict[str, str]:
    """Return the env-var mapping for the Docker healthcheck script.

    Args:
        cfg: Resolved runtime ``Settings`` instance.

    Returns:
        Mapping of env var name to string value. All values are
        non-empty and validated for shell-source safety. Keys form a
        stable contract with ``scripts/mousedroid_healthcheck.sh``.

    Raises:
        ValueError: If any path-typed config value contains characters
            unsafe for shell sourcing.
    """
    stale_s = cfg.loop.watchdog_interval_s * cfg.loop.watchdog_tolerance_factor
    return {
        "MOUSEDROID_HEARTBEAT_PATH": _validate_path(
            cfg.loop.watchdog_heartbeat_path,
            "watchdog_heartbeat_path",
        ),
        "MOUSEDROID_HEARTBEAT_STALE_S": f"{stale_s:.3f}",
        "MOUSEDROID_START_GRACE_S": f"{cfg.loop.start_grace_s:.3f}",
        "MOUSEDROID_START_GRACE_FILE": _validate_path(
            cfg.loop.start_grace_file,
            "start_grace_file",
        ),
    }
