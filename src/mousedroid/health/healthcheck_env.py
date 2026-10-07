"""Derive Docker healthcheck environment variables from runtime Settings.

The container entrypoint calls :func:`derive_healthcheck_env` once at
startup and writes the mapping to its healthcheck env file
(``print_healthcheck_env``). Two scripts read it back:
``scripts/mousedroid_healthcheck.sh`` (the heartbeat and start-grace keys)
and ``scripts/docker_deploy.sh`` (the ``RESOLVED_`` telemetry endpoint).
Every value is derived here, from the ``Settings`` the rover runs on.

The deploy probe re-applies two rules to what it reads back -- the TCP port
range and the URL-path rule -- because that file is the container's to
write. They are the only rules duplicated in shell, and
``tests/unit/scripts/test_docker_deploy_health_endpoint.py`` runs both
sides against the same cases.

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

from mousedroid.constants import HEALTH_ROUTE_SUFFIX, TCP_PORT_MAX

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

# The URL-path rule ``scripts/docker_deploy.sh`` applies to the health path it
# reads back -- absolute, then RFC 3986 unreserved characters plus ``/`` -- so a
# path is published only if the probe will accept it. Every character it allows
# is also literal inside the single quotes the env file wraps values in.
# ``tests/unit/scripts/test_docker_deploy_health_endpoint.py`` runs the shell
# side against the same cases, so the two cannot drift apart unnoticed.
_RESOLVED_URL_PATH_RE = re.compile(r"/[A-Za-z0-9._~/-]*")
_DOT_SEGMENTS = frozenset({".", ".."})

# TCP ports a probe can connect to; see ``TCP_PORT_MAX``.
_TCP_PORTS = range(1, TCP_PORT_MAX + 1)

# The keys ``scripts/docker_deploy.sh`` reads back. ``RESOLVED_`` marks them
# as outputs of resolution, never operator inputs.
RESOLVED_TELEMETRY_PORT_KEY = "MOUSEDROID_RESOLVED_TELEMETRY_PORT"
RESOLVED_HEALTH_PATH_KEY = "MOUSEDROID_RESOLVED_HEALTH_PATH"


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


def _is_probe_safe_url_path(path: str) -> bool:
    """Return whether the deploy probe can use ``path`` as the URL it asks for.

    The rule ``scripts/docker_deploy.sh`` applies (``_is_url_path``) to the
    path it reads back: absolute, RFC 3986 unreserved characters plus ``/``,
    and no ``.`` or ``..`` segment. A dot segment passes the character rule but
    not the trip: the server registers the path literally, while HTTP clients
    (curl, aiohttp) remove dot segments before sending, so the probe would ask
    for a different route from the one served.

    Args:
        path: Candidate URL path.

    Returns:
        ``True`` when the probe would request exactly this path.
    """
    if not _RESOLVED_URL_PATH_RE.fullmatch(path):
        return False
    return not any(segment in _DOT_SEGMENTS for segment in path.split("/"))


def _resolved_telemetry_port(cfg: Settings) -> str | None:
    """Return the port the rover will serve, or ``None`` when it cannot be known.

    Only the ``fixed`` strategy binds ``telemetry.port`` itself: ``fallback_range``
    may bind a later port, and ``kernel_assigned`` lets the OS pick one. For those
    the configured port would be a confident wrong answer, so nothing is
    published and the deploy probe says why.

    The value is re-checked rather than trusted -- an in-range ``int``, written
    back through ``int()`` -- so a ``Settings`` that bypassed validation
    (``model_construct``, a test double) can only ever put digits in the file.

    Args:
        cfg: Resolved runtime ``Settings`` instance.

    Returns:
        The port as a decimal string, or ``None``.
    """
    telemetry = cfg.telemetry
    if telemetry.port_discovery_strategy != "fixed":
        return None
    port = telemetry.port
    if isinstance(port, bool) or not isinstance(port, int) or port not in _TCP_PORTS:
        return None
    return str(int(port))


def _resolved_health_path(cfg: Settings) -> str | None:
    """Return the health route the rover registers, or ``None`` if unusable.

    ``telemetry.api_prefix`` is a free-form string in the schema, so a prefix
    the rover accepts can still fail the probe's URL rule (a space, a quote, no
    leading ``/``, a ``.`` or ``..`` segment). Such a path is OMITTED, never
    raised on: the entrypoint
    writes this file under ``set -eu`` before it execs the rover, so raising
    here would turn a prefix the rover itself accepts into a crash loop.

    Args:
        cfg: Resolved runtime ``Settings`` instance.

    Returns:
        The path, or ``None``.
    """
    path = f"{cfg.telemetry.api_prefix}{HEALTH_ROUTE_SUFFIX}"
    return path if _is_probe_safe_url_path(path) else None


def derive_healthcheck_env(cfg: Settings) -> dict[str, str]:
    """Return the env-var mapping for the Docker healthcheck script.

    Args:
        cfg: Resolved runtime ``Settings`` instance.

    Returns:
        Mapping of env var name to string value, every value validated for
        shell-source safety. The four heartbeat/grace keys form a stable
        contract with ``scripts/mousedroid_healthcheck.sh`` and are never
        empty. The two ``MOUSEDROID_RESOLVED_*`` keys are read by
        ``scripts/docker_deploy.sh``; each is always present and is EMPTY when
        the rover cannot vouch for it (see :func:`_resolved_telemetry_port`
        and :func:`_resolved_health_path`). Present-but-empty is how the probe
        tells "unknowable" apart from "code that predates the key" (absent).

    Raises:
        ValueError: If a heartbeat or start-grace path contains characters
            unsafe for shell sourcing. The resolved keys never raise.
    """
    stale_s = cfg.loop.watchdog_interval_s * cfg.loop.watchdog_tolerance_factor
    env = {
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
    # The telemetry endpoint the rover will actually serve, derived from the
    # SAME ``Settings`` as every value above -- and therefore from the exact
    # ``--config`` the entrypoint hands to ``mousedroid.main``.
    # ``scripts/docker_deploy.sh`` reads these from the running container rather
    # than guessing: before they existed it probed ``MOUSEDROID_TELEMETRY_PORT``,
    # which is not a settings key (the nested delimiter is ``__``), so an
    # operator moving the port the supported way left the probe on the old one.
    env[RESOLVED_TELEMETRY_PORT_KEY] = _resolved_telemetry_port(cfg) or ""
    env[RESOLVED_HEALTH_PATH_KEY] = _resolved_health_path(cfg) or ""
    return env
