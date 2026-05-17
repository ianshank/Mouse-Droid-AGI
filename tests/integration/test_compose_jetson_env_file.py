"""F-014 regression: docker-compose.jetson.yml env_file + env-file-sourced settings.

The smoke-stability sprint surfaced two coupled problems on the live Jetson:

1. ``docker-compose.jetson.yml`` line 31 substituted
   ``MOUSEDROID_MOCK_HARDWARE=${MOUSEDROID_MOCK_HARDWARE:-true}`` from the
   **host shell env at compose-up time**, ignoring whatever was in
   ``/etc/mousedroid/docker.env``. Production thus defaulted to mock mode
   unless the operator explicitly exported the var, which is the opposite
   of what the runbook implied.
2. ``MOUSEDROID_TELEMETRY_TOKEN`` was defined in ``/etc/mousedroid/docker.env``
   but never reached the container because the compose file had no
   ``env_file:`` directive — production restarts crashed with
   ``TelemetryConfigError: telemetry auth_enabled=True but
   $MOUSEDROID_TELEMETRY_TOKEN is unset``.

The fix has two parts: add the ``env_file:`` directive AND keep
``MOUSEDROID_MOCK_HARDWARE`` and ``MOUSEDROID_TELEMETRY_TOKEN`` OUT of the
inline ``environment:`` block. Per Compose spec, inline ``environment:``
ALWAYS overrides ``env_file:`` values — so any inline default
(``${VAR:-...}``) would silently mask the docker.env value when the host
shell var is unset, reintroducing the same crash-loop. The corrected
design makes ``/etc/mousedroid/docker.env`` the single source of truth
for both keys with zero precedence ambiguity.

These tests parse the YAML directly (no Docker daemon required) and pin
the fix end-to-end: env_file declared with ``required: false``,
MOCK_HARDWARE absent from inline environment, TELEMETRY_TOKEN absent from
inline environment, and the operator template at
``config/.env.jetson.example`` documents the canonical settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"


def _load_service() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    services = raw["services"]
    svc: dict[str, Any] = services["mousedroid"]
    return svc


def test_compose_declares_env_file_directive() -> None:
    """``env_file: /etc/mousedroid/docker.env`` is the source of truth for env.

    Without this directive the compose `environment:` block can't pick up
    the operator's docker.env values; the production restart this fix is
    addressing crashed exactly because the telemetry token never reached
    the container. Long-form (``{path:, required: false}``) is required so
    first-time bringup (and ``docker compose config --quiet`` lint runs)
    don't crash when ``/etc/mousedroid/docker.env`` isn't deployed yet.
    """
    svc = _load_service()
    env_files = svc.get("env_file", [])
    assert env_files, "compose service must declare env_file (see F-014)"

    def _matches(entry: object) -> bool:
        # Short-form: bare string path.
        if isinstance(entry, str):
            return "/etc/mousedroid/docker.env" in entry
        # Long-form: {path: ..., required: ...} dict — required for first-
        # time bringup safety.
        if isinstance(entry, dict):
            return "/etc/mousedroid/docker.env" in str(entry.get("path", ""))
        return False

    assert any(
        _matches(p) for p in env_files
    ), f"env_file must include /etc/mousedroid/docker.env; got {env_files!r}"


def test_compose_env_file_marked_required_false() -> None:
    """First-time bringup safety: ``required: false`` so missing docker.env doesn't crash.

    Critical reviewer finding — without ``required: false`` (the Docker Compose
    default), ``docker compose up`` fails immediately with "env file not found"
    on any host that hasn't yet provisioned ``/etc/mousedroid/docker.env``.
    This also breaks the ``docker compose config --quiet`` syntax-check step
    in CI where the file is absent.
    """
    svc = _load_service()
    env_files = svc.get("env_file", [])
    long_form_entries = [e for e in env_files if isinstance(e, dict)]
    assert long_form_entries, (
        "env_file must use long-form ``{path:, required: false}`` so first-time "
        "deployments don't crash"
    )
    docker_env = next(
        (e for e in long_form_entries if "/etc/mousedroid/docker.env" in str(e.get("path", ""))),
        None,
    )
    assert docker_env is not None, "long-form env_file entry for docker.env must exist"
    assert (
        docker_env.get("required") is False
    ), f"docker.env entry must declare ``required: false``; got {docker_env!r}"


def test_mock_hardware_is_supplied_by_env_file_not_inline() -> None:
    """``MOUSEDROID_MOCK_HARDWARE`` must come from env_file, not inline environment.

    Copilot review of PR #101 caught that inline ``environment:`` entries
    ALWAYS override ``env_file:`` values per Compose spec. The earlier draft
    had ``MOUSEDROID_MOCK_HARDWARE=${VAR:-false}`` inline, which silently
    overrode whatever operators set in /etc/mousedroid/docker.env. The fix
    is to leave the variable OUT of the inline block so the env_file value
    is honoured. Operators control the value via docker.env (single source
    of truth, no precedence ambiguity).
    """
    svc = _load_service()
    env: list[str] = svc.get("environment", [])
    inline_mock = [e for e in env if "MOUSEDROID_MOCK_HARDWARE" in e]
    assert inline_mock == [], (
        "MOUSEDROID_MOCK_HARDWARE must NOT appear in inline environment: "
        "(env_file is the source of truth). Got: "
        f"{inline_mock!r}"
    )


def test_telemetry_token_is_supplied_by_env_file_not_inline() -> None:
    """``MOUSEDROID_TELEMETRY_TOKEN`` must come from env_file, not inline environment.

    Copilot HIGH finding on PR #101: the inline default ``${VAR:-}``
    substitutes EMPTY when the host shell var is unset, silently overriding
    whatever ``/etc/mousedroid/docker.env`` provides — reintroducing the
    TelemetryConfigError crash-loop that F-014 was meant to fix. The token
    MUST be supplied by env_file alone (the inline block must omit it).
    """
    svc = _load_service()
    env: list[str] = svc.get("environment", [])
    inline_token = [e for e in env if "MOUSEDROID_TELEMETRY_TOKEN" in e]
    assert inline_token == [], (
        "MOUSEDROID_TELEMETRY_TOKEN must NOT appear in inline environment: "
        "(env_file is the source of truth — inline empty default would "
        "silently mask the docker.env value). Got: "
        f"{inline_token!r}"
    )


def test_env_jetson_example_is_checked_in_and_documents_token() -> None:
    """The operator template at config/.env.jetson.example is committed + canonical.

    Operators copy this file to /etc/mousedroid/docker.env and fill in the
    token. If the template ever loses the token line the runbook breaks.
    """
    template = _REPO_ROOT / "config" / ".env.jetson.example"
    assert template.exists(), "config/.env.jetson.example must be checked in"
    text = template.read_text(encoding="utf-8")
    assert (
        "MOUSEDROID_TELEMETRY_TOKEN" in text
    ), "template must document the telemetry token setting"
    assert (
        "MOUSEDROID_MOCK_HARDWARE=false" in text
    ), "template must show the production-default (real hardware) value"
