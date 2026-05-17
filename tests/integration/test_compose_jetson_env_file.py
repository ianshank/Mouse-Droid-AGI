"""F-014 regression: docker-compose.jetson.yml env_file + defaults + token forwarding.

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

These tests parse the YAML directly (no Docker daemon required) and pin the
fix end-to-end: env_file declared, production default is real-hardware,
telemetry token is forwarded, and the operator-template file documents the
canonical settings.
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
    return services["mousedroid"]


def test_compose_declares_env_file_directive() -> None:
    """``env_file: /etc/mousedroid/docker.env`` is the source of truth for env.

    Without this directive the compose `environment:` block can't pick up
    the operator's docker.env values; the production restart this fix is
    addressing crashed exactly because the telemetry token never reached
    the container.
    """
    svc = _load_service()
    env_files = svc.get("env_file", [])
    assert env_files, "compose service must declare env_file (see F-014)"
    assert any(
        "/etc/mousedroid/docker.env" in p for p in env_files
    ), f"env_file must include /etc/mousedroid/docker.env; got {env_files!r}"


def test_mock_hardware_default_is_false_not_true() -> None:
    """Production deployments must default to real hardware (F-014).

    The previous ``:-true`` default meant every restart silently flipped
    the container into mock mode, hiding real-hardware regressions.
    """
    svc = _load_service()
    env: list[str] = svc.get("environment", [])
    mock_line = next((e for e in env if "MOUSEDROID_MOCK_HARDWARE=" in e), None)
    assert mock_line is not None, "MOUSEDROID_MOCK_HARDWARE must be set in environment"
    assert (
        ":-false" in mock_line
    ), f"MOUSEDROID_MOCK_HARDWARE compose default must be ``:-false``: {mock_line!r}"


def test_telemetry_token_is_forwarded_to_container() -> None:
    """``MOUSEDROID_TELEMETRY_TOKEN`` reaches the container so auth_enabled works.

    Without this line, the container raises ``TelemetryConfigError`` and
    crash-loops on every restart when ``telemetry.auth.auth_enabled=true``
    in jetson_production.yaml (the production default).
    """
    svc = _load_service()
    env: list[str] = svc.get("environment", [])
    assert any(
        "MOUSEDROID_TELEMETRY_TOKEN=" in e for e in env
    ), "compose environment must forward MOUSEDROID_TELEMETRY_TOKEN (see F-014)"


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
