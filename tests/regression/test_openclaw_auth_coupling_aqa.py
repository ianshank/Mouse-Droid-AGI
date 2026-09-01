"""AQA pin — OpenClaw's actuation endpoint must require telemetry auth.

Closes a security gap found in a 2026-09-01 audit: ``openclaw.enabled=true``
wires up ``POST /api/v1/mission`` (real GoalVector actuation dispatch)
independent of ``telemetry.auth.auth_enabled`` — unlike the MCP surface,
which already refuses a non-loopback bind without an auth token via
``harness_mcp.py``'s ``_require_token_for_remote``. An operator flipping
only ``openclaw.enabled: true`` in a dev config previously got an
unauthenticated actuation endpoint reachable on the LAN.

``Settings.openclaw_requires_telemetry_auth`` (``config/schema/root.py``)
closes this by refusing to construct a ``Settings`` instance with
``openclaw.enabled=true`` unless telemetry auth is configured via either
mechanism the server actually checks (``telemetry/server/_lifecycle.py``):
bearer-token auth (``telemetry.auth.auth_enabled``) or the legacy
X-API-Key (``telemetry.api_key``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import (
    OpenClawConfig,
    Settings,
    TelemetryAuthConfig,
    TelemetryConfig,
)


def test_openclaw_enabled_without_any_auth_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"openclaw\.enabled=true requires telemetry auth"):
        Settings(
            mock_hardware=True,
            openclaw=OpenClawConfig(enabled=True),
            telemetry=TelemetryConfig(enabled=True),
        )


def test_openclaw_enabled_with_auth_disabled_is_rejected() -> None:
    """``telemetry.auth`` present but ``auth_enabled=False`` — still not auth."""
    with pytest.raises(ValidationError, match=r"openclaw\.enabled=true requires telemetry auth"):
        Settings(
            mock_hardware=True,
            openclaw=OpenClawConfig(enabled=True),
            telemetry=TelemetryConfig(enabled=True, auth=TelemetryAuthConfig(auth_enabled=False)),
        )


def test_openclaw_enabled_with_bearer_auth_is_accepted() -> None:
    s = Settings(
        mock_hardware=True,
        openclaw=OpenClawConfig(enabled=True),
        telemetry=TelemetryConfig(enabled=True, auth=TelemetryAuthConfig(auth_enabled=True)),
    )
    assert s.openclaw is not None
    assert s.openclaw.enabled is True


def test_openclaw_enabled_with_legacy_api_key_is_accepted() -> None:
    """Legacy X-API-Key alone (no bearer auth block) also satisfies the guard."""
    s = Settings(
        mock_hardware=True,
        openclaw=OpenClawConfig(enabled=True),
        telemetry=TelemetryConfig(enabled=True, api_key="a-secret-key"),  # type: ignore[arg-type]
    )
    assert s.openclaw is not None
    assert s.telemetry.api_key is not None


def test_openclaw_disabled_never_triggers_the_guard() -> None:
    """Backwards-compat: openclaw=None (the default) or enabled=False never raises."""
    Settings(mock_hardware=True, telemetry=TelemetryConfig(enabled=True))
    Settings(
        mock_hardware=True,
        openclaw=OpenClawConfig(enabled=False),
        telemetry=TelemetryConfig(enabled=True),
    )


def test_telemetry_config_rejects_empty_string_api_key() -> None:
    """An empty string is not a real key — reject it at the field, not just here.

    Found in a follow-up audit of this file's own guard: ``api_key=""`` used
    to construct fine (``SecretStr | None`` had no length constraint), which
    made ``telemetry.api_key is not None`` — this guard's own legacy-key
    check — true for an empty key. At runtime the legacy X-API-Key middleware
    (``telemetry/server/_lifecycle.py``) would then accept any unauthenticated
    request, since ``hmac.compare_digest("", "")`` is ``True``.
    """
    with pytest.raises(ValidationError, match="too_short"):
        TelemetryConfig(api_key="")  # type: ignore[arg-type]


def test_openclaw_enabled_with_empty_string_api_key_is_rejected() -> None:
    """Whole-chain proof: this guard's own ``is not None`` check cannot see an
    empty-but-truthy key — the field-level ``min_length=1`` constraint is what
    actually closes the gap, so prove it closes it through ``Settings`` too.
    """
    with pytest.raises(ValidationError):
        Settings(
            mock_hardware=True,
            openclaw=OpenClawConfig(enabled=True),
            telemetry=TelemetryConfig(enabled=True, api_key=""),  # type: ignore[arg-type]
        )
