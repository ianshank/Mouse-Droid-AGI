"""Unit tests for scripts/jetson_probe_logs_ws.py (the P13 log-stream probe).

Regression coverage for a real bug a GitHub Copilot review comment surfaced
on PR #192: ``_auth_headers()`` sent ``X-Telemetry-Api-Key``, a header name
the server's actual auth layer has never recognized — only ``X-API-Key``
(``telemetry/server/_lifecycle.py::_build_middlewares``) is a valid legacy
API-key header. The script's API-key auth path was silently non-functional
from the day it was written; nothing caught it because this file had zero
test coverage. This file closes that gap.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from tests._script_loader import load_script_module


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    """Load the script module by path once per test module (it lives in scripts/)."""
    return load_script_module("jetson_probe_logs_ws")


def test_build_url_uses_defaults(probe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_HOST", raising=False)
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_PORT", raising=False)
    monkeypatch.delenv("MOUSEDROID_LOGS_WS_PATH", raising=False)

    assert probe._build_url() == "ws://127.0.0.1:8080/api/v1/logs/stream"


def test_build_url_honours_env_overrides(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_HOST", "rover.local")
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_PORT", "9090")
    monkeypatch.setenv("MOUSEDROID_LOGS_WS_PATH", "/custom/path")

    assert probe._build_url() == "ws://rover.local:9090/custom/path"


def test_auth_headers_empty_when_no_env_set(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_TOKEN", raising=False)
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_API_KEY", raising=False)

    assert probe._auth_headers() == {}


def test_auth_headers_bearer_token_only(probe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_TOKEN", "secret-token")
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_API_KEY", raising=False)

    assert probe._auth_headers() == {"Authorization": "Bearer secret-token"}


def test_auth_headers_api_key_uses_x_api_key_not_legacy_name(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression pin: the header MUST be ``X-API-Key``.

    The server's legacy auth middleware (``_lifecycle.py``) checks
    ``request.headers.get("X-API-Key", ...)`` — never
    ``X-Telemetry-Api-Key``. Sending the wrong header name means the probe
    always gets rejected with a configured server-side API key, silently
    turning a real auth check into a permanent false failure.
    """
    monkeypatch.delenv("MOUSEDROID_TELEMETRY_TOKEN", raising=False)
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_API_KEY", "my-api-key")

    headers = probe._auth_headers()

    assert headers == {"X-API-Key": "my-api-key"}
    assert "X-Telemetry-Api-Key" not in headers


def test_auth_headers_both_token_and_api_key_set(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_TOKEN", "secret-token")
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_API_KEY", "my-api-key")

    headers = probe._auth_headers()

    assert headers == {
        "Authorization": "Bearer secret-token",
        "X-API-Key": "my-api-key",
    }


def test_auth_headers_strips_whitespace(probe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_TOKEN", "  secret-token  ")
    monkeypatch.setenv("MOUSEDROID_TELEMETRY_API_KEY", "  my-api-key  ")

    headers = probe._auth_headers()

    assert headers == {
        "Authorization": "Bearer secret-token",
        "X-API-Key": "my-api-key",
    }
