"""Unit tests for :class:`BearerAuthMiddleware`."""

from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mousedroid.mcp.auth import BearerTokenValidator
from mousedroid.mcp.middleware import BearerAuthMiddleware

_TOKEN_ENV = "_MOUSEDROID_TEST_MCP_TOKEN"  # noqa: S105 - env var name, not secret


def _validator(monkeypatch: pytest.MonkeyPatch, token: str | None) -> BearerTokenValidator:
    if token is None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_ENV, token)
    return BearerTokenValidator(_TOKEN_ENV, required=True)


async def _ok(_request: object) -> JSONResponse:
    return JSONResponse({"ok": True})


def _app(validator: BearerTokenValidator, *, exempt: tuple[str, ...] = ()) -> Starlette:
    return Starlette(
        routes=[Route("/health", _ok), Route("/echo", _ok)],
        middleware=[
            Middleware(BearerAuthMiddleware, validator=validator, exempt_paths=exempt),
        ],
    )


def test_valid_bearer_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator))
    resp = client.get("/echo", headers={"Authorization": "Bearer secret-1"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_missing_authorization_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator))
    resp = client.get("/echo")
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"
    assert resp.headers["WWW-Authenticate"].startswith("Bearer")


def test_wrong_scheme_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator))
    resp = client.get("/echo", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401


def test_wrong_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator))
    resp = client.get("/echo", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_exempt_path_skips_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator, exempt=("/health",)))
    resp = client.get("/health")
    assert resp.status_code == 200


def test_case_insensitive_bearer_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator(monkeypatch, "secret-1")
    client = TestClient(_app(validator))
    resp = client.get("/echo", headers={"Authorization": "bEaReR secret-1"})
    assert resp.status_code == 200


def test_constant_time_comparison_does_not_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong token of *correct length* still returns 401.

    Validates that we are not relying on a length check as a shortcut —
    hmac.compare_digest is what actually rejects the request.
    """
    validator = _validator(monkeypatch, "abc12345")
    client = TestClient(_app(validator))
    resp = client.get("/echo", headers={"Authorization": "Bearer xyz98765"})
    assert resp.status_code == 401


@settings(max_examples=30, deadline=None)
@given(
    raw=st.text(
        # HTTP headers must be ASCII; non-ASCII is rejected at the
        # transport layer (httpx), which is not what we are testing.
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0,
        max_size=200,
    ),
)
def test_property_random_authorization_never_raises(raw: str) -> None:
    """No malformed Authorization header crashes the middleware."""
    import os

    prev = os.environ.get(_TOKEN_ENV)
    os.environ[_TOKEN_ENV] = "expected"
    try:
        validator = BearerTokenValidator(_TOKEN_ENV, required=True)
        client = TestClient(_app(validator))
        resp = client.get("/echo", headers={"Authorization": raw})
        # Either valid (200) or unauthorized (401) — never 500.
        assert resp.status_code in {200, 401}
    finally:
        if prev is None:
            os.environ.pop(_TOKEN_ENV, None)
        else:
            os.environ[_TOKEN_ENV] = prev
