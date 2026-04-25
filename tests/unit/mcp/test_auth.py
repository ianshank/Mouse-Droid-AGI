from __future__ import annotations

import pytest

from mousedroid.mcp.auth import BearerTokenValidator, MCPAuthError


class TestBearerTokenValidator:
    def test_not_required_passes_anything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_TEST_TOKEN", raising=False)
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=False)
        assert v.validate(None) is True
        assert v.validate("anything") is True
        assert v.required is False

    def test_required_without_secret_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_TEST_TOKEN", raising=False)
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        with pytest.raises(MCPAuthError, match="MOUSEDROID_TEST_TOKEN"):
            v.validate("anything")

    def test_required_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_TEST_TOKEN", "expected")
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        with pytest.raises(MCPAuthError, match="missing bearer token"):
            v.validate(None)

    def test_required_invalid_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_TEST_TOKEN", "expected")
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        with pytest.raises(MCPAuthError, match="invalid bearer token"):
            v.validate("wrong")

    def test_required_valid_token_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_TEST_TOKEN", "expected")
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        assert v.validate("expected") is True

    def test_constant_time_compare_does_not_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sanity check: both empty-prefix and equal-length wrong tokens
        # raise the same error type — no early success leak.
        monkeypatch.setenv("MOUSEDROID_TEST_TOKEN", "abcdefgh")
        v = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        for candidate in ["", "x", "abcdefgX", "abcdefgh!"]:
            with pytest.raises(MCPAuthError):
                v.validate(candidate)

    def test_has_secret_reflects_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_TEST_TOKEN", raising=False)
        v_missing = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        assert v_missing.has_secret is False

        monkeypatch.setenv("MOUSEDROID_TEST_TOKEN", "x")
        v_present = BearerTokenValidator("MOUSEDROID_TEST_TOKEN", required=True)
        assert v_present.has_secret is True
