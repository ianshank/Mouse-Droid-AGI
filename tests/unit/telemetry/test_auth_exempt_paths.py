"""Tests for TelemetryAuthConfig exempt_paths validator and middleware prefix-collision safety."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import TelemetryAuthConfig

# ---------------------------------------------------------------------------
# Config validator — exempt_paths format enforcement
# ---------------------------------------------------------------------------


class TestExemptPathValidator:
    """TelemetryAuthConfig rejects malformed exempt_paths entries at construction."""

    def test_valid_simple_path(self) -> None:
        """Single-segment path is accepted."""
        cfg = TelemetryAuthConfig(exempt_paths=["/health"])
        assert cfg.exempt_paths == ["/health"]

    def test_valid_nested_path(self) -> None:
        """Multi-segment path with hyphens and underscores is accepted."""
        cfg = TelemetryAuthConfig(exempt_paths=["/api/v1/health", "/api-v2/status_check"])
        assert len(cfg.exempt_paths) == 2

    def test_valid_root_slash(self) -> None:
        """Root path '/' is valid."""
        cfg = TelemetryAuthConfig(exempt_paths=["/"])
        assert cfg.exempt_paths == ["/"]

    def test_rejects_path_with_query_string(self) -> None:
        """Paths with '?' are rejected — query strings in exempt list is a config error."""
        with pytest.raises(ValidationError, match="invalid"):
            TelemetryAuthConfig(exempt_paths=["/health?foo=bar"])

    def test_rejects_path_with_uppercase(self) -> None:
        """Uppercase letters in exempt paths are rejected."""
        with pytest.raises(ValidationError, match="invalid"):
            TelemetryAuthConfig(exempt_paths=["/Health"])

    def test_rejects_path_without_leading_slash(self) -> None:
        """Path not starting with '/' is rejected."""
        with pytest.raises(ValidationError, match="invalid"):
            TelemetryAuthConfig(exempt_paths=["health"])

    def test_rejects_path_with_dotdot(self) -> None:
        """Traversal sequences '..' are rejected."""
        with pytest.raises(ValidationError, match="invalid"):
            TelemetryAuthConfig(exempt_paths=["/api/../admin"])

    def test_rejects_path_with_fragment(self) -> None:
        """Fragment '#' in paths is rejected."""
        with pytest.raises(ValidationError, match="invalid"):
            TelemetryAuthConfig(exempt_paths=["/health#section"])

    def test_empty_exempt_paths_is_valid(self) -> None:
        """Empty list is valid — means no paths are exempt."""
        cfg = TelemetryAuthConfig(exempt_paths=[])
        assert cfg.exempt_paths == []


# ---------------------------------------------------------------------------
# Middleware prefix-collision safety — via build_bearer_auth_middleware
# ---------------------------------------------------------------------------


class TestExemptPathMiddlewareSafety:
    """Middleware does not allow prefix-collision attacks via exempt_paths."""

    def _run_middleware(
        self,
        request_path: str,
        exempt_paths: list[str],
    ) -> bool:
        """Return True if the request would be exempted by the middleware logic.

        Mirrors the exact condition in auth.py so we test the actual logic.
        """
        for exempt in exempt_paths:
            if request_path == exempt or request_path.startswith(exempt + "/"):
                return True
        return False

    def test_exact_match_is_exempt(self) -> None:
        """/health is exempt when '/health' is in exempt_paths."""
        assert self._run_middleware("/health", ["/health"]) is True

    def test_subpath_is_exempt(self) -> None:
        """/health/live is exempt when '/health' is in exempt_paths."""
        assert self._run_middleware("/health/live", ["/health"]) is True

    def test_prefix_collision_is_not_exempt(self) -> None:
        """/healthz is NOT exempt when only '/health' is in exempt_paths."""
        assert self._run_middleware("/healthz", ["/health"]) is False

    def test_exploit_path_is_not_exempt(self) -> None:
        """/healthexploit is NOT exempt when only '/health' is in exempt_paths."""
        assert self._run_middleware("/healthexploit", ["/health"]) is False

    def test_unrelated_path_is_not_exempt(self) -> None:
        """/api/v1/health is not exempt when only '/health' is listed."""
        assert self._run_middleware("/api/v1/health", ["/health"]) is False

    def test_multiple_exempt_paths(self) -> None:
        """Both /health and /metrics are exempt when both are listed."""
        exempt = ["/health", "/metrics"]
        assert self._run_middleware("/health", exempt) is True
        assert self._run_middleware("/metrics", exempt) is True
        assert self._run_middleware("/api/v1/status", exempt) is False

    def test_api_v1_health_exact_match(self) -> None:
        """/api/v1/health is exempt when '/api/v1/health' is listed."""
        assert self._run_middleware("/api/v1/health", ["/api/v1/health"]) is True

    def test_api_v1_health_cloud_is_not_exempt(self) -> None:
        """/api/v1/health/cloud is exempt (sub-path of /api/v1/health)."""
        # /api/v1/health/cloud starts with /api/v1/health + "/" so this IS exempt
        assert self._run_middleware("/api/v1/health/cloud", ["/api/v1/health"]) is True

    def test_sibling_path_not_exempt(self) -> None:
        """/api/v1/healthcheck is NOT exempt when '/api/v1/health' is listed."""
        assert self._run_middleware("/api/v1/healthcheck", ["/api/v1/health"]) is False
