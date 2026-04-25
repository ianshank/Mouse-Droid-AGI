"""Bearer-token validation for the MCP server.

The token is sourced exclusively from an environment variable (whose
name is config-driven via :attr:`MCPConfig.auth_token_env_var`); it is
NEVER read from YAML so secrets cannot leak into version control.

For ``stdio`` transport authentication is a no-op because the parent
process owns the connection. For network transports a token is required
when binding to non-loopback (enforced by :class:`MCPConfig` at
configuration load time).
"""

from __future__ import annotations

import hmac
import os

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MCPAuthError(Exception):
    """Raised when an MCP request fails authentication."""


class BearerTokenValidator:
    """Validates a bearer token against a server-side secret.

    The expected secret is loaded once at construction; rotating the
    secret therefore requires restarting the MCP server (matching the
    operational story of every other env-var-driven secret in the
    codebase).
    """

    def __init__(self, env_var: str, *, required: bool) -> None:
        """Initialise the validator.

        Args:
            env_var: Name of the environment variable holding the secret.
                Sourced from :attr:`MCPConfig.auth_token_env_var`.
            required: When True, an unset env var raises
                :class:`MCPAuthError` from :meth:`validate`. When False
                (e.g. stdio transport on loopback), validation is a no-op.
        """
        self._env_var = env_var
        self._required = required
        self._expected: str | None = os.environ.get(env_var)
        if required and not self._expected:
            _log.warning(
                "mcp_auth_token_missing",
                env_var=env_var,
                hint="set the token before starting the MCP server",
            )

    @property
    def required(self) -> bool:
        """Whether authentication is enforced for this validator."""
        return self._required

    @property
    def has_secret(self) -> bool:
        """Whether a server-side secret was successfully loaded."""
        return self._expected is not None

    def validate(self, presented: str | None) -> bool:
        """Constant-time-compare a presented token against the secret.

        Args:
            presented: The bearer token from the incoming request (may
                be ``None`` if the client omitted it).

        Returns:
            ``True`` if validation succeeded (or auth is not required).

        Raises:
            MCPAuthError: When auth is required and either the server
                secret is unset, the presented token is missing, or the
                tokens do not match.
        """
        if not self._required:
            return True
        if self._expected is None:
            msg = f"MCP auth required but {self._env_var} is unset; refusing all requests"
            _log.warning("mcp_auth_secret_unset", env_var=self._env_var)
            raise MCPAuthError(msg)
        if presented is None:
            _log.info("mcp_auth_missing_token")
            raise MCPAuthError("missing bearer token")
        # hmac.compare_digest is constant-time; operate on bytes for safety.
        if not hmac.compare_digest(presented.encode("utf-8"), self._expected.encode("utf-8")):
            _log.info("mcp_auth_invalid_token")
            raise MCPAuthError("invalid bearer token")
        return True
