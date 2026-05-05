from __future__ import annotations

import pytest

from mousedroid.config.schema import MCPConfig, MCPResourcesConfig, Settings


class TestMCPConfigDefaults:
    def test_default_disabled(self) -> None:
        cfg = MCPConfig()
        assert cfg.enabled is False
        assert cfg.transport == "stdio"
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8765

    def test_default_resources(self) -> None:
        cfg = MCPConfig()
        assert isinstance(cfg.resources, MCPResourcesConfig)
        assert cfg.resources.recent_frames_max == 64
        assert cfg.resources.log_tail_max == 200
        assert cfg.resources.config_cache_ttl_s == pytest.approx(1.0)
        assert cfg.resources.memory_enabled is False

    def test_default_actuation_tools_listed(self) -> None:
        cfg = MCPConfig()
        # Defaults are config-driven, not hardcoded at the call site;
        # any side-effecting tool must appear here.
        assert "calibrate_ultrasonic" in cfg.actuation_tools
        assert "tensorrt_compile" in cfg.actuation_tools
        assert "export_experience" in cfg.actuation_tools

    def test_redact_pattern_matches_secrets(self) -> None:
        import re

        cfg = MCPConfig()
        pat = re.compile(cfg.redact_key_pattern)
        for key in ["api_key", "API_KEY", "token", "secret", "credentials", "password"]:
            assert pat.search(key)


class TestMCPConfigValidators:
    def test_health_check_cannot_be_denied(self) -> None:
        with pytest.raises(ValueError, match="health_check"):
            MCPConfig.model_validate({"tools_denylist": ["health_check"]})

    def test_remote_transport_requires_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
        any_iface = "0.0.0.0"  # noqa: S104  - validating the all-interfaces guard
        with pytest.raises(ValueError, match="MOUSEDROID_MCP_TOKEN"):
            MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": any_iface})

    def test_remote_transport_with_token_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        any_iface = "0.0.0.0"  # noqa: S104
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": any_iface})
        assert cfg.host == any_iface

    def test_loopback_does_not_require_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "127.0.0.1"})
        assert cfg.host == "127.0.0.1"

    def test_stdio_does_not_require_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
        any_iface = "0.0.0.0"  # noqa: S104
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "stdio", "host": any_iface})
        assert cfg.host == any_iface

    def test_bind_transport_with_stdio_allowed(self) -> None:
        cfg = MCPConfig.model_validate(
            {"enabled": True, "transport": "stdio", "bind_transport": True}
        )
        assert cfg.bind_transport is True

    def test_bind_transport_with_sse_loopback_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SSE on loopback with a token loads cleanly (Phase B wired the transport)."""
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "transport": "sse",
                "host": "127.0.0.1",
                "bind_transport": True,
            }
        )
        assert cfg.bind_transport is True
        assert cfg.transport == "sse"

    def test_bind_transport_with_streamable_http_loopback_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """streamable_http on loopback with a token loads cleanly."""
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "transport": "streamable_http",
                "host": "127.0.0.1",
                "bind_transport": True,
            }
        )
        assert cfg.bind_transport is True
        assert cfg.transport == "streamable_http"

    def test_bind_external_required_for_non_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-loopback host without bind_external=true fails fast at config load."""
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        with pytest.raises(ValueError, match="bind_external"):
            MCPConfig.model_validate(
                {
                    "enabled": True,
                    "transport": "sse",
                    "host": "0.0.0.0",  # noqa: S104
                    "bind_transport": True,
                }
            )

    def test_bind_external_requires_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bind_external=true with no token in env fails fast."""
        monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
        with pytest.raises(ValueError, match="MOUSEDROID_MCP_TOKEN"):
            MCPConfig.model_validate(
                {
                    "enabled": True,
                    "transport": "sse",
                    "host": "0.0.0.0",  # noqa: S104
                    "bind_transport": True,
                    "bind_external": True,
                }
            )

    def test_bind_external_with_stdio_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bind_external=true requires a network transport (stdio has no listener)."""
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        with pytest.raises(ValueError, match="network transport"):
            MCPConfig.model_validate(
                {
                    "enabled": True,
                    "transport": "stdio",
                    "host": "127.0.0.1",
                    "bind_transport": True,
                    "bind_external": True,
                }
            )

    def test_bind_transport_false_with_sse_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without bind_transport, SSE config still loads (no socket bound)."""
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "abc123")
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "127.0.0.1"})
        assert cfg.transport == "sse"
        assert cfg.bind_transport is False


class TestSettingsIntegration:
    def test_settings_mcp_default_none(self) -> None:
        s = Settings.model_validate({"mock_hardware": True})
        assert s.mcp is None

    def test_settings_with_mcp_block(self) -> None:
        s = Settings.model_validate(
            {
                "mock_hardware": True,
                "mcp": {"enabled": True, "transport": "stdio"},
            }
        )
        assert s.mcp is not None
        assert s.mcp.transport == "stdio"

    def test_env_var_override_via_nested_delimiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pydantic env nesting cannot synthesize the optional MCPConfig
        # parent, but it must still override Settings.mock_hardware via
        # the documented prefix scheme. Spot-check the prefix is honored.
        monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
        s = Settings()
        assert s.mock_hardware is True
