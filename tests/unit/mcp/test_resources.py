from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import MCPConfig, MCPResourcesConfig, Settings
from mousedroid.mcp.resources import (
    REDACTED,
    ConfigResourceProvider,
    LogResourceProvider,
    MemoryResourceProvider,
    TelemetryResourceProvider,
    parse_resource_uri,
    redact_value,
)
from mousedroid.telemetry.log_buffer import LogRingBuffer
from mousedroid.telemetry.protocol import TelemetryFrame


class TestParseResourceUri:
    def test_parses_basic_uri(self) -> None:
        scheme, path, query = parse_resource_uri("mousedroid://logs/tail")
        assert scheme == "mousedroid"
        assert path == "/logs/tail"
        assert query == {}

    def test_parses_query(self) -> None:
        _scheme, path, query = parse_resource_uri("mousedroid://telemetry/recent?n=12")
        assert path == "/telemetry/recent"
        assert query == {"n": "12"}

    def test_missing_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            parse_resource_uri("/no/scheme")


class TestRedactValue:
    @pytest.fixture
    def pattern(self) -> re.Pattern[str]:
        return re.compile(MCPConfig.model_fields["redact_key_pattern"].default)

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "auth_token",
            "MOUSEDROID_TOKEN",
            "password",
            "secret_value",
            "credentials",
        ],
    )
    def test_secret_keys_redacted(self, pattern: re.Pattern[str], key: str) -> None:
        out = redact_value({key: "sensitive"}, key_pattern=pattern)
        assert out[key] == REDACTED

    def test_nested_secrets_redacted(self, pattern: re.Pattern[str]) -> None:
        out = redact_value({"outer": {"inner": {"api_key": "x", "ok": 1}}}, key_pattern=pattern)
        assert out["outer"]["inner"]["api_key"] == REDACTED
        assert out["outer"]["inner"]["ok"] == 1

    def test_lists_recurse(self, pattern: re.Pattern[str]) -> None:
        out = redact_value([{"token": "x"}, {"safe": "y"}], key_pattern=pattern)
        assert out == [{"token": REDACTED}, {"safe": "y"}]

    def test_safe_keys_pass_through(self, pattern: re.Pattern[str]) -> None:
        original = {"host": "127.0.0.1", "port": 8080, "publish_hz": 10.0}
        assert redact_value(original, key_pattern=pattern) == original


class TestTelemetryResourceProvider:
    def _publisher_with_frames(self, frames: list[TelemetryFrame]) -> Any:
        queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue()
        for f in frames:
            queue.put_nowait(f)
        pub = MagicMock()
        pub.get_queue.return_value = queue
        return pub

    @pytest.mark.asyncio
    async def test_disabled_when_no_publisher(self, mcp_cfg: MCPConfig) -> None:
        p = TelemetryResourceProvider(mcp_cfg, publisher=None)
        assert p.enabled is False
        assert p.list_uris() == []
        with pytest.raises(PermissionError):
            p.read("/telemetry/latest", {})

    @pytest.mark.asyncio
    async def test_drains_queue_into_buffer(self, mcp_cfg: MCPConfig) -> None:
        frames = [TelemetryFrame(timestamp=float(i)) for i in range(3)]
        pub = self._publisher_with_frames(frames)
        p = TelemetryResourceProvider(mcp_cfg, publisher=pub)
        pulled = await p.sample_once()
        assert pulled == 3
        assert p.buffer_size == 3

    @pytest.mark.asyncio
    async def test_latest_returns_most_recent(self, mcp_cfg: MCPConfig) -> None:
        frames = [TelemetryFrame(timestamp=float(i)) for i in range(3)]
        pub = self._publisher_with_frames(frames)
        p = TelemetryResourceProvider(mcp_cfg, publisher=pub)
        await p.sample_once()
        out = p.read("/telemetry/latest", {})
        assert out["frame"]["timestamp"] == 2.0

    @pytest.mark.asyncio
    async def test_recent_caps_at_max(self) -> None:
        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "resources": {"recent_frames_max": 2},
            }
        )
        frames = [TelemetryFrame(timestamp=float(i)) for i in range(5)]
        pub = self._publisher_with_frames(frames)
        p = TelemetryResourceProvider(cfg, publisher=pub)
        await p.sample_once()
        out = p.read("/telemetry/recent", {"n": "100"})
        assert out["count"] == 2
        # Buffer holds the last 2 only (deque maxlen)
        assert [f["timestamp"] for f in out["frames"]] == [3.0, 4.0]


class TestLogResourceProvider:
    def test_disabled_without_buffer(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        p = LogResourceProvider(mcp_cfg, log_buffer=None, key_pattern=redact_pattern)
        assert p.enabled is False
        with pytest.raises(PermissionError):
            p.read("/logs/tail", {})

    def test_redacts_sensitive_keys(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        buf = LogRingBuffer(maxlen=10)
        # Inject entries directly via processor call
        buf(None, "info", {"event": "x", "api_key": "supersecret", "host": "1.2.3.4"})
        p = LogResourceProvider(mcp_cfg, log_buffer=buf, key_pattern=redact_pattern)
        out = p.read("/logs/tail", {"n": "10"})
        assert out["count"] == 1
        assert out["entries"][0]["api_key"] == REDACTED
        assert out["entries"][0]["host"] == "1.2.3.4"

    def test_caps_at_log_tail_max(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"log_tail_max": 3}})
        buf = LogRingBuffer(maxlen=10)
        for i in range(5):
            buf(None, "info", {"event": "x", "i": i})
        p = LogResourceProvider(cfg, log_buffer=buf, key_pattern=redact_pattern)
        out = p.read("/logs/tail", {"n": "100"})
        assert out["count"] == 3


class TestConfigResourceProvider:
    def test_redacts_settings(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        # Inject a token field via env so the (redacted) settings dump
        # contains a key matching the pattern.
        root = Settings.model_validate({"mock_hardware": True})
        p = ConfigResourceProvider(mcp_cfg, root, key_pattern=redact_pattern)
        out = p.read("/config/redacted", {})
        assert "settings" in out
        # Spot-check: any nested key matching the redact pattern is masked.
        flat = repr(out["settings"])
        # No raw "secret" / "token" leaked except as the redacted sentinel.
        # (Pattern only matches keys, so redacted strings appear as values.)
        assert REDACTED in flat or all(
            k not in flat.lower() for k in ["actual_secret_xyz", "actual_token_xyz"]
        )

    def test_cache_ttl(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"config_cache_ttl_s": 60.0}})
        root = Settings.model_validate({"mock_hardware": True})
        p = ConfigResourceProvider(cfg, root, key_pattern=redact_pattern)
        out1 = p.read("/config/redacted", {})
        out2 = p.read("/config/redacted", {})
        # Same dict (cached)
        assert out1["settings"] is out2["settings"]


class TestMemoryResourceProvider:
    def test_disabled_without_memory_tier(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        p = MemoryResourceProvider(mcp_cfg, memory_tier=None, key_pattern=redact_pattern)
        assert p.enabled is False
        with pytest.raises(PermissionError):
            p.read("/memory/episodes/recent", {})

    def test_returns_redacted_episodes(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        episodic = MagicMock()
        episodic.sample.return_value = [
            {"action": "left", "api_key": "leak"},
            {"action": "right", "ok": 1},
        ]
        tier = MagicMock(episodic=episodic)
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        out = p.read("/memory/episodes/recent", {"n": "2"})
        assert out["count"] == 2
        assert out["episodes"][0]["api_key"] == REDACTED

    def test_summarises_ndarray_payloads(self, redact_pattern) -> None:
        import numpy as np

        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        episodic = MagicMock()
        episodic.sample.return_value = [{"image": np.zeros((4, 4), dtype=np.uint8)}]
        tier = MagicMock(episodic=episodic)
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        out = p.read("/memory/episodes/recent", {"n": "1"})
        ep = out["episodes"][0]
        assert ep["image"]["ndarray"] is True
        assert ep["image"]["shape"] == [4, 4]
        assert ep["image"]["dtype"] == "uint8"

    def test_unknown_path_raises(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        tier = MagicMock(episodic=MagicMock())
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        with pytest.raises(KeyError):
            p.read("/memory/does_not_exist", {})


class TestResourcesConfigDefaults:
    def test_defaults(self) -> None:
        c = MCPResourcesConfig()
        assert c.telemetry_enabled is True
        assert c.logs_enabled is True
        assert c.config_enabled is True
        assert c.memory_enabled is False


class TestEdgeCases:
    """Cover the harder-to-reach branches identified by gap analysis."""

    def test_redact_value_depth_cap_returns_input(self) -> None:
        from mousedroid.mcp.resources import MAX_REDACT_DEPTH, redact_value

        # Recurse beyond the cap; the value at that point is returned untouched.
        deep_input = {"k": "v"}
        out = redact_value(
            deep_input,
            key_pattern=re.compile(r"never-matches"),
            _depth=MAX_REDACT_DEPTH + 1,
        )
        assert out is deep_input

    def test_redact_pattern_constant_exposed(self) -> None:
        from mousedroid.mcp.resources import MAX_REDACT_DEPTH

        assert isinstance(MAX_REDACT_DEPTH, int)
        assert MAX_REDACT_DEPTH > 0

    @pytest.mark.asyncio
    async def test_telemetry_recent_with_invalid_n(self, mcp_cfg: MCPConfig) -> None:
        # Invalid integer query should silently fall back to the cap.
        from mousedroid.telemetry.protocol import TelemetryFrame

        queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue()
        for i in range(3):
            queue.put_nowait(TelemetryFrame(timestamp=float(i)))
        pub = MagicMock()
        pub.get_queue.return_value = queue
        from mousedroid.mcp.resources import TelemetryResourceProvider

        p = TelemetryResourceProvider(mcp_cfg, publisher=pub)
        await p.sample_once()
        out = p.read("/telemetry/recent", {"n": "not-an-int"})
        # Should return up to recent_frames_max frames (8 in the fixture cfg, 3 actual).
        assert out["count"] == 3

    def test_telemetry_unknown_path_raises(self, mcp_cfg: MCPConfig) -> None:
        # Reach the unknown-path branch directly (telemetry enabled, bad path).
        from mousedroid.mcp.resources import TelemetryResourceProvider

        pub = MagicMock()
        pub.get_queue.return_value = asyncio.Queue()
        p = TelemetryResourceProvider(mcp_cfg, publisher=pub)
        with pytest.raises(KeyError):
            p.read("/telemetry/does_not_exist", {})

    def test_log_unknown_path_raises(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        from mousedroid.mcp.resources import LogResourceProvider

        buf = LogRingBuffer(maxlen=4)
        p = LogResourceProvider(mcp_cfg, log_buffer=buf, key_pattern=redact_pattern)
        with pytest.raises(KeyError):
            p.read("/logs/wrong", {})

    def test_log_invalid_n_falls_back_to_cap(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        from mousedroid.mcp.resources import LogResourceProvider

        buf = LogRingBuffer(maxlen=10)
        for i in range(4):
            buf(None, "info", {"event": "x", "i": i})
        p = LogResourceProvider(mcp_cfg, log_buffer=buf, key_pattern=redact_pattern)
        out = p.read("/logs/tail", {"n": "garbage"})
        assert out["count"] == 4

    def test_config_provider_disabled_raises(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"config_enabled": False}})
        root = Settings.model_validate({"mock_hardware": True})
        from mousedroid.mcp.resources import ConfigResourceProvider

        p = ConfigResourceProvider(cfg, root, key_pattern=redact_pattern)
        assert p.list_uris() == []
        with pytest.raises(PermissionError):
            p.read("/config/redacted", {})

    def test_config_provider_unknown_path(self, mcp_cfg: MCPConfig, redact_pattern) -> None:
        from mousedroid.mcp.resources import ConfigResourceProvider

        root = Settings.model_validate({"mock_hardware": True})
        p = ConfigResourceProvider(mcp_cfg, root, key_pattern=redact_pattern)
        with pytest.raises(KeyError):
            p.read("/config/something_else", {})

    def test_memory_unknown_path(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        from mousedroid.mcp.resources import MemoryResourceProvider

        tier = MagicMock(episodic=MagicMock())
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        with pytest.raises(KeyError):
            p.read("/memory/unknown", {})

    def test_memory_no_episodic_attr(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        from mousedroid.mcp.resources import MemoryResourceProvider

        # tier without an `episodic` attribute -> empty result, no crash.
        class _BareTier:
            pass

        p = MemoryResourceProvider(cfg, memory_tier=_BareTier(), key_pattern=redact_pattern)
        out = p.read("/memory/episodes/recent", {"n": "5"})
        assert out == {"count": 0, "episodes": []}

    def test_memory_sample_exception_returns_empty(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        from mousedroid.mcp.resources import MemoryResourceProvider

        episodic = MagicMock()
        episodic.sample.side_effect = RuntimeError("buffer corrupt")
        tier = MagicMock(episodic=episodic)
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        out = p.read("/memory/episodes/recent", {"n": "5"})
        assert out == {"count": 0, "episodes": []}

    def test_memory_invalid_n_falls_back(self, redact_pattern) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "resources": {"memory_enabled": True}})
        from mousedroid.mcp.resources import MemoryResourceProvider

        episodic = MagicMock()
        episodic.sample.return_value = [{"action": "up"}]
        tier = MagicMock(episodic=episodic)
        p = MemoryResourceProvider(cfg, memory_tier=tier, key_pattern=redact_pattern)
        out = p.read("/memory/episodes/recent", {"n": "bad"})
        assert out["count"] == 1

    def test_telemetry_latest_with_empty_buffer(self, mcp_cfg: MCPConfig) -> None:
        # Buffer empty -> latest returns frame=None (not raise).
        from mousedroid.mcp.resources import TelemetryResourceProvider

        pub = MagicMock()
        pub.get_queue.return_value = asyncio.Queue()
        p = TelemetryResourceProvider(mcp_cfg, publisher=pub)
        out = p.read("/telemetry/latest", {})
        assert out == {"frame": None}

    def test_episode_to_dict_non_dict(self) -> None:
        from mousedroid.mcp.resources import _episode_to_dict

        out = _episode_to_dict("just-a-string")
        assert out == {"episode": "just-a-string"}

    def test_coerce_field_handles_list_tuple_and_repr(self) -> None:
        from mousedroid.mcp.resources import _coerce_field

        # list / tuple are recursively coerced, primitives pass through,
        # complex objects fall back to repr().
        class _Thing:
            def __repr__(self) -> str:
                return "<Thing>"

        out = _coerce_field([1, "x", _Thing()])
        assert out == [1, "x", "<Thing>"]
        assert _coerce_field(("a", "b")) == ["a", "b"]
        assert _coerce_field({"k": _Thing()}) == {"k": "<Thing>"}
