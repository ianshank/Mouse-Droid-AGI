"""Agent-harness, MCP server, and OpenClaw integration configuration models.

The Model Context Protocol server (resources, transport, rate limiting),
the agent harness (task tracker, journal, hooks, approval gate, skills),
and the OpenClaw multi-channel NL control-plane integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mousedroid.config.schema._primitives import Self, StrictBaseModel, _settings_default_factory
from mousedroid.config.schema.misc import CircuitBreakerConfig


class MCPResourcesConfig(StrictBaseModel):
    """Read-only MCP resource exposure toggles and bounds.

    All limits are config-driven so dashboards and clients can request
    larger or smaller windows without code changes. Defaults mirror the
    telemetry log buffer envelope.
    """

    telemetry_enabled: bool = Field(
        True,
        description="Expose `mousedroid://telemetry/*` resources",
    )
    logs_enabled: bool = Field(
        True,
        description="Expose `mousedroid://logs/tail` resource",
    )
    config_enabled: bool = Field(
        True,
        description="Expose `mousedroid://config/redacted` resource",
    )
    memory_enabled: bool = Field(
        False,
        description="Expose `mousedroid://memory/episodes/recent` resource",
    )
    recent_frames_max: int = Field(
        64,
        gt=0,
        le=4096,
        description="Maximum recent telemetry frames a client may request",
    )
    log_tail_max: int = Field(
        200,
        gt=0,
        le=10_000,
        description="Maximum log entries returnable in a single read",
    )
    config_cache_ttl_s: float = Field(
        1.0,
        gt=0,
        le=60.0,
        description="TTL (seconds) for the redacted-config snapshot cache",
    )


class MCPConfig(StrictBaseModel):
    """Model Context Protocol server configuration.

    The MCP server is fully optional and disabled by default. When
    enabled, it bridges the existing :class:`ToolRegistry`, telemetry
    pipeline, log buffer, and (optionally) episodic memory to any
    MCP-compliant client over stdio, SSE, or streamable HTTP.

    All thresholds, timeouts, and toggles are config-driven; no values
    are hardcoded in the server implementation.
    """

    enabled: bool = Field(False, description="Enable MCP server")
    transport: Literal["stdio", "sse", "streamable_http"] = Field(
        "stdio",
        description="MCP transport protocol",
    )
    host: str = Field(
        "127.0.0.1",
        description="Bind address (loopback by default for safety)",
    )
    port: int = Field(8765, gt=0, le=65535, description="Server port (HTTP/SSE only)")
    auth_token_env_var: str = Field(
        "MOUSEDROID_MCP_TOKEN",
        description="Environment variable holding bearer token (never in YAML)",
    )
    tools_allowlist: list[str] | None = Field(
        None,
        description="Explicit allowlist of tool names; None = all registry tools",
    )
    tools_denylist: list[str] = Field(
        default_factory=list,
        description="Tools that must never be exposed (always wins over allowlist)",
    )
    actuation_tools: list[str] = Field(
        default_factory=lambda: [
            "calibrate_ultrasonic",
            "tensorrt_compile",
            "export_experience",
            "set_velocity",
        ],
        description=(
            "Tools considered actuation/side-effecting (config-driven, not hardcoded). "
            "`emergency_stop` is intentionally NOT in this default list — refusing "
            "an e-stop call during a safety emergency would defeat its purpose. "
            "`read_encoders` is read-only and stays out of the list as well. "
            "Existing YAML overrides win; this default only changes for clients that "
            "never set the field."
        ),
    )
    expose_actuation_tools: bool = Field(
        False,
        description="If False, actuation_tools are hidden from list_tools and refused",
    )
    resources: MCPResourcesConfig = Field(default_factory=MCPResourcesConfig)
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-tool-call timeout (seconds)",
    )
    rate_limit_rps: float = Field(
        10.0,
        gt=0,
        description="Per-session token-bucket rate limit (requests per second)",
    )
    sample_telemetry_hz: float = Field(
        10.0,
        gt=0,
        le=60.0,
        description="Background sampler rate that pulls TelemetryFrames into MCP buffer",
    )
    circuit_breaker: CircuitBreakerConfig | None = Field(
        None,
        description="Circuit breaker override; falls back to root cfg.circuit_breaker",
    )
    redact_key_pattern: str = Field(
        r"(?i)token|secret|api[_-]?key|password|credential",
        description="Regex (case-insensitive) for keys whose values must be redacted",
    )
    bind_transport: bool = Field(
        False,
        description=(
            "Bind the configured transport via the optional `mcp` SDK. "
            "Defaults to False so unit tests and in-process callers keep "
            "the bridge usable without spinning up a real server. Set "
            "True in deployment YAML (or via MOUSEDROID_MCP__BIND_TRANSPORT=true) "
            "to expose the server over stdio/SSE/streamable_http."
        ),
    )
    smoke_test_poll_rps: float = Field(
        5.0,
        gt=0,
        description="MCP resource polling rate during the rover hardware smoke (RPS)",
    )
    smoke_test_duration_s: float = Field(
        2.0,
        gt=0,
        description="Duration of the MCP-polling-during-actuation smoke window (s)",
    )
    bind_external: bool = Field(
        False,
        description=(
            "Permit binding a non-loopback host (e.g. 0.0.0.0) for cross-host "
            "OpenClaw access. When False, ``host`` other than 127.0.0.1/localhost "
            "fails validation early so an operator does not accidentally expose "
            "the MCP server. Pair with ``transport`` in {sse, streamable_http} "
            "and a non-empty ``MOUSEDROID_MCP_TOKEN`` env var."
        ),
    )

    @field_validator("tools_denylist")
    @classmethod
    def _no_required_in_denylist(cls, v: list[str]) -> list[str]:
        """Reject denylists that include required liveness tools.

        Args:
            v: Proposed denylist.

        Returns:
            The validated denylist.

        Raises:
            ValueError: If a required tool name is included.
        """
        if "health_check" in v:
            msg = "health_check cannot be denied (required liveness signal)"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _require_token_for_remote(self) -> Self:
        """Refuse to enable a non-loopback transport without an auth token.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If MCP is enabled on a non-loopback bind without
                a token in the configured environment variable.
        """
        if not self.enabled:
            return self
        if self.transport == "stdio":
            return self
        if self.host == "127.0.0.1" or self.host == "localhost":
            return self
        import os

        if not os.environ.get(self.auth_token_env_var):
            msg = (
                f"MCP enabled on non-loopback host '{self.host}' requires the "
                f"{self.auth_token_env_var} environment variable to be set"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _bind_transport_only_for_supported(self) -> Self:
        """Validate bind_transport ↔ transport ↔ external-bind interplay.

        Three guards run here so misconfigurations fail fast at config
        load rather than from a background task hours later:

        1. ``bind_transport=true`` requires a known transport string.
        2. Non-loopback ``host`` requires ``bind_external=true`` so an
           operator never exposes the server by accident.
        3. ``bind_external=true`` requires a non-stdio transport (stdio
           has no listener anyway) AND a non-empty token in the
           ``auth_token_env_var`` env var so the listening port can never
           accept unauthenticated requests.

        Returns:
            The validated config instance.

        Raises:
            ValueError: When any of the three guards trip.
        """
        if not self.bind_transport:
            return self
        supported = {"stdio", "sse", "streamable_http"}
        if self.transport not in supported:
            msg = (
                f"mcp.bind_transport=true is only supported with "
                f"mcp.transport in {sorted(supported)}; "
                f"got mcp.transport={self.transport!r}."
            )
            raise ValueError(msg)
        is_loopback = self.host == "127.0.0.1" or self.host == "localhost"
        if not is_loopback and not self.bind_external:
            msg = (
                f"mcp.host={self.host!r} is non-loopback but "
                "mcp.bind_external is False. Set bind_external=true "
                "explicitly to expose the MCP server outside the host."
            )
            raise ValueError(msg)
        if self.bind_external:
            if self.transport == "stdio":
                msg = "mcp.bind_external=true requires a network transport (sse/streamable_http)"
                raise ValueError(msg)
            import os

            if not os.environ.get(self.auth_token_env_var):
                msg = (
                    f"mcp.bind_external=true requires {self.auth_token_env_var} to be set "
                    "in the environment; refusing to bind without a bearer secret."
                )
                raise ValueError(msg)
        return self


class HarnessTrackerConfig(StrictBaseModel):
    """Task-tracker configuration for the agent harness.

    The tracker persists in-memory state of submitted tasks and their
    acceptance predicates; the orchestrator consults it once per tick.
    Disabled by default — enabling is opt-in and adds no work to the
    30 Hz hot loop while ``enabled=False``.
    """

    enabled: bool = Field(
        False,
        description="Enable in-memory task tracker (None=disabled)",
    )
    history_size: int = Field(
        256,
        gt=0,
        description="Bounded deque size for completed-task history",
    )
    default_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Fallback timeout (s) for tasks that do not specify one",
    )
    max_active: int = Field(
        8,
        gt=0,
        description="Hard cap on simultaneously active tasks",
    )


class HarnessJournalConfig(StrictBaseModel):
    """Persistent agent ledger backend selection and tunables."""

    backend: Literal["null", "jsonl", "lmdb"] = Field(
        "null",
        description="Journal backend (null=disabled)",
    )
    path: Path = Field(
        Path("var/harness/journal"),
        description="Journal directory (LMDB) or file path (JSONL)",
    )
    map_size_gb: float = Field(
        1.0,
        gt=0,
        description="LMDB map size cap in GB (LMDB backend only)",
    )
    flush_every_n: int = Field(
        16,
        gt=0,
        description="Flush LMDB transactions every N writes",
    )
    queue_max: int = Field(
        1024,
        gt=0,
        description="Max queued entries; on full, oldest is dropped (warn-log)",
    )


class HarnessHooksConfig(StrictBaseModel):
    """Tick-loop middleware configuration."""

    enabled_hooks: list[str] = Field(
        default_factory=list,
        description="Names of hooks to wire from the registry (empty=no-op)",
    )
    error_policy: Literal["raise", "warn", "swallow"] = Field(
        "warn",
        description="How hook exceptions are handled",
    )
    journal_events: bool = Field(
        True,
        description="When True, default JournalAppendHook is auto-registered",
    )
    fail_fast: bool = Field(
        False,
        description="Abort tick on first hook failure (overrides error_policy)",
    )


class HarnessApprovalConfig(StrictBaseModel):
    """Human-in-the-loop / policy approval configuration."""

    gate: Literal["auto", "cli", "callback", "policy"] = Field(
        "auto",
        description="Approval gate strategy (auto=AutoApproveGate)",
    )
    require_approval_tool_patterns: list[str] = Field(
        default_factory=list,
        description="fnmatch patterns of tool names that require approval",
    )
    require_approval_skill_patterns: list[str] = Field(
        default_factory=list,
        description="fnmatch patterns of skill names that require approval",
    )
    cli_timeout_s: float = Field(
        30.0,
        gt=0,
        description="CLI approval prompt timeout (s)",
    )
    on_timeout: Literal["deny", "approve"] = Field(
        "deny",
        description="Decision when approval times out (default: fail-closed)",
    )
    callback_dotted_path: str | None = Field(
        None,
        description="Dotted path to async callable for callback gate",
    )


class SkillsConfig(StrictBaseModel):
    """Sub-agent / skill registry configuration."""

    enabled: bool = Field(
        False,
        description="Enable skill registry and sub-agent delegation",
    )
    manifest_glob: str = Field(
        "config/skills/*.yaml",
        description="Glob for YAML skill manifests",
    )
    markdown_agent_dirs: list[Path] = Field(
        default_factory=lambda: [Path("src/mousedroid/agents")],
        description="Directories scanned for markdown agent definitions",
    )
    default_system_prompt: str = Field(
        "",
        description="Fallback system prompt when a skill omits its own",
    )
    backend: Literal["llm_gateway", "anthropic", "noop"] = Field(
        "noop",
        description="Default sub-agent backend",
    )


class HarnessConfig(StrictBaseModel):
    """Top-level agent-harness configuration.

    Bundles task tracker, hook registry, journal, approval gate, and skills
    sub-models. Every nested section ships a working default; the entire
    harness is opt-in via ``Settings.harness`` (None=disabled).
    """

    tracker: HarnessTrackerConfig = Field(
        default_factory=_settings_default_factory(HarnessTrackerConfig),
    )
    hooks: HarnessHooksConfig = Field(
        default_factory=_settings_default_factory(HarnessHooksConfig),
    )
    journal: HarnessJournalConfig = Field(
        default_factory=_settings_default_factory(HarnessJournalConfig),
    )
    approval: HarnessApprovalConfig = Field(
        default_factory=_settings_default_factory(HarnessApprovalConfig),
    )
    skills: SkillsConfig = Field(
        default_factory=_settings_default_factory(SkillsConfig),
    )


class OpenClawMemoryConfig(StrictBaseModel):
    """Memory parameters for OpenClaw/MCP integration."""

    episodic_limit: int = Field(50, description="Max episodic events to return per cursor query.")
    semantic_limit: int = Field(10, description="Max semantic memories to retrieve per query.")


class OpenClawPolicyConfig(StrictBaseModel):
    """Policy constraints for OpenClaw sandbox enforcement."""

    openshell_policy_path: Path | None = Field(
        None,
        description="Path to the openshell policy constraints file. If None, static limits apply.",
    )
    max_skills_per_mission: int = Field(
        5,
        ge=1,
        description="Static limit on skill invocations per mission if openshell is missing.",
    )
    allow_actuation: bool = Field(
        True, description="Static limit on whether physical actuation is permitted."
    )
    actuation_skill_names: tuple[str, ...] = Field(
        ("move", "arm", "drive", "actuate"),
        description=(
            "Skill names treated as actuation (blocked when allow_actuation is False). "
            "Operators extend this list via YAML for custom actuator skills."
        ),
    )
    max_tracked_missions: int = Field(
        1000,
        ge=1,
        description=(
            "Maximum number of concurrent mission task IDs tracked in the "
            "in-memory skill-count dict. Oldest entries are evicted when full."
        ),
    )


class OpenClawConfig(StrictBaseModel):
    """OpenClaw integration — multi-channel NL control plane.

    OpenClaw runs on a dedicated Mac mini host and dispatches NL commands
    into MouseDroid either via the REST ``POST /api/v1/mission``
    endpoint or via the MCP server (cross-host SSE / streamable_http).
    Both channels enforce the same prompt-injection envelope, the same
    rate-limit token bucket, and (for actuation skills) the same safety
    gate — wiring described in ``docs/openclaw_skills/README.md``.

    Disabled by default. Existing YAML files load unchanged because the
    ``openclaw`` field on :class:`Settings` defaults to ``None`` and every
    field on this model has a default.
    """

    enabled: bool = Field(
        False,
        description="Enable the OpenClaw control plane (REST + MCP gating)",
    )
    mac_mini_origin: str | None = Field(
        None,
        description=(
            "Origin URL of the OpenClaw host (e.g. https://mini.tail-xxxx.ts.net). "
            "When set AND ``telemetry.cors_origins`` is restrictive (does not "
            "contain '*'), :class:`TelemetryServer` automatically appends this "
            "origin to the CORS allow-list at boot so the OpenClaw dashboard "
            "can hit the REST mission endpoint without operators having to "
            "duplicate the URL in two YAML keys."
        ),
    )
    allowed_channels: tuple[Literal["rest", "mcp"], ...] = Field(
        ("rest", "mcp"),
        description="Channels the dispatcher accepts; others are refused.",
    )
    memory: OpenClawMemoryConfig = Field(
        default_factory=OpenClawMemoryConfig,
        description="Memory access limits for the MCP memory resource.",
    )
    policy: OpenClawPolicyConfig = Field(
        default_factory=OpenClawPolicyConfig,
        description="Sandbox and policy configuration.",
    )
    dm_pairing_required: bool = Field(
        True,
        description=(
            "Mac-mini-side hint: enforce dmPolicy=pairing in OpenClaw config. "
            "Mirrored here so operator docs and integration tests stay in sync."
        ),
    )
    max_command_len: int = Field(
        512,
        gt=0,
        description="Maximum NL command length accepted by the dispatcher.",
    )
    shared_memory_path: Path | None = Field(
        None,
        description=(
            "Filesystem path (Tailscale-shared dir or NFS mount) where the "
            "Phase D MarkdownReplayExporter writes MEMORY.md. None disables "
            "the exporter entirely."
        ),
    )
    mdns_service_name: str = Field(
        "_mousedroid._tcp.local.",
        description="Advisory mDNS service name; Tailscale MagicDNS is preferred.",
    )
    command_dedup_window_s: float = Field(
        5.0,
        gt=0,
        description="In-memory TTL window for idempotency_key dedup on REST.",
    )
    export_every_n_ticks: int = Field(
        600,
        gt=0,
        description=(
            "How often the MEMORY.md exporter is allowed to fire (ticks). "
            "At the default 30 Hz control loop this is one snapshot every 20 s."
        ),
    )
    rest_rate_limit_rps: float = Field(
        2.0,
        gt=0,
        description="POST /api/v1/mission token-bucket refill rate (req/s).",
    )
    rest_rate_limit_burst: int = Field(
        4,
        gt=0,
        description="POST /api/v1/mission token-bucket burst capacity.",
    )
    require_actuation_ack: bool = Field(
        True,
        description=(
            "Skills declared with metadata['actuation']=True require this flag "
            "AND mcp.expose_actuation_tools=true. Defence-in-depth even when "
            "an operator flips one of the two by accident."
        ),
    )
    export_max_entries: int = Field(
        32,
        gt=0,
        description=(
            "Cap on episodic samples included in each MEMORY.md snapshot "
            "(threaded into MarkdownReplayExporter)."
        ),
    )
    export_entry_truncate_chars: int = Field(
        240,
        gt=0,
        description=(
            "Per-entry display cap (chars) in MEMORY.md so large episodic "
            "payloads don't blow the OpenClaw agent's context window."
        ),
    )
