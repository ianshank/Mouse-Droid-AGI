# Specification: NemoClaw Integration

## REQ 1: Live Memory Query
The system must support querying episodic memory and semantic index natively over MCP.
- **Scenario 1**: Given an active mission, when a live query arrives over MCP for episodic samples, the provider returns cursor-paginated samples directly from the running `EpisodicReplay` deque, serialized safely (tensors reduced to statistics).
- **Scenario 2**: When a live query arrives over MCP for semantic retrieval, the provider delegates to `SemanticIndex.retrieve()` via `asyncio.to_thread` and returns the serialized distances.
- **Scenario 3**: Given the system is containerized, when offline replay tools request memory, they use the filesystem volume mounts populated by the `MarkdownReplayExporter`. Both memory paths (file and MCP) can operate simultaneously.

## REQ 2: Transport-Identical Gating
REST and MCP channels must share a common gating envelope.
- **Scenario 1**: Given a skill invocation payload, when the payload arrives via the REST orchestrator (`MissionDispatcher`) or the MCP `ToolBridge`, it is passed through the identical `ApprovalGateProtocol`. The exact same decision is returned in both cases.
- **Scenario 2**: If the MCP SDK is uninstalled, the factory safely returns `None` for the MCP transport, while REST gating continues unaffected.

## REQ 3: Policy Single Source of Truth
`SkillSpec.tool_names` is the definitive source for policy generation.
- **Scenario 1**: When `openshell policy generate` is run, it extracts allowed tools solely from `SkillSpec.tool_names` within the `SkillRegistry`. Any drift between policy and registry fails generation.

## REQ 4: Audit Before Enforce & Live Downgrade (Spike)
The sandbox must support a safe rollout posture.
- **Scenario 1**: When NemoClaw is first enabled, it runs in audit mode. Policy violations are logged but permitted.
- **Scenario 2**: When flipped to enforce mode, policy violations are hard-blocked.
- **Scenario 3**: If sandbox latency exceeds `sandbox_downgrade_sla_minutes`, the system safely falls back to open execution.

## REQ 5: Config Gating
All features must be default-disabled to guarantee backward compatibility.
- **Scenario 1**: When `OpenClawConfig.enabled` is `False` (the default), the factory ensures no new resources, providers, or sandbox policies are instantiated, leaving the system functionally and byte-identical to `9e157f0`.

## REQ 6: Baseline-Referenced Gates
Performance constraints must be enforced against a config schema.
- **Scenario 1**: When a memory query latency exceeds the value specified in `baselines.yaml`, an explicit `baseline_missing` failure is logged.
